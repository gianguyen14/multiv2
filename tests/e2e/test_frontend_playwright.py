from __future__ import annotations

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Page, Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        return


@pytest.fixture(scope="session")
def frontend_url():
    handler = partial(QuietHandler, directory=str(FRONTEND))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    context.add_init_script(
        """
        Object.defineProperty(navigator, 'clipboard', {
          configurable: true,
          value: {
            writeText: async value => { window.__copiedSubmission = value; }
          }
        });
        """
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page._page_errors_for_test = errors
    try:
        yield page
    finally:
        context.close()


def _assert_no_page_errors(page: Page):
    assert page._page_errors_for_test == []


def _fulfill_json(route: Route, payload: dict, status: int = 200):
    route.fulfill(
        status=status,
        content_type="application/json",
        body=json.dumps(payload),
    )


def test_kis_keyboard_submit_render_and_copy(page: Page, frontend_url: str):
    captured: list[dict] = []

    def handle_search(route: Route):
        captured.append(route.request.post_data_json)
        _fulfill_json(
            route,
            {
                "results": [
                    {
                        "video_id": "V001",
                        "frame_id": 123,
                        "timestamp_seconds": 4.1,
                        "score": 0.91,
                        "visual_score": 0.8,
                        "ocr_score": 0.2,
                        "asr_score": 0.1,
                        "image_url": "/api/frames/V001/000000123.jpg",
                    }
                ]
            },
        )

    page.route("**/api/search", handle_search)
    page.route("**/api/frames/**", lambda route: route.fulfill(status=204))
    page.goto(frontend_url)

    assert page.locator("#status").inner_text() == "ready"
    page.locator("#query-type").focus()
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.id") == "query"

    page.locator("#query").fill("red truck")
    page.keyboard.press("Control+Enter")
    page.locator("#status").filter(has_text="1 results").wait_for()

    assert captured[-1]["query_type"] == "kis"
    assert captured[-1]["query"] == "red truck"
    card = page.locator("article.card")
    assert card.count() == 1
    assert "#1 V001" in card.inner_text()
    assert "frame 123" in card.inner_text()
    assert "score 0.9100" in card.inner_text()

    card.get_by_role("button", name="Copy submission").click()
    assert page.evaluate("window.__copiedSubmission") == "V001,123"
    _assert_no_page_errors(page)


def test_qa_renders_answer_evidence_as_text_not_html(page: Page, frontend_url: str):
    def handle_search(route: Route):
        assert route.request.post_data_json["query_type"] == "qa"
        _fulfill_json(
            route,
            {
                "results": [
                    {
                        "video_id": "V_QA",
                        "frame_id": 9,
                        "score": 0.7,
                        "answer": "<img src=x onerror=alert(1)>",
                        "evidence_sources": ["visual", "ocr"],
                        "image_url": "/api/frames/V_QA/000000009.jpg",
                    }
                ]
            },
        )

    page.route("**/api/search", handle_search)
    page.route("**/api/frames/**", lambda route: route.fulfill(status=204))
    page.goto(frontend_url)
    page.locator("#query-type").select_option("qa")
    page.locator("#query").fill("What is shown?")
    page.locator("#submit-btn").click()
    page.locator("#status").filter(has_text="1 results").wait_for()

    card = page.locator("article.card")
    assert "answer: <img src=x onerror=alert(1)>" in card.inner_text()
    assert "evidence: visual, ocr" in card.inner_text()
    assert card.locator("img").count() == 1
    _assert_no_page_errors(page)


def test_trake_events_only_submission_and_mode_switching(page: Page, frontend_url: str):
    captured: list[dict] = []

    def handle_search(route: Route):
        captured.append(route.request.post_data_json)
        _fulfill_json(
            route,
            {
                "results": [
                    {
                        "video_id": "V_TRAKE",
                        "frame_id": 10,
                        "score": 1.0,
                        "image_url": "/api/frames/V_TRAKE/000000010.jpg",
                    }
                ]
            },
        )

    page.route("**/api/search", handle_search)
    page.route("**/api/frames/**", lambda route: route.fulfill(status=204))
    page.goto(frontend_url)
    page.locator("#query-type").select_option("trake")

    assert page.locator("#events").is_visible()
    page.locator("#events").fill("person enters\nperson sits\nperson leaves")
    page.locator("#submit-btn").click()
    page.locator("#status").filter(has_text="1 results").wait_for()

    assert captured[-1]["query_type"] == "trake"
    assert captured[-1]["events"] == ["person enters", "person sits", "person leaves"]
    assert captured[-1]["query"] == ""

    page.locator("#query-type").select_option("kis")
    assert page.locator("#events").is_hidden()
    _assert_no_page_errors(page)


def test_image_mode_uploads_file_and_restores_text_mode(page: Page, frontend_url: str):
    calls: list[tuple[str, str, int]] = []

    def handle_image(route: Route):
        body = route.request.post_data_buffer or b""
        calls.append((route.request.method, route.request.url, len(body)))
        _fulfill_json(
            route,
            {
                "results": [
                    {
                        "video_id": "V_IMAGE",
                        "frame_id": 77,
                        "score": 0.88,
                        "image_url": "/api/frames/V_IMAGE/000000077.jpg",
                    }
                ]
            },
        )

    page.route("**/api/search/image?top_k=100", handle_image)
    page.route("**/api/frames/**", lambda route: route.fulfill(status=204))
    page.goto(frontend_url)
    page.locator("#query-type").select_option("image")

    assert page.locator("#query").is_hidden()
    assert page.locator("#image-file").is_visible()
    assert page.locator("#image-file").get_attribute("required") is not None

    page.locator("#image-file").set_input_files(
        {
            "name": "query.png",
            "mimeType": "image/png",
            "buffer": b"\x89PNG\r\n\x1a\nfrontend-smoke",
        }
    )
    page.locator("#submit-btn").click()
    page.locator("#status").filter(has_text="1 results").wait_for()

    assert calls and calls[-1][0] == "POST"
    assert "/api/search/image?top_k=100" in calls[-1][1]
    assert calls[-1][2] > 0
    assert "V_IMAGE" in page.locator("article.card").inner_text()

    page.locator("#query-type").select_option("kis")
    assert page.locator("#query").is_visible()
    assert page.locator("#image-file").is_hidden()
    assert page.locator("#query").get_attribute("required") is not None
    _assert_no_page_errors(page)


def test_empty_results_and_server_error_have_explicit_ui_states(page: Page, frontend_url: str):
    calls = 0

    def handle_search(route: Route):
        nonlocal calls
        calls += 1
        if calls == 1:
            _fulfill_json(route, {"results": []})
        else:
            route.fulfill(status=503, content_type="text/plain", body="backend unavailable")

    page.route("**/api/search", handle_search)
    page.goto(frontend_url)
    page.locator("#query").fill("first")
    page.locator("#submit-btn").click()
    page.locator("#status").filter(has_text="0 results").wait_for()
    assert page.locator("#results").inner_text() == "No candidates"

    page.locator("#query").fill("second")
    page.locator("#submit-btn").click()
    page.locator("#status").filter(has_text="error").wait_for()
    assert "backend unavailable" in page.locator("#results").inner_text()
    _assert_no_page_errors(page)
