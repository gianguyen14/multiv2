(() => {
  "use strict";

  const Api = window.ChiLangApi;
  const Shortcuts = window.ChiLangShortcuts;

  const state = {
    mode: "kis",
    status: "idle",
    events: [""],
    selectedImage: null,
    previewUrl: null,
    results: [],
    latencyMs: null,
    latencyTimerId: null,
    latencyStartedAt: null,
    queryMetrics: null,
    requestVersion: 0,
    lastResultMode: null,
    lastTrakeEventCount: null,
    selectedResultIndex: -1,
  };

  const form = document.querySelector("#search-form");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const queryPanel = document.querySelector("#query-panel");
  const queryLabel = document.querySelector("#query-label");
  const queryInput = document.querySelector("#query");
  const trakePanel = document.querySelector("#trake-panel");
  const eventList = document.querySelector("#event-list");
  const addEventButton = document.querySelector("#add-event");
  const imagePanel = document.querySelector("#image-panel");
  const imageFile = document.querySelector("#image-file");
  const imageFilename = document.querySelector("#image-filename");
  const imagePreviewFrame = document.querySelector("#image-preview-frame");
  const imagePreview = document.querySelector("#image-preview");
  const clearImageButton = document.querySelector("#clear-image");
  const submitButton = document.querySelector("#submit-btn");
  const statusPill = document.querySelector("#status-pill");
  const statusMessage = document.querySelector("#status-message");
  const resultCount = document.querySelector("#result-count");
  const latencyValue = document.querySelector("#latency-value");
  const results = document.querySelector("#results");
  const metricsPanel = document.querySelector("#metrics-panel");
  const metricsOutput = document.querySelector("#query-metrics");

  const MODE_LABELS = {
    kis: "KIS",
    qa: "Q&A",
    trake: "TRAKE",
    image: "Image Search",
  };

  const UI_STATES = new Set(["idle", "loading", "success", "empty", "error"]);

  function hasOwn(object, key) {
    return Object.prototype.hasOwnProperty.call(object, key);
  }

  function createNode(tagName, className, text) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }
    if (text !== undefined) {
      element.textContent = String(text);
    }
    return element;
  }

  function setBusy(isBusy) {
    submitButton.disabled = isBusy;
    submitButton.setAttribute("aria-busy", String(isBusy));
    addEventButton.disabled = isBusy || state.events.length >= 20;
    clearImageButton.disabled = isBusy || !state.selectedImage;
  }

  function setStatus(kind, message) {
    if (!UI_STATES.has(kind)) {
      throw new Error(`Unknown UI state: ${kind}`);
    }
    state.status = kind;
    statusPill.dataset.state = kind;
    statusPill.textContent = kind.toUpperCase();
    statusMessage.textContent = message;
  }

  function stopLatencyTimer() {
    if (state.latencyTimerId !== null) {
      window.clearInterval(state.latencyTimerId);
    }
    state.latencyTimerId = null;
    state.latencyStartedAt = null;
  }

  function startLatencyTimer(startedAt) {
    stopLatencyTimer();
    state.latencyStartedAt = startedAt;

    const update = () => {
      if (state.status !== "loading" || state.latencyStartedAt === null) {
        return;
      }
      latencyValue.textContent = `${Math.round(performance.now() - state.latencyStartedAt)} ms`;
    };

    update();
    state.latencyTimerId = window.setInterval(update, 50);
  }

  function clearMetrics() {
    state.queryMetrics = null;
    metricsOutput.textContent = "";
    metricsPanel.hidden = true;
  }

  function renderMetrics(metrics) {
    if (metrics === undefined || metrics === null) {
      clearMetrics();
      return;
    }

    state.queryMetrics = metrics;
    try {
      metricsOutput.textContent = JSON.stringify(metrics, null, 2);
    } catch (error) {
      metricsOutput.textContent = String(metrics);
    }
    metricsPanel.hidden = false;
  }

  function resetOutput(message = "Run a query to inspect ranked evidence.") {
    stopLatencyTimer();
    state.results = [];
    state.latencyMs = null;
    state.lastResultMode = null;
    state.lastTrakeEventCount = null;
    state.selectedResultIndex = -1;
    clearMetrics();
    resultCount.textContent = "0 results";
    latencyValue.textContent = "—";
    results.className = "results result-empty";
    results.replaceChildren(createNode("p", "", message));
    setStatus("idle", "Ready");
  }

  function setMode(mode, shouldFocus = false) {
    if (!Object.prototype.hasOwnProperty.call(MODE_LABELS, mode)) {
      return;
    }

    const cancelledRequest = state.status === "loading";
    if (cancelledRequest) {
      state.requestVersion += 1;
      stopLatencyTimer();
      setBusy(false);
    }

    const changed = state.mode !== mode;
    state.mode = mode;

    modeButtons.forEach((button) => {
      const active = button.dataset.mode === mode;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    queryPanel.hidden = !(mode === "kis" || mode === "qa");
    trakePanel.hidden = mode !== "trake";
    imagePanel.hidden = mode !== "image";
    queryInput.required = mode === "kis" || mode === "qa";

    if (mode === "kis") {
      queryLabel.textContent = "KIS query";
      queryInput.placeholder = "Describe the target scene or event";
    } else if (mode === "qa") {
      queryLabel.textContent = "Question";
      queryInput.placeholder = "Ask a question about the video content";
    }

    submitButton.textContent = `Search ${MODE_LABELS[mode]}`;

    if (changed || cancelledRequest) {
      resetOutput(`${MODE_LABELS[mode]} mode selected.`);
    }

    if (shouldFocus) {
      focusPrimaryInput();
    }
  }

  function renderEvents(focusIndex = null) {
    const rows = state.events.map((value, index) => {
      const row = createNode("li", "event-row");
      const number = createNode("span", "event-number", index + 1);
      number.setAttribute("aria-hidden", "true");

      const input = createNode("input", "event-input");
      input.type = "text";
      input.maxLength = 2000;
      input.value = value;
      input.placeholder = `Event ${index + 1}`;
      input.setAttribute("aria-label", `TRAKE event ${index + 1}`);
      input.dataset.eventIndex = String(index);
      input.addEventListener("input", () => {
        state.events[index] = input.value;
      });

      const remove = createNode("button", "icon-button", "−");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove TRAKE event ${index + 1}`);
      remove.disabled = state.status === "loading";
      remove.addEventListener('click', () => {
        removeTrakeEvent(index);
      });

      row.append(number, input, remove);
      return row;
    });

    eventList.replaceChildren(...rows);
    addEventButton.disabled = state.events.length >= 20 || state.status === "loading";

    if (focusIndex !== null) {
      const input = eventList.querySelector(`[data-event-index="${focusIndex}"]`);
      if (input) {
        input.focus();
      }
    }
  }

  function addTrakeEvent() {
    if (state.status === "loading") {
      return;
    }
    if (state.events.length >= 20) {
      setStatus("error", "TRAKE supports at most 20 events.");
      return;
    }
    state.events.push("");
    renderEvents(state.events.length - 1);
  }

  function removeTrakeEvent(index) {
    if (state.status === "loading") {
      return;
    }
    if (state.events.length === 0 || index < 0 || index >= state.events.length) {
      return;
    }
    state.events.splice(index, 1);
    if (state.events.length === 0) {
      renderEvents();
      addEventButton.focus();
      return;
    }
    renderEvents(Math.max(0, Math.min(index, state.events.length - 1)));
  }

  function removeLastTrakeEvent() {
    removeTrakeEvent(state.events.length - 1);
  }

  function clearImageSelection() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
    }
    state.previewUrl = null;
    state.selectedImage = null;
    imageFile.value = "";
    imageFilename.textContent = "No image selected";
    imagePreview.removeAttribute("src");
    imagePreviewFrame.hidden = true;
    clearImageButton.disabled = true;
  }

  function selectImage(file) {
    const validationMessage = Api.validateImageFile(file);
    if (validationMessage) {
      clearImageSelection();
      setStatus("error", validationMessage);
      return;
    }

    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
    }

    state.selectedImage = file;
    state.previewUrl = URL.createObjectURL(file);
    imageFilename.textContent = `${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MiB`;
    imagePreview.src = state.previewUrl;
    imagePreview.alt = `Selected image preview: ${file.name}`;
    imagePreviewFrame.hidden = false;
    clearImageButton.disabled = false;
    setStatus("idle", "Image ready for search.");
  }

  function focusPrimaryInput() {
    if (state.mode === "trake") {
      const firstEvent = eventList.querySelector(".event-input");
      if (firstEvent) {
        firstEvent.focus();
      } else {
        addEventButton.focus();
      }
      return;
    }
    if (state.mode === "image") {
      imageFile.focus();
      return;
    }
    queryInput.focus();
  }

  function formatNumber(value, digits = 4) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : String(value);
  }

  function appendMetaItem(container, label, value) {
    const item = createNode("div", "meta-item");
    item.append(
      createNode("span", "meta-label", label),
      createNode("span", "meta-value", value)
    );
    container.append(item);
  }

  function appendOptionalBlock(container, label, value) {
    const block = createNode("div", "optional-block");
    block.append(createNode("strong", "", label), createNode("p", "", value));
    container.append(block);
  }

  function primitiveSubmissionValue(value) {
    return typeof value === "string"
      || (typeof value === "number" && Number.isFinite(value));
  }

  function integerFrameValue(value) {
    if (typeof value === "number") {
      return Number.isInteger(value) && value >= 0;
    }
    return typeof value === "string" && /^\d+$/.test(value);
  }

  function csvEscape(value) {
    const text = String(value);
    if (/[",\r\n]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function buildSubmissionRow(item, mode, context) {
    if (!primitiveSubmissionValue(item.video_id) || String(item.video_id).length === 0) {
      return { row: "", error: "Cannot copy submission: missing video_id." };
    }
    if (String(item.video_id).toLowerCase().endsWith(".mp4")) {
      return { row: "", error: "Cannot copy submission: video_id contains .mp4." };
    }

    if (mode === "trake") {
      if (!Array.isArray(item.frame_ids) || item.frame_ids.length === 0) {
        return { row: "", error: "Cannot copy submission: incomplete backend TRAKE result." };
      }
      if (context.eventCount !== null && item.frame_ids.length !== context.eventCount) {
        return { row: "", error: "Cannot copy submission: TRAKE frame sequence length does not match the submitted events." };
      }
      if (!item.frame_ids.every(integerFrameValue)) {
        return { row: "", error: "Cannot copy submission: TRAKE frame IDs must be non-negative integers." };
      }
      return {
        row: [item.video_id, ...item.frame_ids].map(csvEscape).join(","),
        error: "",
      };
    }

    if (!integerFrameValue(item.frame_id)) {
      return { row: "", error: "Cannot copy submission: frame_id must be a non-negative integer." };
    }

    if (mode === "qa") {
      if (!hasOwn(item, "answer") || item.answer === null || item.answer === undefined) {
        return { row: "", error: "Cannot copy submission: backend result has no answer." };
      }
      const answerText = String(item.answer);
      if (!answerText.trim()) {
        return { row: "", error: "Cannot copy submission: backend answer is empty." };
      }
      if (answerText.length > 100) {
        return { row: "", error: "Cannot copy submission: answer exceeds the 100-character competition limit." };
      }
      return {
        row: [item.video_id, item.frame_id, item.answer].map(csvEscape).join(","),
        error: "",
      };
    }

    if (mode === "kis") {
      return {
        row: [item.video_id, item.frame_id].map(csvEscape).join(","),
        error: "",
      };
    }

    return { row: "", error: "Submission copy is unavailable for Image Search." };
  }

  async function copySubmission(item, mode, context) {
    const submission = buildSubmissionRow(item, mode, context);
    if (!submission.row) {
      setStatus("error", submission.error);
      return;
    }
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
      setStatus("error", "Clipboard API is unavailable in this browser context.");
      return;
    }

    try {
      await navigator.clipboard.writeText(submission.row);
      setStatus("success", "Submission row copied.");
    } catch (error) {
      setStatus("error", "Could not write to the clipboard.");
    }
  }

  function setSelectedResultIndex(index, options = {}) {
    const cards = Array.from(results.querySelectorAll(".result-card"));
    if (!cards.length) {
      state.selectedResultIndex = -1;
      return false;
    }

    const boundedIndex = Math.max(0, Math.min(index, cards.length - 1));
    state.selectedResultIndex = boundedIndex;

    cards.forEach((card, cardIndex) => {
      const selected = cardIndex === boundedIndex;
      card.classList.toggle("is-selected", selected);
      card.tabIndex = selected ? 0 : -1;
      if (selected) {
        card.setAttribute("aria-current", "true");
      } else {
        card.removeAttribute("aria-current");
      }
    });

    const selectedCard = cards[boundedIndex];
    if (options.focus) {
      selectedCard.focus({ preventScroll: true });
    }
    if (options.scroll) {
      selectedCard.scrollIntoView({ block: "nearest", inline: "nearest" });
    }
    return true;
  }

  function moveResultSelection(delta) {
    if (!state.results.length) {
      return false;
    }
    const current = state.selectedResultIndex >= 0 ? state.selectedResultIndex : 0;
    return setSelectedResultIndex(current + delta, { focus: true, scroll: true });
  }

  function copySelectedResult() {
    if (
      state.selectedResultIndex < 0
      || state.selectedResultIndex >= state.results.length
      || !state.lastResultMode
    ) {
      setStatus("error", "No result is selected.");
      return false;
    }

    const item = state.results[state.selectedResultIndex];
    copySubmission(item, state.lastResultMode, { eventCount: state.lastTrakeEventCount });
    return true;
  }

  function createFrameMedia(item, rank) {
    const shell = createNode("div", "frame-shell");
    shell.append(createNode("span", "rank-badge", `#${rank}`));
    const placeholder = createNode("div", "frame-placeholder", "Frame image unavailable");

    if (typeof item.image_url === "string" && item.image_url.trim()) {
      const image = createNode("img");
      image.src = item.image_url;
      image.alt = `Candidate frame for ${String(item.video_id ?? "unknown video")}, frame ${String(item.frame_id ?? "unknown")}`;
      image.loading = "lazy";
      image.decoding = "async";
      image.addEventListener("error", () => {
        image.remove();
        placeholder.hidden = false;
      });
      placeholder.hidden = true;
      shell.append(image, placeholder);
    } else {
      shell.append(placeholder);
    }

    return shell;
  }

  function createResultCard(item, index, mode, context) {
    const card = createNode("article", "result-card");
    card.dataset.resultIndex = String(index);
    card.tabIndex = -1;
    card.addEventListener("click", () => {
      setSelectedResultIndex(index, { scroll: false });
    });
    card.append(createFrameMedia(item, index + 1));

    const meta = createNode("div", "result-meta");
    meta.append(createNode("div", "result-title", hasOwn(item, "video_id") ? item.video_id : "Missing video_id"));

    const core = createNode("div", "meta-grid");
    appendMetaItem(core, "Frame", hasOwn(item, "frame_id") ? item.frame_id : "Unavailable");
    appendMetaItem(core, "Score", hasOwn(item, "score") ? formatNumber(item.score, 4) : "Unavailable");

    if (hasOwn(item, "timestamp_seconds") && item.timestamp_seconds !== null && item.timestamp_seconds !== undefined) {
      appendMetaItem(core, "Timestamp", `${formatNumber(item.timestamp_seconds, 3)} s`);
    }

    meta.append(core);

    const scoreFields = [
      ["visual_score", "Visual"],
      ["ocr_score", "OCR"],
      ["asr_score", "ASR"],
    ];
    const scoreList = createNode("div", "score-list");
    scoreFields.forEach(([field, label]) => {
      if (hasOwn(item, field) && item[field] !== null && item[field] !== undefined) {
        scoreList.append(createNode("span", "score-chip", `${label} ${formatNumber(item[field], 4)}`));
      }
    });
    if (scoreList.childNodes.length) {
      meta.append(scoreList);
    }

    if (mode === "trake" && Array.isArray(item.frame_ids)) {
      appendOptionalBlock(meta, "TRAKE frames", item.frame_ids.join(" → "));
    }

    if (hasOwn(item, "answer") && item.answer !== null && item.answer !== undefined) {
      appendOptionalBlock(meta, "Answer", item.answer);
    }

    if (hasOwn(item, "evidence_sources") && item.evidence_sources !== null && item.evidence_sources !== undefined) {
      const evidence = Array.isArray(item.evidence_sources)
        ? item.evidence_sources.join(", ")
        : String(item.evidence_sources);
      appendOptionalBlock(meta, "Evidence", evidence);
    }

    if (mode === "kis" || mode === "qa" || mode === "trake") {
      const actions = createNode("div", "card-actions");
      const copy = createNode("button", "secondary-button", "Copy submission");
      copy.type = "button";
      const submission = buildSubmissionRow(item, mode, context);
      copy.disabled = !submission.row;
      if (submission.error) {
        copy.title = submission.error;
      }
      copy.addEventListener('click', () => {
        copySubmission(item, mode, context);
      });
      actions.append(copy);
      meta.append(actions);
    }

    card.append(meta);
    return card;
  }

  function renderResults(items, mode, context) {
    state.results = items;
    state.selectedResultIndex = -1;
    resultCount.textContent = `${items.length} result${items.length === 1 ? "" : "s"}`;

    if (!items.length) {
      results.className = "results result-empty";
      results.replaceChildren(createNode("p", "", "The backend returned no candidates."));
      return;
    }

    results.className = "results result-grid";
    results.replaceChildren(
      ...items.map((item, index) => createResultCard(item, index, mode, context))
    );
    setSelectedResultIndex(0, { focus: true, scroll: false });
  }

  function validateCurrentInput() {
    if (state.mode === "kis" || state.mode === "qa") {
      const query = queryInput.value.trim();
      if (!query) {
        return { error: "Enter a query before searching.", query: "", events: [] };
      }
      return { error: "", query, events: [] };
    }

    if (state.mode === "trake") {
      const events = state.events.map((event) => event.trim());
      if (!events.length) {
        return { error: "Add at least one TRAKE event before searching.", query: "", events: [] };
      }
      if (events.some((event) => !event)) {
        return { error: "Every TRAKE event must contain text.", query: "", events: [] };
      }
      return { error: "", query: "", events };
    }

    if (state.mode === "image") {
      const error = Api.validateImageFile(state.selectedImage);
      return { error, query: "", events: [] };
    }

    return { error: "Unsupported search mode.", query: "", events: [] };
  }

  async function submitSearch() {
    if (state.status === "loading") {
      return;
    }

    const input = validateCurrentInput();
    if (input.error) {
      setStatus("error", input.error);
      focusPrimaryInput();
      return;
    }

    const modeAtRequest = state.mode;
    const eventCount = modeAtRequest === "trake" ? input.events.length : null;
    const requestVersion = state.requestVersion + 1;
    state.requestVersion = requestVersion;

    setBusy(true);
    setStatus("loading", `Searching ${MODE_LABELS[modeAtRequest]}…`);
    results.className = "results result-empty";
    results.replaceChildren(createNode("p", "loading-indicator", "Searching"));
    resultCount.textContent = "Searching…";
    latencyValue.textContent = "—";
    clearMetrics();

    const startedAt = performance.now();
    startLatencyTimer(startedAt);

    try {
      const data = modeAtRequest === "image"
        ? await Api.searchImage(state.selectedImage)
        : await Api.searchText({
            query: input.query,
            queryType: modeAtRequest,
            events: input.events,
          });

      if (requestVersion !== state.requestVersion) {
        return;
      }

      state.latencyMs = performance.now() - startedAt;
      stopLatencyTimer();
      state.lastResultMode = modeAtRequest;
      state.lastTrakeEventCount = eventCount;
      latencyValue.textContent = `${Math.round(state.latencyMs)} ms`;

      renderResults(data.results, modeAtRequest, { eventCount });
      renderMetrics(data.query_metrics);

      if (data.results.length === 0) {
        setStatus("empty", `No ${MODE_LABELS[modeAtRequest]} results.`);
      } else {
        setStatus("success", `${data.results.length} result${data.results.length === 1 ? "" : "s"} received.`);
      }
    } catch (error) {
      if (requestVersion !== state.requestVersion) {
        return;
      }

      state.latencyMs = performance.now() - startedAt;
      stopLatencyTimer();
      latencyValue.textContent = `${Math.round(state.latencyMs)} ms`;
      resultCount.textContent = "0 results";
      results.className = "results result-empty";
      results.replaceChildren(createNode("p", "", error && error.message ? error.message : "Search request failed."));
      setStatus("error", error && error.message ? error.message : "Search request failed.");
    } finally {
      if (requestVersion === state.requestVersion) {
        setBusy(false);
      }
    }
  }

  modeButtons.forEach((button) => {
    button.addEventListener('click', () => {
      setMode(button.dataset.mode, true);
    });
  });

  addEventButton.addEventListener('click', () => {
    addTrakeEvent();
  });

  imageFile.addEventListener("change", () => {
    const file = imageFile.files && imageFile.files[0] ? imageFile.files[0] : null;
    if (!file) {
      clearImageSelection();
      return;
    }
    selectImage(file);
  });

  clearImageButton.addEventListener('click', () => {
    clearImageSelection();
    setStatus("idle", "Image selection cleared.");
    imageFile.focus();
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitSearch();
  });

  window.addEventListener("beforeunload", () => {
    stopLatencyTimer();
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
    }
  });

  renderEvents();
  setMode("kis");
  resetOutput();

  Shortcuts.install({
    addTrakeEvent,
    copySelectedResult,
    focusPrimaryInput,
    getMode: () => state.mode,
    moveResultSelection,
    removeLastTrakeEvent,
    setMode: (mode) => setMode(mode, true),
    submit: () => form.requestSubmit(),
  });
})();
