#!/usr/bin/env bash
# Run final runtime smoke checks without changing DB, model, or configuration.
set -u

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:3000}"
SEARCH_URL="$BACKEND_URL/api/search"
VIDEO_ID="L21_V013"
TMP_DIR="${TMPDIR:-/tmp}/finals_smoke.$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

check() {
  name="$1"; status="$2"; latency="$3"
  if [ "$status" = "0" ]; then
    printf '%s PASS %sms\n' "$name" "$latency"
  else
    printf '%s FAIL %sms\n' "$name" "$latency"
  fi
}

request() {
  name="$1"; url="$2"; expected="$3"; payload="${4:-}"
  body="$TMP_DIR/$name.body"; meta="$TMP_DIR/$name.meta"
  if [ -n "$payload" ]; then
    curl -sS --max-time 180 -H 'Content-Type: application/json' \
      --data "$payload" -o "$body" -w '%{http_code} %{time_total}' "$url" >"$meta" 2>/dev/null || :
  else
    curl -sS --max-time 30 -o "$body" -w '%{http_code} %{time_total}' "$url" >"$meta" 2>/dev/null || :
  fi
  code=$(awk '{print $1}' "$meta" 2>/dev/null || printf '000')
  seconds=$(awk '{print $2}' "$meta" 2>/dev/null || printf '0')
  latency=$(awk -v s="$seconds" 'BEGIN {printf "%.0f", s*1000}')
  [ "${code:-000}" = "$expected" ] && check "$name" 0 "$latency" || check "$name" 1 "$latency"
}

# Backend health
read -r code seconds <<EOF
$(curl -sS --max-time 30 -o "$TMP_DIR/health.body" -w '%{http_code} %{time_total}' "$BACKEND_URL/health/ready" 2>/dev/null || printf '000 0')
EOF
latency=$(awk -v s="${seconds:-0}" 'BEGIN {printf "%.0f", s*1000}')
[ "${code:-000}" = 200 ] && check BACKEND_HEALTH 0 "$latency" || check BACKEND_HEALTH 1 "$latency"

# Frontend HTTP
read -r code seconds <<EOF
$(curl -sS --max-time 30 -o "$TMP_DIR/frontend.body" -w '%{http_code} %{time_total}' "$FRONTEND_URL/" 2>/dev/null || printf '000 0')
EOF
latency=$(awk -v s="${seconds:-0}" 'BEGIN {printf "%.0f", s*1000}')
[ "${code:-000}" = 200 ] && check FRONTEND_HTTP 0 "$latency" || check FRONTEND_HTTP 1 "$latency"

# Real KIS, QA, and known-good two-event TRAKE request.
request KIS "$SEARCH_URL" 200 '{"query":"một người phụ nữ đang nấu ăn trong chảo","query_type":"kis","top_k":3,"query_refine":false,"rerank":true}'
request QA "$SEARCH_URL" 200 '{"query":"Hội chợ game năm nay quy tụ bao nhiêu hãng game?","query_type":"qa","top_k":3,"query_refine":false,"rerank":true}'
request TRAKE "$SEARCH_URL" 200 '{"query_type":"trake","events":["cháy rừng bạch đàn","diện tích rừng bị cháy"],"top_k":10,"query_refine":false,"rerank":true}'

# Video preview and one byte-range request for the known-good TRAKE video.
read -r code seconds <<EOF
$(curl -sS --max-time 30 -o "$TMP_DIR/video.body" -w '%{http_code} %{time_total}' "$BACKEND_URL/api/video/$VIDEO_ID" 2>/dev/null || printf '000 0')
EOF
latency=$(awk -v s="${seconds:-0}" 'BEGIN {printf "%.0f", s*1000}')
[ "${code:-000}" = 200 ] && check VIDEO_PREVIEW 0 "$latency" || check VIDEO_PREVIEW 1 "$latency"
read -r code seconds range <<EOF
$(curl -sS --max-time 30 -r 0-1023 -D "$TMP_DIR/range.headers" -o "$TMP_DIR/range.body" -w '%{http_code} %{time_total} %{size_download}' "$BACKEND_URL/api/video/$VIDEO_ID" 2>/dev/null || printf '000 0 0')
EOF
latency=$(awk -v s="${seconds:-0}" 'BEGIN {printf "%.0f", s*1000}')
[ "${code:-000}" = 206 ] && check VIDEO_RANGE 0 "$latency" || check VIDEO_RANGE 1 "$latency"
