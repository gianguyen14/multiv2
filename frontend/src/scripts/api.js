(() => {
  "use strict";

  const TOP_K = 100;
  const MAX_IMAGE_BYTES = 15 * 1024 * 1024;
  const SUPPORTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

  class ApiError extends Error {
    constructor(message, status = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  function readableDetail(detail) {
    if (typeof detail === "string" && detail.trim()) {
      return detail.trim();
    }
    if (detail !== undefined && detail !== null) {
      try {
        return JSON.stringify(detail);
      } catch (error) {
        return String(detail);
      }
    }
    return "";
  }

  async function parseResponse(response) {
    const raw = await response.text();
    let data = null;

    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (error) {
        if (!response.ok) {
          throw new ApiError(`HTTP ${response.status}: ${raw.slice(0, 500)}`, response.status);
        }
        throw new ApiError("Backend returned invalid JSON.", response.status);
      }
    }

    if (!response.ok) {
      const detail = data && typeof data === "object" ? readableDetail(data.detail) : "";
      throw new ApiError(detail || `Backend request failed with HTTP ${response.status}.`, response.status);
    }

    if (!data || typeof data !== "object" || !Array.isArray(data.results)) {
      throw new ApiError("Backend response is malformed: expected a results array.", response.status);
    }

    if (data.results.some((item) => !item || typeof item !== "object" || Array.isArray(item))) {
      throw new ApiError("Backend response is malformed: every result must be an object.", response.status);
    }

    return data;
  }

  async function requestJson(requestFactory) {
    try {
      const response = await requestFactory();
      return await parseResponse(response);
    } catch (error) {
      if (error instanceof ApiError) {
        throw error;
      }
      throw new ApiError("Network error: unable to reach the backend.");
    }
  }

  function searchText({ query, queryType, events }) {
    let body;

    if (queryType === "trake") {
      body = {
        query_type: "trake",
        events: Array.from(events),
        top_k: TOP_K,
      };
    } else if (queryType === "kis" || queryType === "qa") {
      body = {
        query: String(query),
        query_type: queryType,
        top_k: TOP_K,
      };
    } else {
      return Promise.reject(new ApiError("Unsupported search mode."));
    }

    return requestJson(() => fetch('/api/search', {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    }));
  }

  function searchImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    return requestJson(() => fetch('/api/search/image?top_k=100', {
      method: "POST",
      body: formData,
    }));
  }

  function validateImageFile(file) {
    if (!file) {
      return "Please select an image file.";
    }
    if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
      return "Please choose a JPEG, PNG, or WebP image.";
    }
    if (file.size > MAX_IMAGE_BYTES) {
      return "The selected image exceeds the 15 MiB client-side limit.";
    }
    return "";
  }

  window.ChiLangApi = Object.freeze({
    ApiError,
    MAX_IMAGE_BYTES,
    SUPPORTED_IMAGE_TYPES,
    TOP_K,
    searchImage,
    searchText,
    validateImageFile,
  });
})();
