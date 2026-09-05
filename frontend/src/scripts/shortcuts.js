(() => {
  "use strict";

  const MODE_SHORTCUTS = {
    "1": "kis",
    "2": "qa",
    "3": "trake",
    "4": "image",
  };

  function isInteractiveTarget(target) {
    if (!(target instanceof Element)) {
      return false;
    }
    const tag = target.tagName;
    return target.isContentEditable
      || tag === "INPUT"
      || tag === "TEXTAREA"
      || tag === "SELECT"
      || tag === "BUTTON";
  }

  function isTextEntryTarget(target) {
    if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement)) {
      return false;
    }
    if (target instanceof HTMLTextAreaElement) {
      return true;
    }
    return target.type === "text" || target.type === "search";
  }

  function install(actions) {
    document.addEventListener("keydown", (event) => {
      if (event.isComposing) {
        return;
      }

      const target = event.target;
      const interactive = isInteractiveTarget(target);

      if (
        event.key === "Enter"
        && !event.altKey
        && !event.ctrlKey
        && !event.metaKey
        && !event.shiftKey
        && (
          isTextEntryTarget(target)
          || !interactive
          || (target instanceof HTMLInputElement && target.type === "file")
        )
      ) {
        event.preventDefault();
        actions.submit();
        return;
      }

      if (
        event.altKey
        && !event.ctrlKey
        && !event.metaKey
        && event.key === "ArrowUp"
        && actions.getMode() === "trake"
      ) {
        event.preventDefault();
        actions.addTrakeEvent();
        return;
      }

      if (
        event.altKey
        && !event.ctrlKey
        && !event.metaKey
        && event.key === "ArrowDown"
        && actions.getMode() === "trake"
      ) {
        event.preventDefault();
        actions.removeLastTrakeEvent();
        return;
      }

      if (interactive || event.altKey || event.ctrlKey || event.metaKey) {
        return;
      }

      const mode = MODE_SHORTCUTS[event.key];
      if (mode) {
        event.preventDefault();
        actions.setMode(mode);
        return;
      }

      if (event.key === "/") {
        event.preventDefault();
        actions.focusPrimaryInput();
        return;
      }

      if (!event.shiftKey && (event.key === "ArrowLeft" || event.key === "ArrowUp")) {
        if (actions.moveResultSelection(-1)) {
          event.preventDefault();
        }
        return;
      }

      if (!event.shiftKey && (event.key === "ArrowRight" || event.key === "ArrowDown")) {
        if (actions.moveResultSelection(1)) {
          event.preventDefault();
        }
        return;
      }

      if (!event.shiftKey && event.key.toLowerCase() === "c") {
        actions.copySelectedResult();
      }
    });
  }

  window.ChiLangShortcuts = Object.freeze({
    install,
  });
})();
