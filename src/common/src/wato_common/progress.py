"""Keep tqdm bars from spamming non-interactive output.

tqdm redraws with ``\\r``; when stderr isn't a TTY (``watod``, ``docker
compose``, CI logs) every redraw becomes a new line. That includes bars we
don't own (SAM2's ``propagate_in_video``), some of which pass an explicit
``disable=False``, so ``configure_tqdm()`` patches ``tqdm.std.tqdm.__init__``
process-wide. Call it once at CLI startup. ``WATO_PROGRESS`` selects:

    auto   (default) live bar on a TTY, silent when captured
    on     leave bars as-is
    off    never show bars
"""

from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

_configured = False


def _stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 — closed / fake / non-stream file objects
        return False


def configure_tqdm() -> None:
    """Patch tqdm per WATO_PROGRESS (see module docstring). Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    mode = os.environ.get("WATO_PROGRESS", "auto").strip().lower()
    if mode in ("on", "1", "true", "yes"):
        return

    force_off = mode in ("off", "0", "false", "no")

    try:
        from tqdm.std import tqdm as _tqdm
    except Exception as exc:  # noqa: BLE001 — tqdm optional / import failure
        log.debug("configure_tqdm: tqdm unavailable (%s) — skipping", exc)
        return

    if getattr(_tqdm, "_wato_progress_patched", False):
        return

    _orig_init = _tqdm.__init__

    def _patched_init(self, *args, **kwargs):
        # tqdm measures the width once; in a docker pty it is often wrong, and
        # each redraw then leaves a wrapped remainder behind.
        kwargs.setdefault("dynamic_ncols", True)
        if force_off:
            kwargs["disable"] = True
        else:  # auto: disable only when the target stream isn't a terminal
            stream = kwargs.get("file") or sys.stderr
            if not _stream_is_tty(stream):
                kwargs["disable"] = True
        _orig_init(self, *args, **kwargs)

    _tqdm.__init__ = _patched_init
    _tqdm._wato_progress_patched = True
    log.debug("configure_tqdm: patched tqdm (WATO_PROGRESS=%s)", mode)
