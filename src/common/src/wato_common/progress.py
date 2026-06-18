"""One place to make tqdm behave in batch / non-interactive runs.

The problem: tqdm draws its bar by rewriting the current line with a carriage
return (``\\r``). On a real terminal that overwrites in place — one tidy line.
But when stdout/stderr is NOT a TTY (piped through ``watod`` / ``docker compose``,
tee'd to a file, captured by CI), the ``\\r`` updates are recorded as separate
lines and the bar "spams" hundreds of lines.

This bites every bar in a run, including ones we don't own — SAM2's internal
``propagate_in_video`` / frame-loading bars use the same ``tqdm.std.tqdm`` class
we do. We can't fix those at the call site; worse, a third-party bar may pass an
explicit ``disable=False``, so a mere default wouldn't help. So we wrap
``tqdm.std.tqdm.__init__`` process-wide and,
in the default "auto" mode, *force* ``disable=True`` whenever the bar's output
stream isn't a TTY — overriding even an explicit ``disable=False``. Result: a
live single-line bar on a terminal, silence when the output is being captured.

Call ``configure_tqdm()`` once at CLI startup (before any bar is created). It is
idempotent. Override with the ``WATO_PROGRESS`` env var:

    WATO_PROGRESS=auto   (default) live bar on a TTY, silent when captured
    WATO_PROGRESS=on     leave bars as-is (the old, possibly spammy, behaviour)
    WATO_PROGRESS=off    never show bars
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
    """Patch tqdm so progress bars don't spam non-TTY output.

    Wraps ``tqdm.std.tqdm.__init__`` to set ``disable`` per WATO_PROGRESS:
      - "auto" (default): force-disable on non-TTY output (overrides an explicit
        ``disable=False``, which is what some third-party bars pass); leave TTY
        bars alone so they still render live.
      - "off": always disable.
      - "on": leave the caller's value untouched.
    Idempotent; a no-op if tqdm isn't installed.
    """
    global _configured
    if _configured:
        return
    _configured = True

    mode = os.environ.get("WATO_PROGRESS", "auto").strip().lower()
    if mode in ("on", "1", "true", "yes"):
        return  # nothing to do — keep upstream behaviour

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
