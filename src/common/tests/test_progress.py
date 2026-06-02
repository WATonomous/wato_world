"""configure_tqdm() must stop tqdm from spamming non-TTY output (the watod /
docker / file-capture case) while leaving live terminal bars alone — including
overriding SAM 3.1's explicit ``disable=False``."""

from __future__ import annotations

import io

import wato_common.progress as progress


class _NonTTY(io.StringIO):
    def isatty(self) -> bool:
        return False


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _reset_patch_state() -> None:
    """Let configure_tqdm re-patch with a fresh mode in-process."""
    progress._configured = False
    from tqdm.std import tqdm as _tqdm

    if getattr(_tqdm, "_wato_progress_patched", False):
        # Tests run in one process; we can't un-wrap __init__ cleanly, so only
        # the first configure wins. The default-auto test runs first and covers
        # the behaviour; subsequent calls are asserted as no-ops.
        pass


def test_auto_forces_off_on_non_tty_even_with_explicit_disable_false():
    progress.configure_tqdm()  # auto (no env) → off on non-TTY
    from tqdm import tqdm

    stream = _NonTTY()
    # sam3 passes disable=self.rank > 0 → disable=False on the single GPU.
    bar = tqdm(range(5), file=stream, disable=False, desc="propagate_in_video")
    for _ in bar:
        pass
    bar.close()
    assert stream.getvalue() == ""  # nothing drawn


def test_auto_leaves_tty_bar_rendering():
    progress.configure_tqdm()
    from tqdm import tqdm

    stream = _TTY()
    bar = tqdm(range(5), file=stream, disable=False, desc="live")
    for _ in bar:
        pass
    bar.close()
    assert stream.getvalue() != ""  # bar drew to the terminal


def test_configure_is_idempotent():
    progress.configure_tqdm()
    from tqdm.std import tqdm as _tqdm

    first = _tqdm.__init__
    progress.configure_tqdm()  # second call must not re-wrap
    assert _tqdm.__init__ is first
