"""Cut an MCAP bag to the window in which every topic is recording.

The window is `bag_check.common_window`: [latest first message, earliest last
message] over every topic not named with --static-topic, on log time.  Static
topics (/tf_static, one-shot topics) are copied whole, every message at its
original log time, so the output bag's own start can precede the window.

The copy is done by the mcap CLI (pinned in docker/ingest.Dockerfile), not
rosbag2: rosbag2's writer re-derives each schema from the message packages
installed in this image, so custom types (novatel_oem7_msgs, roscco_msg,
eidos_msgs) would come out with empty schemas.  `mcap filter` / `mcap merge`
copy records as they are — payloads, schemas, channel metadata (QoS) — so the
only thing that changes is which messages are in the file.

    1. mcap filter <in> --start <window start> --end <window end + 1>
           --exclude-topic-regex <each static topic>     -> window part
    2. mcap filter <in> --include-topic-regex <each static topic> -> static part
    3. mcap merge <window part> <static part>                        -> output
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from wato_ingest.inputs import bag_check


class TrimError(RuntimeError):
    pass


def resolve_mcap_file(bag_path: str) -> Path:
    """The single .mcap file of a bag given as a file or a rosbag2 directory."""
    p = Path(bag_path)
    if p.is_file():
        if p.suffix != ".mcap":
            raise TrimError(f"{bag_path}: trim-bag works on MCAP bags only")
        return p
    files = sorted(p.glob("*.mcap"))
    if len(files) != 1:
        raise TrimError(
            f"{bag_path}: expected exactly one .mcap file in the bag directory, "
            f"found {[f.name for f in files]}"
        )
    return files[0]


def default_output(mcap_file: Path, out_dir: str) -> Path:
    """<out_dir>/<input stem>_trimmed.mcap"""
    stem = mcap_file.stem
    if (mcap_file.parent / "metadata.yaml").exists():
        # A rosbag2 directory names the bag; its file is <dir>_0.mcap.
        stem = mcap_file.parent.name
    return Path(out_dir) / f"{stem}_trimmed.mcap"


def _topic_regex(topic: str) -> str:
    # ROS topic names are [A-Za-z0-9_/~]; anchor so /tf never matches /tf_static.
    return f"^{topic}$"


def mcap_commands(
    src: Path,
    window_part: Path,
    static_part: Path,
    out: Path,
    *,
    start_ns: int,
    end_ns: int,
    static_topics: list[str],
    compression: str,
) -> list[list[str]]:
    """The mcap CLI invocations that write `out`.  `end_ns` is inclusive."""
    # Nothing to merge without static topics: filter straight to the output.
    window_out = window_part if static_topics else out
    window = [
        "mcap", "filter", str(src), "-o", str(window_out),
        "--start", str(start_ns),
        # --end is exclusive ("before this time"); the window's last message
        # sits exactly at end_ns.
        "--end", str(end_ns + 1),
        "--compression", compression,
    ]  # fmt: skip
    for t in static_topics:
        window += ["--exclude-topic-regex", _topic_regex(t)]
    cmds = [window]
    if static_topics:
        static = ["mcap", "filter", str(src), "-o", str(static_part)]
        static += ["--compression", compression]
        for t in static_topics:
            static += ["--include-topic-regex", _topic_regex(t)]
        cmds.append(static)
        cmds.append(
            [
                "mcap",
                "merge",
                str(window_part),
                str(static_part),
                "-o",
                str(out),
                "--compression",
                compression,
                # rosbag2 writes the same metadata record into both parts.
                "--allow-duplicate-metadata",
            ]  # fmt: skip
        )
    return cmds


@dataclass
class TrimResult:
    output: str
    window: bag_check.Window
    input_start_ns: int
    input_end_ns: int
    static_topics: list[str]
    # topic -> (messages in the input, messages in the output)
    counts: dict[str, tuple[int, int]]


def expected_counts(
    bag: bag_check.BagScan, start_ns: int, end_ns: int, static_topics: set[str]
) -> dict[str, int]:
    """Messages each topic should have after the cut."""
    out = {}
    for name, t in bag.topics.items():
        if name in static_topics:
            out[name] = len(t.log_ns)
        else:
            out[name] = int(((t.log_ns >= start_ns) & (t.log_ns <= end_ns)).sum())
    return out


def trim(
    bag_path: str,
    out_path: str,
    *,
    static_topics: list[str],
    storage_id: str = "",
    compression: str = "none",
    overwrite: bool = False,
) -> TrimResult:
    src = resolve_mcap_file(bag_path)
    out = Path(out_path)
    if out.exists() and not overwrite:
        raise TrimError(f"{out} exists; pass --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)

    bag = bag_check.scan(bag_path, storage_id=storage_id, headers=False)
    static = set(static_topics)
    unknown = sorted(static - set(bag.topics))
    if unknown:
        raise TrimError(f"--static-topic not in the bag: {unknown}")
    window = bag_check.common_window(bag, static)
    if window.start_ns is None:
        raise TrimError(f"no common window: {window.reason}")

    tmp = out.with_name(f".{out.stem}.partial.mcap")
    window_part = out.with_name(f".{out.stem}.window.mcap")
    static_part = out.with_name(f".{out.stem}.static.mcap")
    try:
        cmds = mcap_commands(
            src,
            window_part,
            static_part,
            tmp,
            start_ns=window.start_ns,
            end_ns=window.end_ns,
            static_topics=sorted(static),
            compression=compression,
        )
        for cmd in cmds:
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise TrimError(f"{' '.join(cmd)} failed:\n{done.stderr.strip()}")

        # Check the result message-for-message before it takes the final name.
        want = expected_counts(bag, window.start_ns, window.end_ns, static)
        got_scan = bag_check.scan(str(tmp), headers=False)
        got = {n: len(t.log_ns) for n, t in got_scan.topics.items()}
        wrong = {n: (want[n], got.get(n, 0)) for n in want if got.get(n, 0) != want[n]}
        if wrong or set(got) - set(want):
            raise TrimError(
                f"trimmed bag doesn't match the plan (topic: expected, written): "
                f"{wrong}; unexpected topics {sorted(set(got) - set(want))}"
            )
        os.replace(tmp, out)
    finally:
        for p in (tmp, window_part, static_part):
            p.unlink(missing_ok=True)

    return TrimResult(
        output=str(out),
        window=window,
        input_start_ns=bag.start_ns,
        input_end_ns=bag.end_ns,
        static_topics=sorted(static),
        counts={n: (len(bag.topics[n].log_ns), want[n]) for n in sorted(want)},
    )
