"""Copy standalone corpora to node-local storage (e.g. NVMe) before training.

``estimate`` reports how much would be copied, ``stage`` copies and keeps the
copies, and ``staged`` copies on entry and deletes the copies on exit.
Each corpus folder ``<root>`` is copied to ``<target_dir>/<root.name>``.

A target under ``/dev/shm`` stages into RAM: the copies count against the job's
memory limit, so ``stage`` also refuses a copy that would not fit in it.
"""

from __future__ import annotations

import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

# Files are copied in 64 MB ranges by 32 threads. On beacon (NFS -> /scratch1)
# this ran at 1.2-1.8 GB/s, against ~0.12 GB/s for one thread per file.
RANGE_BYTES = 64 * 1024 * 1024
THREADS = 32
# Rough copy speed for the time estimate, measured once on beacon (low end).
ASSUMED_BYTES_PER_SECOND = 1.2e9
RAM_DIR = Path("/dev/shm")


def _files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def estimate(corpus_roots: Sequence[str | Path], target_dir: str | Path) -> dict[str, float]:
    """Bytes to copy, free bytes on the target, and an approximate copy time.

    ``approx_seconds`` assumes ~1.2 GB/s (measured once on beacon); actual
    speed depends on the source filesystem load and the node.
    """
    total_bytes = sum(path.stat().st_size for root in corpus_roots for path in _files(Path(root)))
    free_bytes = shutil.disk_usage(target_dir).free
    return {
        "bytes": total_bytes,
        "free_bytes": free_bytes,
        "approx_seconds": total_bytes / ASSUMED_BYTES_PER_SECOND,
    }


def _memory_headroom() -> float:
    """Bytes this process can still add to its memory before the job's limit (cgroup v2).

    Files in ``/dev/shm`` are charged to the cgroup of the process that writes them. The
    limit can be set on any ancestor cgroup (Slurm on beacon sets it on ``.../step_*/user``,
    not on the task's own cgroup), so this is the smallest headroom over all of them;
    ``inf`` if none has a limit. Page cache (``active_file``, ``inactive_file``) is not
    counted as used, because the kernel drops it before running out of memory.
    """
    cgroup_root = Path("/sys/fs/cgroup")
    directory = cgroup_root / Path("/proc/self/cgroup").read_text().strip().removeprefix("0::/")
    headroom = math.inf
    while directory != cgroup_root:
        limit = (directory / "memory.max").read_text().strip()
        if limit != "max":
            stat = dict(line.split() for line in (directory / "memory.stat").read_text().splitlines())
            used = int((directory / "memory.current").read_text()) - int(stat["active_file"]) - int(stat["inactive_file"])
            headroom = min(headroom, int(limit) - used)
        directory = directory.parent
    return headroom


def _copy_range(source: Path, destination: Path, offset: int) -> None:
    # pread/pwrite may move fewer bytes than asked (e.g. on NFS), so loop until the range is done.
    with open(source, "rb", buffering=0) as src, open(destination, "r+b", buffering=0) as dst:
        position, stop = offset, offset + RANGE_BYTES
        while position < stop:
            data = memoryview(os.pread(src.fileno(), stop - position, position))
            if not data:  # end of file
                break
            written = 0
            while written < len(data):
                written += os.pwrite(dst.fileno(), data[written:], position + written)
            position += len(data)


def stage(corpus_roots: Sequence[str | Path], target_dir: str | Path) -> list[Path]:
    """Copy each corpus folder into ``target_dir`` and return the staged roots.

    Refuses before copying anything if the corpora do not fit in the free space, or,
    for a target under ``/dev/shm`` (RAM), in the job's memory limit.
    """
    roots = [Path(root) for root in corpus_roots]
    target = Path(target_dir)
    needed = estimate(roots, target)
    if needed["bytes"] > needed["free_bytes"]:
        raise OSError(
            f"Staging needs {needed['bytes'] / 1e9:.1f} GB but {target} has "
            f"{needed['free_bytes'] / 1e9:.1f} GB free"
        )
    if target.resolve().is_relative_to(RAM_DIR):
        headroom = _memory_headroom()
        if needed["bytes"] > headroom:
            raise OSError(
                f"Staging into RAM needs {needed['bytes'] / 1e9:.1f} GB but the job's memory limit "
                f"leaves {headroom / 1e9:.1f} GB"
            )
    # Every file is created at full size first, then its 64 MB ranges are
    # copied in parallel across all files.
    staged_roots = [target / root.name for root in roots]
    ranges = []
    for root, staged_root in zip(roots, staged_roots):
        # Raises if a staged copy (or another corpus with the same name) is already there.
        staged_root.mkdir()
        for source in _files(root):
            destination = staged_root / source.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            with open(destination, "wb") as out:
                out.truncate(size)
            ranges += [(source, destination, offset) for offset in range(0, size, RANGE_BYTES)]
    with ThreadPoolExecutor(THREADS) as threads:
        list(threads.map(lambda task: _copy_range(*task), ranges))
    return staged_roots


@contextmanager
def staged(corpus_roots: Sequence[str | Path], target_dir: str | Path) -> Iterator[list[Path]]:
    """Stage corpora for the duration of a ``with`` block, then delete the copies."""
    staged_roots = stage(corpus_roots, target_dir)
    try:
        yield staged_roots
    finally:
        for root in staged_roots:
            shutil.rmtree(root)
