#!/usr/bin/env python3
"""Download the scorpus demo datasets from HuggingFace.

Downloads the two small demo .h5ad files and checksums from
``weililab/perturb-data-lab-demo`` into a local directory.

Usage
-----
.. code-block:: bash

    python scripts/download_demo_data.py

    python scripts/download_demo_data.py --output-dir ./my_demo_data

    python scripts/download_demo_data.py --method wget
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

_REPO = "weililab/perturb-data-lab-demo"
_BASE_URL = f"https://huggingface.co/datasets/{_REPO}/resolve/main"

_FILES = [
    "h5ad/demo_marson_d2_rest.h5ad",
    "h5ad/demo_xorion_hct116_dual_guide.h5ad",
    "checksums.txt",
]


def _download_hub(output_dir: Path) -> None:
    """Download via huggingface_hub (requires the package)."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError:
        print(
            "huggingface_hub is not installed. Install it with:\n"
            "  pip install huggingface_hub\n\n"
            "Or use --method wget instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in _FILES:
        dest_dir = output_dir / rel_path
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            hf_hub_download(
                repo_id=_REPO,
                filename=rel_path,
                repo_type="dataset",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Downloaded {rel_path}")
        except HfHubHTTPError as exc:
            print(f"Error downloading {rel_path}: {exc}", file=sys.stderr)
            sys.exit(1)


def _download_wget(output_dir: Path) -> None:
    """Download via wget (no extra Python packages needed)."""
    import subprocess

    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in _FILES:
        dest_dir = output_dir / rel_path
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE_URL}/{rel_path}"
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(dest_dir), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"Error downloading {rel_path}: {result.stderr.strip()}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Downloaded {rel_path}")


def _download_requests(output_dir: Path) -> None:
    """Download via requests with progress bar."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in _FILES:
        dest_dir = output_dir / rel_path
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"{_BASE_URL}/{rel_path}"
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_dir, "wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                fh.write(chunk)
                downloaded += len(chunk)
        print(f"Downloaded {rel_path} ({downloaded} bytes)")


def _verify_checksums(output_dir: Path) -> None:
    """Verify downloaded files against checksums.txt."""
    checksum_file = output_dir / "checksums.txt"
    if not checksum_file.exists():
        print("Warning: checksums.txt not found; skipping verification.")
        return
    with open(checksum_file, encoding="utf-8") as fh:
        expected: dict[str, str] = {}
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                expected[parts[1]] = parts[0]

    ok = True
    for rel_path, expected_hash in expected.items():
        filepath = output_dir / rel_path
        if not filepath.exists():
            print(f"Missing: {rel_path}", file=sys.stderr)
            ok = False
            continue
        actual_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            print(
                f"Checksum mismatch for {rel_path}:\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}",
                file=sys.stderr,
            )
            ok = False
    if ok:
        print("All checksums verified.")
    else:
        print("Checksum verification FAILED.", file=sys.stderr)
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download scorpus demo datasets from HuggingFace.",
    )
    parser.add_argument(
        "--output-dir",
        default="./demo_data",
        help="Directory to write downloaded files (default: ./demo_data).",
    )
    parser.add_argument(
        "--method",
        choices=["huggingface_hub", "wget", "requests"],
        default="huggingface_hub",
        help="Download method (default: huggingface_hub).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    dispatch = {
        "huggingface_hub": _download_hub,
        "wget": _download_wget,
        "requests": _download_requests,
    }

    print(f"Downloading to {output_dir}")
    dispatch[args.method](output_dir)

    if not args.no_verify:
        _verify_checksums(output_dir)

    print("\nDownload complete. Place the data in a convenient location and")
    print("update paths in the demo scripts if needed.")


if __name__ == "__main__":
    main()
