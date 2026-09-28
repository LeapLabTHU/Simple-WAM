#!/usr/bin/env python3
"""Upgrade SAPIEN's bundled Open Image Denoise libraries for Blackwell GPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request


OIDN_VERSION = "2.3.3"
ARCHIVE_NAME = f"oidn-{OIDN_VERSION}.x86_64.linux.tar.gz"
ARCHIVE_URL = (
    "https://github.com/RenderKit/oidn/releases/download/"
    f"v{OIDN_VERSION}/{ARCHIVE_NAME}"
)
ARCHIVE_SHA256 = "3c385230d9e6f63527ba72f2229594dbac5051674219d72e0044b5d0b841796f"
LIBRARY_SHA256 = {
    f"libOpenImageDenoise.so.{OIDN_VERSION}": (
        "9ac9dcd18318cae879efb01c14cff72c2c4275428a15e5ab6e94c8ff8a43c4dc"
    ),
    f"libOpenImageDenoise_core.so.{OIDN_VERSION}": (
        "eadb2437d6ac4a88e5746980495ba1091f7d14745e3000b82a9382093d22f351"
    ),
    f"libOpenImageDenoise_device_cuda.so.{OIDN_VERSION}": (
        "12b3ff87cfa4b6a90481b757f5a56ee69380a45e3dcaeadf8dd22cb73d4f714a"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_sapien() -> Path:
    try:
        distribution = importlib.metadata.distribution("sapien")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit("SAPIEN is not installed in the active Python environment.") from exc

    package_dir = Path(distribution.locate_file("sapien")).resolve()
    if not (package_dir / "_oidn_tricks.py").is_file():
        raise SystemExit(f"Cannot find SAPIEN's _oidn_tricks.py under {package_dir}")
    return package_dir


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARCHIVE_URL}")
    with urllib.request.urlopen(ARCHIVE_URL) as response, tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as output:
        temporary_path = Path(output.name)
        shutil.copyfileobj(response, output)
    temporary_path.replace(destination)


def verified_archive(path: Path | None) -> Path:
    if path is None:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        path = cache_root / "simplewam" / ARCHIVE_NAME
        if not path.is_file():
            download_archive(path)

    if not path.is_file():
        raise SystemExit(f"OIDN archive does not exist: {path}")
    actual_hash = sha256(path)
    if actual_hash != ARCHIVE_SHA256:
        raise SystemExit(
            f"OIDN archive checksum mismatch for {path}:\n"
            f"expected {ARCHIVE_SHA256}\nactual   {actual_hash}"
        )
    return path


def install_libraries(archive: Path, package_dir: Path) -> None:
    library_dir = package_dir / "oidn_library"
    library_dir.mkdir(parents=True, exist_ok=True)
    archive_prefix = f"oidn-{OIDN_VERSION}.x86_64.linux/lib"

    with tarfile.open(archive, "r:gz") as tar:
        for filename, expected_hash in LIBRARY_SHA256.items():
            member_name = f"{archive_prefix}/{filename}"
            try:
                member = tar.getmember(member_name)
            except KeyError as exc:
                raise SystemExit(f"Missing {member_name} in {archive}") from exc
            source = tar.extractfile(member)
            if source is None:
                raise SystemExit(f"Cannot read {member_name} from {archive}")

            destination = library_dir / filename
            with tempfile.NamedTemporaryFile(
                dir=library_dir, prefix=f".{filename}.", delete=False
            ) as output:
                temporary_path = Path(output.name)
                shutil.copyfileobj(source, output)
            if sha256(temporary_path) != expected_hash:
                temporary_path.unlink()
                raise SystemExit(f"Checksum mismatch after extracting {member_name}")
            temporary_path.chmod(0o755)
            temporary_path.replace(destination)


def patch_loader(package_dir: Path) -> None:
    loader = package_dir / "_oidn_tricks.py"
    source = loader.read_text()
    if OIDN_VERSION in source:
        return
    if "2.0.1" not in source:
        raise SystemExit(
            f"Unsupported SAPIEN OIDN loader in {loader}; expected version 2.0.1."
        )

    patched = source.replace("2.0.1", OIDN_VERSION)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=loader.parent, prefix=f".{loader.name}.", delete=False
    ) as output:
        temporary_path = Path(output.name)
        output.write(patched)
    shutil.copymode(loader, temporary_path)
    temporary_path.replace(loader)


def verify_installation(package_dir: Path) -> None:
    for filename, expected_hash in LIBRARY_SHA256.items():
        path = package_dir / "oidn_library" / filename
        if not path.is_file() or sha256(path) != expected_hash:
            raise SystemExit(f"OIDN installation verification failed: {path}")

    loader = package_dir / "_oidn_tricks.py"
    source = loader.read_text()
    if OIDN_VERSION not in source or "2.0.1" in source:
        raise SystemExit(f"SAPIEN still loads the old OIDN version: {loader}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an existing official OIDN archive instead of downloading it.",
    )
    parser.add_argument(
        "--sapien-dir",
        type=Path,
        help="Override the active environment's SAPIEN package directory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that the patch is already installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_dir = args.sapien_dir.resolve() if args.sapien_dir else locate_sapien()
    if args.check:
        verify_installation(package_dir)
        print(f"SAPIEN OIDN {OIDN_VERSION} patch verified: {package_dir}")
        return

    archive = verified_archive(args.archive.resolve() if args.archive else None)
    install_libraries(archive, package_dir)
    patch_loader(package_dir)
    verify_installation(package_dir)
    print(f"Installed SAPIEN OIDN {OIDN_VERSION} patch: {package_dir}")
    print("Restart any Python evaluation processes that imported SAPIEN.")


if __name__ == "__main__":
    main()
