"""Check this interpreter against the exported ER environment, without using GPUs."""

import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check(snapshot, prefix, repository):
    errors = []
    if platform.system().lower() != snapshot["platform"] or platform.machine() != snapshot["machine"]:
        errors.append("This snapshot requires Linux x86_64.")
    if platform.python_version() != snapshot["python_version"]:
        errors.append(f"Python: expected {snapshot['python_version']}, found {platform.python_version()}")

    installed = {}
    origins = {}
    for distribution in metadata.distributions():
        name = canonical_name(distribution.metadata["Name"])
        installed.setdefault(name, set()).add(distribution.version)
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            origins[name] = json.loads(direct_url)

    for name, version in snapshot["python_packages"].items():
        if installed.get(name) != {version}:
            found = ", ".join(sorted(installed.get(name, []))) or "missing"
            errors.append(f"{name}: expected {version}, found {found}")
    extras = sorted(set(installed) - set(snapshot["python_packages"]))
    if extras:
        errors.append(f"Additional Python packages not in snapshot: {', '.join(extras)}")

    for name, relative_path in snapshot["editable_packages"].items():
        origin = origins.get(name, {})
        location = urlparse(origin.get("url", ""))
        if not origin.get("dir_info", {}).get("editable") or location.scheme != "file":
            errors.append(f"{name}: expected an editable installation from this checkout")
        elif Path(unquote(location.path)).resolve() != (repository / relative_path).resolve():
            errors.append(f"{name}: editable installation points to a different checkout")

    flash_archive = origins.get("flash-attn", {}).get("archive_info", {})
    flash_hash = flash_archive.get("hashes", {}).get("sha256")
    if flash_hash != snapshot["flash_attention_wheel"]["sha256"]:
        errors.append("flash-attn: installed wheel origin does not record the expected SHA256")

    expected_conda = set()
    for package in snapshot["conda_packages"]:
        record_name = f"{package['name']}-{package['version']}-{package['build']}.json"
        expected_conda.add(record_name)
        record_path = prefix / "conda-meta" / record_name
        if not record_path.is_file():
            errors.append(f"Conda package missing or different build: {record_name}")
            continue
        record = json.loads(record_path.read_text())
        if record.get("md5") != package["md5"]:
            errors.append(f"Conda artifact checksum differs: {package['name']}")
    extra_conda = {path.name for path in (prefix / "conda-meta").glob("*.json")} - expected_conda
    if extra_conda:
        errors.append(f"Additional Conda packages not in snapshot: {', '.join(sorted(extra_conda))}")
    return errors


def main():
    directory = Path(__file__).resolve().parent
    snapshot = json.loads((directory / "snapshot.json").read_text())
    errors = check(snapshot, Path(sys.prefix), directory.parents[1])
    if errors:
        print("Environment does not match the snapshot:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        f"Matched Python {snapshot['python_version']}, {len(snapshot['python_packages'])} Python packages, "
        f"{len(snapshot['conda_packages'])} Conda builds, editable paths, and FlashAttention wheel origin."
    )
    print("This checks package metadata, not GPU/driver compatibility or bitwise file contents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
