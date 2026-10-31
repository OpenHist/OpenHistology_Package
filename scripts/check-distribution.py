"""Check local wheel/source artifacts before sharing; this does not publish them."""

from argparse import ArgumentParser
from email import message_from_bytes
from pathlib import Path, PurePosixPath
import tarfile
from zipfile import ZipFile

from packaging.requirements import Requirement


REPOSITORY = "https://github.com/OpenHist/OpenHistology_Package"


def check_metadata(data):
    metadata = message_from_bytes(data)
    direct = [value for value in metadata.get_all("Requires-Dist", []) if Requirement(value).url]
    if direct:
        raise RuntimeError(f"Distribution contains direct URL dependencies: {direct}")
    if metadata.get("License-Expression") != "MIT":
        raise RuntimeError("Distribution is missing the project license expression")
    if metadata.get("Author") != "Abteen Arab":
        raise RuntimeError("Distribution is missing the project author")
    urls = metadata.get_all("Project-URL", [])
    if f"Repository, {REPOSITORY}" not in urls:
        raise RuntimeError("Distribution has an incorrect repository URL")


def check_files(names, read):
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Distribution contains an unsafe path: {name}")
        if ".git" in path.parts or "__pycache__" in path.parts:
            raise RuntimeError(f"Distribution includes local metadata: {name}")
        if name.endswith((".pyc", ".pyo", "/.DS_Store")):
            raise RuntimeError(f"Distribution includes a generated file: {name}")
        if name.startswith("benchmarks/") or name.startswith("tests/test_benchmark_"):
            raise RuntimeError(f"Distribution includes experiment files: {name}")
        if name.endswith(".py"):
            compile(read(name), name, "exec")


def check_wheel(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        check_metadata(archive.read(metadata_path))
        license_path = metadata_path.removesuffix("METADATA") + "licenses/LICENSE"
        if license_path not in names:
            raise RuntimeError("Wheel does not include the project LICENSE file")
        check_files(names, archive.read)


def check_sdist(path):
    with tarfile.open(path) as archive:
        files = [item for item in archive.getmembers() if item.isfile()]
        roots = {PurePosixPath(item.name).parts[0] for item in files}
        if len(roots) != 1:
            raise RuntimeError("Source archive must have one top-level directory")
        prefix = roots.pop() + "/"
        members = {item.name.removeprefix(prefix): item for item in files}

        def read(name):
            stream = archive.extractfile(members[name])
            if stream is None:
                raise RuntimeError(f"Cannot read source archive member: {name}")
            return stream.read()

        for required in ("PKG-INFO", "LICENSE", "README.md", "CONTRIBUTING.md", "pyproject.toml"):
            if required not in members:
                raise RuntimeError(f"Source archive is missing {required}")
        check_metadata(read("PKG-INFO"))
        check_files(members, read)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    for artifact in parser.parse_args().artifacts:
        if artifact.suffix == ".whl":
            check_wheel(artifact)
        elif artifact.name.endswith(".tar.gz"):
            check_sdist(artifact)
        else:
            parser.error(f"Unsupported distribution: {artifact}")
        print(f"Verified metadata, licenses, contents and Python syntax: {artifact.name}")


if __name__ == "__main__":
    main()
