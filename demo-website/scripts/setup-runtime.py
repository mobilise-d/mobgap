#!/usr/bin/env python3
"""Build browser assets from pinned packages and this checkout, without system installs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import venv
import zipfile
from pathlib import Path

import tomllib

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parent
RUNTIME = DEMO / "runtime"
BUILD = RUNTIME / "build"
PUBLIC = DEMO / "public" / "runtime"
MICROMAMBA_URL = "https://api.anaconda.org/download/conda-forge/micromamba/2.9.0/linux-64/micromamba-2.9.0-0.tar.bz2"
MICROMAMBA_SHA = "8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd"


def download(url: str, path: Path, sha256: str) -> Path:
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
        return path
    data = urllib.request.urlopen(url, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"Checksum mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def index_channel() -> None:
    expected = {
        entry["filename"]: entry["sha256"] for entry in json.loads((RUNTIME / "artifacts.lock.json").read_text())
    }
    for subdir in ("emscripten-wasm32", "noarch"):
        folder = RUNTIME / "channel" / subdir
        folder.mkdir(parents=True, exist_ok=True)
        packages = {}
        for path in folder.glob("*.tar.bz2"):
            data = path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            if expected.get(path.name) != sha:
                raise RuntimeError(f"Local package checksum mismatch: {path.name}")
            with tarfile.open(path) as archive:
                metadata = json.load(archive.extractfile("info/index.json"))
            metadata.update(size=len(data), sha256=sha, md5=hashlib.md5(data).hexdigest())
            packages[path.name] = metadata
        (folder / "repodata.json").write_text(
            json.dumps(
                {"info": {"subdir": subdir}, "packages": packages, "packages.conda": {}, "repodata_version": 1},
                indent=2,
            )
        )


def bundle_sources() -> None:
    """Bundle repository Python and pure wheels, retaining their dist-info/licenses."""
    PUBLIC.mkdir(parents=True, exist_ok=True)
    shutil.copytree(RUNTIME / "licenses", PUBLIC / "licenses", dirs_exist_ok=True)
    shutil.copy2(RUNTIME / "workerfs" / "LICENSE.emscripten", PUBLIC / "licenses" / "LICENSE.emscripten")
    wheels = []
    for entry in json.loads((RUNTIME / "wheels.lock.json").read_text()):
        wheels.append(download(entry["url"], BUILD / "wheels" / entry["filename"], entry["sha256"]))
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    with zipfile.ZipFile(PUBLIC / "bootstrap.zip", "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted((REPO / "src" / "mobgap").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                output.write(path, path.relative_to(REPO / "src").as_posix())
        output.write(REPO / "LICENSE", "mobgap-LICENSE")
        output.writestr(
            f"mobgap-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: mobgap\nVersion: {version}\n"
        )
        output.write(DEMO / "python" / "mobgap_demo_api.py", "mobgap_demo_api.py")
        for name in ("file_access_probe.py", "cwa_window_probe.py", "cwa_pipeline_probe.py"):
            output.write(DEMO / "python" / name, name)
        for name in ("workerfs.js", "bridge.js"):
            output.write(RUNTIME / "workerfs" / name, name)
        output.write(RUNTIME / "workerfs" / "LICENSE.emscripten", "LICENSE.emscripten")
        names = set(output.namelist())
        for wheel in wheels:
            with zipfile.ZipFile(wheel) as archive:
                for entry in archive.infolist():
                    if not entry.is_dir() and entry.filename not in names:
                        output.writestr(entry.filename, archive.read(entry))
                        names.add(entry.filename)
    print(f"Created {PUBLIC / 'bootstrap.zip'}")


def build_runtime(jupyter: str | None, micromamba: str | None) -> None:
    index_channel()
    BUILD.mkdir(parents=True, exist_ok=True)
    if not jupyter:
        prefix = RUNTIME / ".venv"
        builder = venv.EnvBuilder(with_pip=True)
        context = builder.ensure_directories(str(prefix))
        executable = Path(context.bin_path) / ("jupyter.exe" if sys.platform == "win32" else "jupyter")
        if not executable.exists():
            builder.create(prefix)
            subprocess.run(
                [context.env_exe, "-m", "pip", "install", "-r", str(RUNTIME / "build-requirements.txt")],
                check=True,
            )
        jupyter = str(executable)
    if not micromamba:
        if sys.platform != "linux" or os.uname().machine != "x86_64":
            raise RuntimeError("Pass --micromamba /path/to/micromamba version 2.9.0 on this platform.")
        package = download(MICROMAMBA_URL, BUILD / "micromamba.tar.bz2", MICROMAMBA_SHA)
        executable = BUILD / "bin" / "micromamba"
        executable.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(package) as archive:
            executable.write_bytes(archive.extractfile("bin/micromamba").read())
        executable.chmod(0o755)
        micromamba = str(executable)
    if subprocess.check_output([micromamba, "--version"], text=True).strip() != "2.9.0":
        raise RuntimeError("This build requires micromamba 2.9.0.")
    packages = json.loads((RUNTIME / "packages.lock.json").read_text())["packages"]
    environment = [
        "name: mobgap-browser",
        "channels:",
        f"  - {(RUNTIME / 'channel').as_uri()}",
        "  - https://repo.prefix.dev/emscripten-forge-4x",
        "  - https://repo.prefix.dev/conda-forge",
        "dependencies:",
    ]
    environment += [f"  - {p['name']}={p['version']}={p['build']}" for p in packages]
    (BUILD / "environment.yml").write_text("\n".join(environment) + "\n")
    env = dict(os.environ)
    env["PATH"] = str(Path(micromamba).resolve().parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([jupyter, "lite", "build", "--output-dir", str(PUBLIC)], cwd=BUILD, env=env, check=True)
    config = PUBLIC / "jupyter-lite.json"
    data = json.loads(config.read_text())
    data.setdefault("jupyter-config-data", {})["exposeAppInBrowser"] = True
    config.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    global PUBLIC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jupyter", help="Existing Jupyter executable with the pinned JupyterLite packages")
    parser.add_argument("--micromamba", help="Existing micromamba 2.9.0 executable")
    parser.add_argument("--output-dir", type=Path, help="Alternative static runtime output directory")
    parser.add_argument("--bundle-only", action="store_true", help="Refresh Python source after assets have been built")
    args = parser.parse_args()
    if args.output_dir:
        PUBLIC = args.output_dir.resolve()
    if not args.bundle_only:
        build_runtime(args.jupyter, args.micromamba)
    bundle_sources()


if __name__ == "__main__":
    main()
