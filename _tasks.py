import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def task_docs(clean=False, builder="html"):
    """Build the html docs using Sphinx."""
    # Delete Autogenerated files from previous run
    if clean:
        shutil.rmtree(str(HERE / "docs/modules/generated"), ignore_errors=True)
        shutil.rmtree(str(HERE / "docs/_build"), ignore_errors=True)
        shutil.rmtree(str(HERE / "docs/auto_examples"), ignore_errors=True)

    subprocess.run(f"sphinx-build -b {builder} -j auto -d docs/_build docs docs/_build/html", shell=True, check=True)


def update_version_strings(file_path, new_version):
    # taken from:
    # https://stackoverflow.com/questions/57108712/replace-updated-version-strings-in-files-via-python
    version_regex = re.compile(r"(^_*?version_*?\s*=\s*\")(\d+\.\d+\.\d+-?\S*)\"", re.M)
    with open(file_path, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(
            re.sub(
                version_regex,
                lambda match: f'{match.group(1)}{new_version}"',
                content,
            )
        )
        f.truncate()


def update_version(version):
    subprocess.run(["poetry", "version", version], shell=False, check=True)
    new_version = (
        subprocess.run(["poetry", "version"], shell=False, check=True, capture_output=True)
        .stdout.decode()
        .strip()
        .split(" ", 1)[1]
    )
    update_version_strings(HERE.joinpath("gaitlink/__init__.py"), new_version)


def task_update_version():
    update_version(sys.argv[1])
