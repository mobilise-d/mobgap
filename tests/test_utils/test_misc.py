import os
import subprocess
import sys

from mobgap.utils import misc


def test_get_env_var_loads_dotenv_from_project_root(monkeypatch, tmp_path):
    env_name = "MOBGAP_TEST_PROJECT_ROOT_ENV"
    monkeypatch.delenv(env_name, raising=False)

    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(f"{env_name}=from_project_root\n")
    outside_project = tmp_path / "outside"
    outside_project.mkdir()

    monkeypatch.setattr(misc, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(outside_project)

    assert misc.get_env_var(env_name) == "from_project_root"


def test_get_env_var_ignores_dotenv_from_caller_directory(tmp_path):
    env_name = "MOBGAP_TEST_DOTENV_PRECEDENCE"
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(f"{env_name}=from_project_root\n")
    caller_directory = tmp_path / "caller"
    caller_directory.mkdir()
    (caller_directory / ".env").write_text(f"{env_name}=from_caller_directory\n")

    script = (
        "from pathlib import Path; "
        "from mobgap.utils import misc; "
        f"misc.PROJECT_ROOT = Path({str(project_root)!r}); "
        f"print(misc.get_env_var({env_name!r}))"
    )
    env = os.environ.copy()
    env.pop(env_name, None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=caller_directory,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "from_project_root"
