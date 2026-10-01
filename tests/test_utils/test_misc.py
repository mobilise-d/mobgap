from mobgap.utils import misc


def test_get_env_var_prefers_environment(monkeypatch):
    env_name = "MOBGAP_TEST_ENV_PRECEDENCE"
    monkeypatch.setenv(env_name, "from_environment")
    assert misc.get_env_var(env_name) == "from_environment"


def test_get_env_var_uses_default_when_missing(monkeypatch):
    env_name = "MOBGAP_TEST_MISSING_ENV_DEFAULT"
    monkeypatch.delenv(env_name, raising=False)
    assert misc.get_env_var(env_name, "fallback") == "fallback"
