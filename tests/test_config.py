"""
The config contract: defaults are exactly the values behind the committed
metrics, env vars override yaml, and a typo'd option is a loud error.
"""
import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from ledgersentry.config import REPO_ROOT, Settings


def test_defaults_are_the_committed_configuration():
    cfg = Settings()
    assert cfg.model == "hist_gbdt"
    assert cfg.random_state == 42
    assert cfg.max_iter == 200
    assert cfg.learning_rate == 0.1
    assert cfg.test_size == 0.2
    assert cfg.review_thresholds == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    assert cfg.data_dir == REPO_ROOT / "data"
    assert cfg.artifact_dir == REPO_ROOT / "artifacts"
    assert cfg.max_batch == 10_000


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("LEDGERSENTRY_MODEL", "logreg")
    monkeypatch.setenv("LEDGERSENTRY_MAX_ITER", "50")
    cfg = Settings()
    assert cfg.model == "logreg"
    assert cfg.max_iter == 50


def test_yaml_file_is_read_and_env_beats_it(tmp_path, monkeypatch):
    yaml_path = tmp_path / "ledgersentry.yaml"
    yaml_path.write_text("model: logreg\ntest_size: 0.3\n")

    class YamlSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="LEDGERSENTRY_", yaml_file=yaml_path, extra="forbid"
        )

    cfg = YamlSettings()
    assert cfg.model == "logreg"
    assert cfg.test_size == 0.3

    monkeypatch.setenv("LEDGERSENTRY_MODEL", "hist_gbdt")
    cfg = YamlSettings()
    assert cfg.model == "hist_gbdt"  # env wins
    assert cfg.test_size == 0.3  # yaml still applies where env is silent


def test_unknown_option_is_a_loud_error(tmp_path):
    yaml_path = tmp_path / "ledgersentry.yaml"
    yaml_path.write_text("modle: logreg\n")  # typo on purpose

    class YamlSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="LEDGERSENTRY_", yaml_file=yaml_path, extra="forbid"
        )

    with pytest.raises(ValidationError):
        YamlSettings()


def test_train_records_model_name_in_metrics(tmp_path, monkeypatch):
    """Provenance rule: with a registry, metrics must say which model made them."""
    monkeypatch.setenv("LEDGERSENTRY_DATA_DIR", str(tmp_path / "no_data_here"))
    monkeypatch.setenv("LEDGERSENTRY_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("LEDGERSENTRY_MAX_ITER", "20")
    from ledgersentry import config, train

    config.reset_settings()
    try:
        metrics = train.main()
    finally:
        config.reset_settings()
    assert metrics["model"] == "hist_gbdt"
    assert metrics["is_synthetic"] is True
    assert (tmp_path / "artifacts" / "metrics_synthetic.json").exists()
