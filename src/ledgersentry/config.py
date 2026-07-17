"""
One settings object for everything that was previously a per-module constant.

Precedence, highest first:
  1. constructor kwargs (tests)
  2. environment variables, prefixed LEDGERSENTRY_  (e.g. LEDGERSENTRY_MODEL=logreg)
  3. ledgersentry.yaml at the repo root, if present (see ledgersentry.example.yaml)
  4. the defaults below - which are exactly the values behind the committed
     metrics, so a repo with no yaml and no env vars reproduces them unchanged.

Deliberately NOT here: anything with only one imaginable value. The canonical
schema column names, the f_ feature prefix, and decision labels are contracts,
not tunables; putting them in config would just be a second place for them to
be wrong.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEDGERSENTRY_",
        yaml_file=REPO_ROOT / "ledgersentry.yaml",
        extra="forbid",  # a typo'd option is an error, not a silent no-op
    )

    # paths
    data_dir: Path = REPO_ROOT / "data"
    artifact_dir: Path = REPO_ROOT / "artifacts"

    # split + training (the values behind the committed ULB metrics)
    test_size: float = 0.2
    model: str = "hist_gbdt"
    random_state: int = 42
    max_iter: int = 200
    learning_rate: float = 0.1

    # evaluation: the review thresholds train.py sweeps for the committed curve
    review_thresholds: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

    # calibration: fraction of the TRAIN window held out (temporally last) to
    # fit the calibrator on - data the calibration model never trained on
    calibration_size: float = 0.2

    # bootstrap: resamples behind the committed 95% intervals on the headline.
    # 1000 is the convention and is plenty for a 95% interval; the Monte Carlo
    # error on the 2.5/97.5 percentiles is small next to the sampling error the
    # interval is measuring, which on 75 positives is what dominates.
    bootstrap_resamples: int = 1000

    # serving
    max_batch: int = 10_000

    # drift monitoring: PSI bin count + the conventional credit-scoring
    # rule-of-thumb thresholds (<0.1 stable, 0.1-0.25 watch, >=0.25 alert)
    psi_bins: int = 10
    psi_watch: float = 0.1
    psi_alert: float = 0.25

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings, loaded once. Tests construct Settings() directly
    (or monkeypatch env vars and call reset_settings())."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
