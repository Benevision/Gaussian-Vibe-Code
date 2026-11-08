"""optimizerPrecision config key (PROTOCOL.md §3)."""
import pytest

from splatworker.config import ConfigError, TrainingConfig


def test_optimizer_precision_default_and_values():
    assert TrainingConfig.from_dict({}).optimizerPrecision == "fp32"
    assert TrainingConfig.from_dict({"optimizerPrecision": "compact"}).optimizerPrecision == "compact"
    assert TrainingConfig.from_dict({"optimizerPrecision": "COMPACT"}).optimizerPrecision == "compact"
    assert TrainingConfig.from_dict({"optimizerPrecision": None}).optimizerPrecision == "fp32"


@pytest.mark.parametrize("bad", ["int8", "fp16", "", 8, True])
def test_optimizer_precision_rejects_bad_values(bad):
    with pytest.raises(ConfigError):
        TrainingConfig.from_dict({"optimizerPrecision": bad})
