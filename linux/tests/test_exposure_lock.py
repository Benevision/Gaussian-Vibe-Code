"""Appearance correction is skipped when every photo has the same recorded exposure."""
from types import SimpleNamespace as NS

from splatworker.trainer import exposure_locked


def test_exposure_locked():
    assert exposure_locked([NS(exposure=0.0319)] * 5)
    assert exposure_locked([NS(exposure=0.030), NS(exposure=0.032)])        # within 10 %
    assert not exposure_locked([NS(exposure=0.02), NS(exposure=0.04)])     # auto exposure: correct it
    assert not exposure_locked([NS(exposure=None), NS(exposure=0.03)])     # unknown for some photos
    assert not exposure_locked([NS()])                                      # photo sets without metadata
    assert not exposure_locked([])
