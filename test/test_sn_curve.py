import numpy as np
import pytest

from pybeam.datamodels import SnCurveDnv, SnCurveEurocode


def test_dnv_sn_curve():
    """Check that the number of cycles to failure is computed correctly."""
    dnv_c2 = SnCurveDnv(
        m_1=3.0,
        log_a_bar_1=12.301,
        log_a_bar_2=15.835,
        limit_at_1e7=58.48,
        k=0.15,
        t=25e-3,  # t = t_ref if t < t_ref
        t_ref=25e-3,
        m_2=5.0,
    )

    # We simply use the stress range for 10^7 cycles
    n = int(dnv_c2.get_cycles_at_range(dnv_c2.limit_at_1e7))

    # Check that result is within 0.1% of the value given by DNV.
    assert pytest.approx(n, rel=0.001) == 1e7


def test_eurocode_sn_curve():
    """Test that SN curve from EN 1993 1-9 is correctly defined."""
    sn_cat_125 = SnCurveEurocode(d_sigma_c=125)

    # Test at detail category low-cycle reference value
    assert sn_cat_125.get_cycles_at_range(125) == 2e6
    # Test at detail category high-cycle reference value
    assert sn_cat_125.get_cycles_at_range(sn_cat_125.d_sigma_d) == 5e6
    # Test at detail category cutoff limit
    assert sn_cat_125.get_cycles_at_range(sn_cat_125.d_sigma_l) == pytest.approx(1e8, rel=0.000001)
    # Test below detail category cutoff limit
    assert sn_cat_125.get_cycles_at_range(sn_cat_125.d_sigma_l - 1) == np.inf
