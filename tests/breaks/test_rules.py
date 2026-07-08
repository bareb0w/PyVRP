import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from pyvrp.breaks import EU_RULES, DriverRules


def test_eu_rules_defaults():
    """
    The EU default rules should encode the common Regulation (EC) 561/2006
    limits, in minutes.
    """
    assert_equal(EU_RULES.max_continuous_driving, 270)  # 4.5h
    assert_equal(EU_RULES.break_duration, 45)
    assert_equal(EU_RULES.max_daily_driving, 540)  # 9h
    assert_equal(EU_RULES.daily_rest_duration, 660)  # 11h
    assert_(EU_RULES.max_daily_duty is None)


def test_rules_are_frozen():
    """
    Rules are immutable value objects.
    """
    rules = DriverRules()
    with assert_raises(Exception):
        rules.break_duration = 30  # type: ignore


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_continuous_driving": 0},
        {"max_continuous_driving": -1},
        {"break_duration": 0},
        {"max_daily_driving": 0},
        {"daily_rest_duration": 0},
        {"max_daily_duty": 0},
        {"max_daily_duty": -5},
        # Daily driving cannot be smaller than a single continuous stint.
        {"max_continuous_driving": 300, "max_daily_driving": 200},
        # Working day cannot be shorter than the daily driving limit.
        {"max_daily_driving": 540, "max_daily_duty": 500},
    ],
)
def test_rules_validation_raises(kwargs):
    """
    Invalid rule combinations should raise a ValueError.
    """
    with assert_raises(ValueError):
        DriverRules(**kwargs)


def test_rules_valid_construction():
    """
    A sensible custom (non-EU) configuration should construct fine.
    """
    rules = DriverRules(
        max_continuous_driving=240,
        break_duration=30,
        max_daily_driving=600,
        daily_rest_duration=600,
        max_daily_duty=780,
    )

    assert_equal(rules.max_continuous_driving, 240)
    assert_equal(rules.max_daily_duty, 780)
