import pickle

import numpy as np
import pytest
from numpy.testing import assert_, assert_equal, assert_raises

from pyvrp import VehicleType

_INT_MAX = np.iinfo(np.int64).max
_MAX_SIZE = np.iinfo(np.uint64).max


@pytest.mark.parametrize(
    (
        "capacity",
        "num_available",
        "tw_early",
        "tw_late",
        "shift_duration",
        "max_distance",
        "fixed_cost",
        "unit_distance_cost",
        "unit_duration_cost",
        "start_late",
        "initial_load",
    ),
    [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # num_available must be positive
        (-1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0),  # capacity cannot be negative
        (-100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),  # this is just wrong
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0),  # early > start late
        (0, 1, 1, 1, 0, 0, 0, 0, 0, 2, 0),  # start late > late
        (0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0),  # negative early
        (0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0),  # negative late
        (0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0),  # negative shift_duration
        (0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0),  # negative max_distance
        (0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0),  # negative unit_distance_cost
        (0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0),  # negative unit_duration_cost
        (0, 1, 0, 0, 0, 0, 0, 0, 0, -1, 0),  # negative start late
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1),  # negative initial load
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2),  # initial load exceeds capacity
    ],
)
def test_raises_invalid_data(
    capacity: int,
    num_available: int,
    tw_early: int,
    tw_late: int,
    shift_duration: int,
    max_distance: int,
    fixed_cost: int,
    unit_distance_cost: int,
    unit_duration_cost: int,
    start_late: int,
    initial_load: int,
):
    """
    Tests that the vehicle type constructor raises when given invalid
    arguments.
    """
    with assert_raises(ValueError):
        VehicleType(
            num_available=num_available,
            capacity=[capacity],
            fixed_cost=fixed_cost,
            tw_early=tw_early,
            tw_late=tw_late,
            shift_duration=shift_duration,
            max_distance=max_distance,
            unit_distance_cost=unit_distance_cost,
            unit_duration_cost=unit_duration_cost,
            start_late=start_late,
            initial_load=[initial_load],
        )


@pytest.mark.parametrize(
    ("max_overtime", "unit_overtime_cost"),
    [(-1, 0), (0, -1)],
)
def test_raises_negative_overtime_data(
    max_overtime: int,
    unit_overtime_cost: int,
):
    with assert_raises(ValueError):
        VehicleType(
            max_overtime=max_overtime,
            unit_overtime_cost=unit_overtime_cost,
        )


def test_does_not_raise_for_all_zero_edge_case():
    """
    The vehicle type constructor should allow the following edge case where all
    data has been zeroed out.
    """
    vehicle_type = VehicleType(
        num_available=1,
        capacity=[],
        start_depot=0,
        end_depot=0,
        fixed_cost=0,
        tw_early=0,
        tw_late=0,
        shift_duration=0,
        max_distance=0,
        unit_distance_cost=0,
        unit_duration_cost=0,
        start_late=0,
    )

    assert_equal(vehicle_type.num_available, 1)
    assert_equal(vehicle_type.start_depot, 0)
    assert_equal(vehicle_type.end_depot, 0)
    assert_equal(vehicle_type.capacity, [])
    assert_equal(vehicle_type.fixed_cost, 0)
    assert_equal(vehicle_type.tw_early, 0)
    assert_equal(vehicle_type.tw_late, 0)
    assert_equal(vehicle_type.shift_duration, 0)
    assert_equal(vehicle_type.max_distance, 0)
    assert_equal(vehicle_type.unit_distance_cost, 0)
    assert_equal(vehicle_type.unit_duration_cost, 0)
    assert_equal(vehicle_type.start_late, 0)


def test_default_values():
    """
    Tests that the default values for costs and shift time windows are set
    correctly.
    """
    vehicle_type = VehicleType()
    assert_equal(vehicle_type.num_available, 1)
    assert_equal(vehicle_type.start_depot, 0)
    assert_equal(vehicle_type.end_depot, 0)
    assert_equal(vehicle_type.capacity, [])
    assert_equal(vehicle_type.fixed_cost, 0)
    assert_equal(vehicle_type.tw_early, 0)
    assert_equal(vehicle_type.unit_distance_cost, 1)
    assert_equal(vehicle_type.unit_duration_cost, 0)
    assert_equal(vehicle_type.unit_overtime_cost, 0)
    assert_equal(vehicle_type.name, "")

    # The default value for the following fields is the largest representable
    # integral value.
    assert_equal(vehicle_type.tw_late, _INT_MAX)
    assert_equal(vehicle_type.shift_duration, _INT_MAX)
    assert_equal(vehicle_type.max_duration, _INT_MAX)
    assert_equal(vehicle_type.max_distance, _INT_MAX)

    # The default value for start_late is the value of tw_late.
    assert_equal(vehicle_type.start_late, vehicle_type.tw_late)


def test_attribute_access():
    """
    Smoke test that checks all attributes are equal to the values they were
    given in the constructor's arguments.
    """
    vehicle_type = VehicleType(
        num_available=7,
        start_depot=29,
        end_depot=43,
        capacity=[13],
        fixed_cost=3,
        tw_early=17,
        tw_late=19,
        shift_duration=23,
        max_distance=31,
        unit_distance_cost=37,
        unit_duration_cost=41,
        start_late=18,
        max_overtime=43,
        name="vehicle_type name",
    )

    assert_equal(vehicle_type.num_available, 7)
    assert_equal(vehicle_type.start_depot, 29)
    assert_equal(vehicle_type.end_depot, 43)
    assert_equal(vehicle_type.capacity, [13])
    assert_equal(vehicle_type.fixed_cost, 3)
    assert_equal(vehicle_type.tw_early, 17)
    assert_equal(vehicle_type.tw_late, 19)
    assert_equal(vehicle_type.shift_duration, 23)
    assert_equal(vehicle_type.max_distance, 31)
    assert_equal(vehicle_type.unit_distance_cost, 37)
    assert_equal(vehicle_type.unit_duration_cost, 41)
    assert_equal(vehicle_type.start_late, 18)
    assert_equal(vehicle_type.max_overtime, 43)

    assert_equal(vehicle_type.name, "vehicle_type name")
    assert_equal(str(vehicle_type), "vehicle_type name")


@pytest.mark.parametrize(
    ("shift_duration", "max_overtime", "expected"),
    [
        (_INT_MAX, _INT_MAX, _INT_MAX),  # should not overflow
        (_INT_MAX, 0, _INT_MAX),  # borderline
        (0, _INT_MAX, _INT_MAX),  # borderline
        (_INT_MAX - 1, 1, _INT_MAX),  # check for off-by-one
        (1, _INT_MAX - 1, _INT_MAX),  # check for off-by-one
        (10, 10, 20),  # completely OK, should sum both terms
    ],
)
def test_max_duration(shift_duration: int, max_overtime: int, expected: int):
    """
    Tests that the maximum duration property is correctly computed, and does
    not over- or underflow.
    """
    veh_type = VehicleType(
        shift_duration=shift_duration,
        max_overtime=max_overtime,
    )

    assert_equal(veh_type.shift_duration, shift_duration)
    assert_equal(veh_type.max_overtime, max_overtime)
    assert_equal(veh_type.max_duration, expected)


def test_replace():
    """
    Tests that calling replace() on a VehicleType functions correctly.
    """
    vehicle_type = VehicleType(num_available=7, capacity=[10], name="test")
    assert_equal(vehicle_type.num_available, 7)
    assert_equal(vehicle_type.capacity, [10])
    assert_equal(vehicle_type.name, "test")

    # Replacing the number of available vehicles and name should be reflected
    # in the returned vehicle type, but any other values should remain the same
    # as the original. In particular, capacity should not be changed.
    new = vehicle_type.replace(num_available=5, name="new")
    assert_equal(new.num_available, 5)
    assert_equal(new.capacity, [10])
    assert_equal(new.name, "new")


def test_multiple_capacities():
    """
    Tests that vehicle types correctly handle multiple capacities.
    """
    vehicle_type = VehicleType(capacity=[998, 37], num_available=10)
    assert_equal(vehicle_type.num_available, 10)
    assert_equal(vehicle_type.capacity, [998, 37])


def test_eq():
    """
    Tests the equality operator.
    """
    veh_type1 = VehicleType(num_available=3, profile=0)
    veh_type2 = VehicleType(num_available=3, profile=1)
    assert_(veh_type1 != veh_type2)

    # This vehicle type is equivalent to veh_type1.
    veh_type3 = VehicleType(num_available=3, profile=0)
    assert_(veh_type1 == veh_type3)

    # And some things that are not vehicle types.
    assert_(veh_type1 != "text")
    assert_(veh_type1 != 5)


def test_eq_name():
    """
    Tests that the equality operator considers names.
    """
    assert_(VehicleType(name="1") != VehicleType(name="2"))


def test_pickle():
    """
    Tests that vehicle types can be serialised and unserialised.
    """
    before_pickle = VehicleType(num_available=12, capacity=[3], name="test123")
    bytes = pickle.dumps(before_pickle)
    assert_equal(pickle.loads(bytes), before_pickle)


@pytest.mark.parametrize(
    ("capacity", "initial_load", "exp_capacity", "exp_initial_load"),
    [
        ([0], [0], [0], [0]),
        ([0], [0, 0, 0], [0, 0, 0], [0, 0, 0]),
        ([0, 1, 2], [0], [0, 1, 2], [0, 0, 0]),
        ([1, 2], [1], [1, 2], [1, 0]),
        ([], [], [], []),
    ],
)
def test_load_dimensions_are_padded_with_zeroes(
    capacity: list[int],
    initial_load: list[int],
    exp_capacity: list[int],
    exp_initial_load: list[int],
):
    """
    Tests that any missing load dimensions for the capacity and initial_load
    VehicleType arguments are padded with zeroes.
    """
    vehicle_type = VehicleType(capacity=capacity, initial_load=initial_load)
    assert_equal(vehicle_type.capacity, exp_capacity)
    assert_equal(vehicle_type.initial_load, exp_initial_load)


def test_max_trips(ok_small_multiple_trips):
    """
    Tests that the vehicle type correctly handles the case where max_reloads
    is set to its largest allowed size - then max_trips should not overflow.
    """
    veh_type = ok_small_multiple_trips.vehicle_type(0)
    assert_equal(veh_type.max_reloads, 1)
    assert_equal(veh_type.max_trips, 2)

    # Normally, max_trips == max_reloads + 1, but when max_reloads is at the
    # maximum size, we do not want max_trips to overflow and wrap around to
    # zero. These asserts check that does not happen.
    veh_type = veh_type.replace(max_reloads=_MAX_SIZE)
    assert_equal(veh_type.max_reloads, _MAX_SIZE)
    assert_equal(veh_type.max_trips, _MAX_SIZE)


def test_max_trips_is_one_if_no_reload_depots(ok_small):
    """
    Tests that a vehicle type's max_trips is one if there's no reload depots,
    despite max_reloads being unconstrained.
    """
    veh_type = ok_small.vehicle_type(0)
    assert_equal(veh_type.reload_depots, [])
    assert_equal(veh_type.max_reloads, _MAX_SIZE)
    assert_equal(veh_type.max_trips, 1)


def test_allows_negative_fixed_cost():
    """
    Tests that the vehicle type allows negative fixed costs. This was initially
    not allowed.
    """
    veh_type = VehicleType(fixed_cost=-100)
    assert_equal(veh_type.fixed_cost, -100)


def test_break_params_default_to_disabled():
    """
    By default a vehicle type has no driving breaks: break_duration is zero and
    max_continuous_driving is unconstrained.
    """
    veh_type = VehicleType()
    assert_equal(veh_type.break_duration, 0)
    assert_equal(veh_type.max_continuous_driving, _INT_MAX)


def test_break_params_attribute_access():
    """
    Tests that the break parameters can be set and read back.
    """
    veh_type = VehicleType(max_continuous_driving=270, break_duration=45)
    assert_equal(veh_type.max_continuous_driving, 270)
    assert_equal(veh_type.break_duration, 45)


@pytest.mark.parametrize(
    ("max_continuous_driving", "break_duration"),
    [
        (-1, 0),  # negative max_continuous_driving
        (270, -1),  # negative break_duration
        (_INT_MAX, 45),  # break without a finite continuous-driving limit
    ],
)
def test_raises_invalid_break_data(
    max_continuous_driving: int, break_duration: int
):
    """
    Tests that invalid break parameters are rejected.
    """
    with assert_raises(ValueError):
        VehicleType(
            max_continuous_driving=max_continuous_driving,
            break_duration=break_duration,
        )


def test_daily_rest_params_default_and_access():
    """
    Daily-rest parameters default to disabled and can be set and read back.
    """
    default = VehicleType()
    assert_equal(default.daily_rest_duration, 0)
    assert_equal(default.max_daily_driving, _INT_MAX)
    assert_equal(default.max_daily_duty, _INT_MAX)

    veh_type = VehicleType(
        max_continuous_driving=270,
        break_duration=45,
        max_daily_driving=540,
        daily_rest_duration=660,
        max_daily_duty=780,
    )
    assert_equal(veh_type.max_daily_driving, 540)
    assert_equal(veh_type.daily_rest_duration, 660)
    assert_equal(veh_type.max_daily_duty, 780)

    # Survives replace() and a pickle round-trip.
    replaced = veh_type.replace(daily_rest_duration=600)
    assert_equal(replaced.daily_rest_duration, 600)
    assert_equal(replaced.max_daily_driving, 540)
    assert_(pickle.loads(pickle.dumps(veh_type)) == veh_type)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        (dict(max_daily_driving=-1), "negative max_daily_driving"),
        (dict(daily_rest_duration=-1), "negative daily_rest_duration"),
        (dict(max_daily_duty=-1), "negative max_daily_duty"),
        (dict(daily_rest_duration=660), "rest without a finite daily limit"),
        (
            dict(
                max_continuous_driving=270,
                break_duration=45,
                max_daily_driving=200,  # < max_continuous_driving
                daily_rest_duration=660,
            ),
            "max_daily_driving < max_continuous_driving",
        ),
        (
            dict(
                max_daily_driving=540,
                daily_rest_duration=660,
                max_daily_duty=400,  # < max_daily_driving
            ),
            "max_daily_duty < max_daily_driving",
        ),
    ],
)
def test_raises_invalid_daily_rest_data(kwargs, reason):
    """
    Tests that invalid daily-rest parameter combinations are rejected.
    """
    with assert_raises(ValueError):
        VehicleType(**kwargs)


def test_break_params_replace_and_pickle():
    """
    Tests that break parameters survive replace() and a pickle round-trip, and
    participate in equality.
    """
    veh_type = VehicleType(max_continuous_driving=270, break_duration=45)

    # replace() overrides one field and carries the other.
    replaced = veh_type.replace(break_duration=30)
    assert_equal(replaced.break_duration, 30)
    assert_equal(replaced.max_continuous_driving, 270)
    assert_equal(veh_type.replace(num_available=5).break_duration, 45)

    # pickle round-trip preserves the fields and equality.
    unpickled = pickle.loads(pickle.dumps(veh_type))
    assert_equal(unpickled.break_duration, 45)
    assert_equal(unpickled.max_continuous_driving, 270)
    assert_(unpickled == veh_type)

    # The break fields distinguish vehicle types under equality.
    assert_(veh_type != VehicleType())
