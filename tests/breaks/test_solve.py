import numpy as np
from numpy.testing import assert_, assert_array_less, assert_equal

from pyvrp import (
    Client,
    Depot,
    Location,
    Model,
    ProblemData,
    VehicleType,
)
from pyvrp.breaks import (
    DriverRules,
    break_aware_durations,
    solve_with_breaks,
)
from pyvrp.breaks.schedule import CompliantSchedule
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxIterations


def _tiny_data(dur):
    dur = np.asarray(dur, dtype=np.int64)
    n = dur.shape[0]
    locations = [Location(x=float(i), y=0.0) for i in range(n)]
    depots = [Depot(location=0)]
    clients = [Client(location=i) for i in range(1, n)]
    vehicle = VehicleType(1, start_depot=0, end_depot=0)
    return ProblemData(locations, clients, depots, [vehicle], [dur], [dur])


def test_break_aware_durations_scales_by_the_right_factor():
    """
    Durations are inflated to reserve break time. With break/continuous equal
    to 1/6 and no daily-rest reservation, a leg of 270 becomes 315.
    """
    dur = [[0, 270], [270, 0]]
    data = _tiny_data(dur)
    rules = DriverRules(
        max_continuous_driving=270,
        break_duration=45,
        max_daily_driving=540,
        daily_rest_duration=660,
    )

    inflated = break_aware_durations(data, rules, reserve_daily_rest=False)
    matrix = inflated.duration_matrix(0)

    assert_equal(matrix[0, 1], 315)  # 270 * (1 + 45/270)
    assert_equal(matrix[0, 0], 0)  # diagonal untouched
    assert_equal(inflated.num_profiles, data.num_profiles)


def test_break_aware_durations_reserves_daily_rest_by_default():
    """
    By default room is also reserved for daily rests, so the factor is larger.
    """
    dur = [[0, 270], [270, 0]]
    data = _tiny_data(dur)
    rules = DriverRules(
        max_continuous_driving=270,
        break_duration=45,
        max_daily_driving=540,
        daily_rest_duration=660,
    )

    inflated = break_aware_durations(data, rules)  # reserve_daily_rest=True
    matrix = inflated.duration_matrix(0)

    # factor = 1 + 45/270 + 660/540 = 2.3889 -> 270 * factor = 645 (floored).
    assert_equal(matrix[0, 1], 645)


def test_break_aware_durations_clamps_large_values():
    """
    Scaling must not overflow the maximum representable value.
    """
    dur = [[0, MAX_VALUE], [MAX_VALUE, 0]]
    data = _tiny_data(dur)
    rules = DriverRules()

    inflated = break_aware_durations(data, rules)
    matrix = inflated.duration_matrix(0)
    assert_array_less(matrix - 1, MAX_VALUE)  # all entries <= MAX_VALUE


def test_solve_with_breaks_returns_compliant_schedules():
    """
    Solving a small long-haul instance and planning breaks returns one feasible
    compliant schedule per route.
    """
    model = Model()
    locs = [model.add_location(x=i, y=0) for i in range(4)]
    depot = model.add_depot(locs[0])
    for loc in locs[1:]:
        model.add_client(loc, service_duration=10)
    model.add_vehicle_type(1, start_depot=depot, end_depot=depot)

    # Long, symmetric legs so that breaks (but no tight windows) are required.
    for i in range(4):
        for j in range(4):
            if i != j:
                dist = 100 * abs(i - j)
                model.add_edge(locs[i], locs[j], distance=dist, duration=dist)

    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    result, schedules = solve_with_breaks(
        model, MaxIterations(50), rules, reserve_daily_rest=False
    )

    assert_(len(schedules) >= 1)
    for schedule in schedules:
        assert_(isinstance(schedule, CompliantSchedule))
        assert_(schedule.total_driving > 0)
        assert_(schedule.is_feasible)  # windows are loose, so no time warp
