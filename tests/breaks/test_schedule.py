import types

import numpy as np
from numpy.testing import assert_, assert_equal

from pyvrp import (
    Client,
    Depot,
    Location,
    ProblemData,
    Route,
    Solution,
    VehicleType,
)
from pyvrp.breaks import (
    DriverRules,
    EntryType,
    plan_breaks,
    plan_route_breaks,
)


def _data(dur, *, service=0, windows=None, start_late=None):
    """
    Builds a simple line instance: location 0 is the depot, locations 1.. are
    clients. ``dur`` is the full duration (and distance) matrix. ``windows`` is
    an optional ``{location: (tw_early, tw_late)}`` mapping.
    """
    dur = np.asarray(dur, dtype=np.int64)
    num_locs = dur.shape[0]
    windows = windows or {}

    locations = [Location(x=float(i), y=0.0) for i in range(num_locs)]
    depots = [Depot(location=0)]
    clients = []
    for loc in range(1, num_locs):
        early, late = windows.get(loc, (0, None))
        extra = {"tw_late": late} if late is not None else {}
        clients.append(
            Client(
                location=loc, service_duration=service, tw_early=early, **extra
            )
        )

    extra = {"start_late": start_late} if start_late is not None else {}
    vehicle = VehicleType(1, start_depot=0, end_depot=0, **extra)

    return ProblemData(locations, clients, depots, [vehicle], [dur], [dur])


def _route(data):
    """
    A single route visiting every client in index order. Routes take client
    indices (0-based), which map to locations 1.. in these instances.
    """
    return Route(data, list(range(data.num_clients)), 0)


def _assert_monotone(schedule):
    """
    Times should be non-decreasing and well-formed.
    """
    for prev, cur in zip(schedule.entries, schedule.entries[1:]):
        assert_(cur.start_time >= prev.end_time)
        assert_(cur.end_time >= cur.start_time)


def test_no_rest_needed_on_short_route():
    """
    A route well within the driving limits gets no breaks or rests.
    """
    dur = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=1000,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    schedule = plan_route_breaks(_route(data), data, rules)

    assert_equal(schedule.num_breaks, 0)
    assert_equal(schedule.num_daily_rests, 0)
    assert_equal(schedule.num_days, 1)
    assert_equal(schedule.total_driving, 300)
    assert_(schedule.is_feasible)
    # Only the stops: start depot, two clients, end depot.
    assert_equal(len(schedule), 4)
    _assert_monotone(schedule)


def test_single_break_is_inserted_mid_leg():
    """
    Once 4.5h of continuous driving is reached, a break is inserted part-way
    along the leg, at the exact point the limit is hit.
    """
    dur = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    schedule = plan_route_breaks(_route(data), data, rules)

    assert_equal(schedule.num_breaks, 1)
    assert_equal(schedule.num_daily_rests, 0)
    assert_equal(schedule.total_driving, 300)
    assert_(schedule.is_feasible)

    (rest,) = schedule.rests
    assert_equal(rest.type, EntryType.BREAK)
    assert_equal(rest.start_time, 150)  # 150 units of driving from depot
    assert_equal(rest.end_time, 195)  # plus the 45 minute break
    assert_equal(rest.location, 1)  # taken en route, after leaving client 1
    assert_(rest.idx is None)
    _assert_monotone(schedule)


def test_multiple_breaks_on_a_long_route():
    """
    Long legs need several breaks; the continuous-driving counter resets after
    each one.
    """
    dur = [[0, 400], [100, 0]]  # depot -> c1 = 400, c1 -> depot = 100
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    schedule = plan_route_breaks(_route(data), data, rules)

    # 400 driving -> break after 150 and after 300; return leg of 100 pushes
    # the counter to 150 again -> one more break.
    assert_equal(schedule.num_breaks, 3)
    assert_equal(schedule.num_daily_rests, 0)
    assert_equal(schedule.total_driving, 500)
    _assert_monotone(schedule)


def test_daily_rest_and_multiple_days():
    """
    Once the daily driving limit is reached, an overnight rest is inserted and
    a new day begins.
    """
    dur = [[0, 500], [100, 0]]  # depot -> c1 = 500, c1 -> depot = 100
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=200,
        daily_rest_duration=660,
    )

    schedule = plan_route_breaks(_route(data), data, rules)

    assert_equal(schedule.num_daily_rests, 2)
    assert_equal(schedule.num_days, 3)
    assert_equal(schedule.num_breaks, 3)
    assert_equal(schedule.total_driving, 600)
    assert_equal(schedule.total_rest, 2 * 660)

    first_rest = next(e for e in schedule if e.type == EntryType.DAILY_REST)
    assert_equal(first_rest.start_time, 245)  # 150 + 45 break + 50 driving
    assert_equal(first_rest.end_time, 905)  # plus 11h (660) rest
    _assert_monotone(schedule)


def test_waiting_time_absorbs_a_break():
    """
    A sufficiently long wait for a time window lets the driver rest, avoiding a
    break that would otherwise be required.
    """
    # depot -> c1 = 100, c1 -> c2 = 100, c2 -> depot = 50.
    dur = [[0, 100, 100], [100, 0, 100], [50, 100, 0]]
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    # Without a window, the second leg triggers a break.
    base = _data(dur)
    without = plan_route_breaks(_route(base), base, rules)
    assert_equal(without.num_breaks, 1)

    # With client 1 opening late (and the shift pinned to start at 0), the
    # vehicle waits 100 at client 1; that wait absorbs the pending break.
    data = _data(dur, windows={1: (200, None)}, start_late=0)
    with_wait = plan_route_breaks(_route(data), data, rules)
    assert_equal(with_wait.num_breaks, 0)
    assert_equal(with_wait.total_driving, 250)
    _assert_monotone(with_wait)


def test_inserting_a_break_can_violate_a_time_window():
    """
    Inserting the legally required break may push a stop past its time window;
    this is reported as time warp and makes the schedule infeasible.
    """
    dur = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]

    # Client 2 must be served by time 200.
    windows = {2: (0, 200)}
    tight = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    data = _data(dur, windows=windows)
    schedule = plan_route_breaks(_route(data), data, tight)

    # A break at 150 delays arrival at client 2 to 245, i.e. 45 late.
    assert_equal(schedule.num_breaks, 1)
    assert_equal(schedule.time_warp, 45)
    assert_(not schedule.is_feasible)

    # Without the break requirement, client 2 is reached at 200 -- on time.
    loose = DriverRules(
        max_continuous_driving=1000,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )
    feasible = plan_route_breaks(_route(data), data, loose)
    assert_equal(feasible.time_warp, 0)
    assert_(feasible.is_feasible)


def test_plan_breaks_over_solution_and_result():
    """
    ``plan_breaks`` accepts a Solution or a Result-like object and returns one
    schedule per route.
    """
    dur = [[0, 100, 100], [100, 0, 100], [100, 100, 0]]
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    solution = Solution(data, [[0, 1]])
    schedules = plan_breaks(solution, data, rules)
    assert_equal(len(schedules), 1)
    assert_equal(schedules[0].num_breaks, 1)

    # A Result-like object exposes a ``best`` solution.
    result = types.SimpleNamespace(best=solution)
    schedules = plan_breaks(result, data, rules)
    assert_equal(len(schedules), 1)
    assert_equal(schedules[0].num_breaks, 1)


def test_schedule_str_is_readable():
    """
    The string representation lists days, breaks, and rests without error.
    """
    dur = [[0, 400], [100, 0]]
    data = _data(dur)
    rules = DriverRules(
        max_continuous_driving=150,
        break_duration=45,
        max_daily_driving=100_000,
        daily_rest_duration=660,
    )

    schedule = plan_route_breaks(_route(data), data, rules)
    text = str(schedule)
    assert_("break" in text)
    assert_("day" in text)
