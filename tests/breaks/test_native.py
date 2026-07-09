"""
Tests that native break-aware route evaluation (enabled by setting break
parameters on the vehicle type) matches the reference post-hoc simulation in
:func:`pyvrp.breaks.plan_route_breaks`. The reference is run on a
breaks-disabled vehicle, so its schedule is break-less and it inserts the
breaks itself; the native side uses a breaks-enabled vehicle and must arrive
at the same duration and time warp.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyvrp import Client, Depot, Location, ProblemData, Route, VehicleType
from pyvrp.breaks import DriverRules, plan_route_breaks

# Breaks only: the daily limits are pushed out of reach so no daily rest is
# ever inserted, matching the native code, which currently models only breaks.
_MAX_CONTINUOUS = 270
_BREAK = 45
BREAKS_ONLY = DriverRules(
    max_continuous_driving=_MAX_CONTINUOUS,
    break_duration=_BREAK,
    max_daily_driving=10**15,
    daily_rest_duration=10**15,
)


def _data(dur, windows, service, *, breaks):
    dur = np.asarray(dur, dtype=np.int64)
    num_locs = dur.shape[0]
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

    kwargs = (
        dict(max_continuous_driving=_MAX_CONTINUOUS, break_duration=_BREAK)
        if breaks
        else {}
    )
    vehicle = VehicleType(1, start_depot=0, end_depot=0, **kwargs)
    return ProblemData(locations, clients, depots, [vehicle], [dur], [dur])


CASES = [
    # (name, duration matrix, windows, service, visited client indices)
    ("far_client", [[0, 600], [600, 0]], {}, 0, [0]),
    ("just_over_limit", [[0, 300], [300, 0]], {}, 0, [0]),
    ("exact_limit_boundary", [[0, 270], [270, 0]], {}, 0, [0]),
    ("just_under_limit", [[0, 269], [269, 0]], {}, 0, [0]),
    ("with_service", [[0, 400], [400, 0]], {}, 30, [0]),
    (
        "two_clients",
        [[0, 200, 400], [200, 0, 200], [400, 200, 0]],
        {},
        10,
        [0, 1],
    ),
    ("break_causes_time_warp", [[0, 600], [600, 0]], {1: (0, 500)}, 0, [0]),
    ("wait_absorbs_break", [[0, 250], [250, 0]], {1: (300, None)}, 0, [0]),
]


@pytest.mark.parametrize(
    ("name", "dur", "windows", "service", "visits"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_native_matches_plan_route_breaks(name, dur, windows, service, visits):
    """
    Native break-aware ``Route.duration()`` and ``Route.time_warp()`` equal the
    span and time warp that ``plan_route_breaks`` computes for the same route.
    """
    ref_data = _data(dur, windows, service, breaks=False)
    schedule = plan_route_breaks(Route(ref_data, visits, 0), ref_data, BREAKS_ONLY)
    ref_span = schedule.entries[-1].end_time - schedule.entries[0].start_time

    data = _data(dur, windows, service, breaks=True)
    route = Route(data, visits, 0)

    assert_equal(route.duration(), ref_span)
    assert_equal(route.time_warp(), schedule.time_warp)


def test_breaks_extend_duration_over_disabled():
    """
    A route long enough to require a break has a strictly larger duration when
    breaks are enabled than when they are not.
    """
    dur = [[0, 600], [600, 0]]
    disabled = Route(_data(dur, {}, 0, breaks=False), [0], 0)
    enabled = Route(_data(dur, {}, 0, breaks=True), [0], 0)

    # Four 45-minute breaks (two each way) are inserted.
    assert_equal(enabled.duration() - disabled.duration(), 4 * _BREAK)
