"""
Tests native break-and-rest-aware route evaluation against
:func:`pyvrp.breaks.plan_route_breaks` for vehicle types that require daily
(overnight) rests as well as breaks -- i.e. multi-day (tramping) routes. The
reference is run on a rules-disabled vehicle; the native side uses a vehicle
with the full rule set and must match its duration and time warp.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal

from pyvrp import (
    ActivityType,
    Client,
    Depot,
    Location,
    ProblemData,
    Route,
    VehicleType,
)
from pyvrp.breaks import DriverRules, plan_route_breaks
from pyvrp.search._search import Node
from pyvrp.search._search import Route as SearchRoute

# Full EU-style rules: 45-min break after 4.5h driving, 11h rest after 9h.
_RULES = dict(
    max_continuous_driving=270,
    break_duration=45,
    max_daily_driving=540,
    daily_rest_duration=660,
)


def _data(dur, windows, service, *, rules, duty=None):
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

    kwargs = {}
    if rules:
        kwargs = dict(_RULES)
        if duty is not None:
            kwargs["max_daily_duty"] = duty
    vehicle = VehicleType(1, start_depot=0, end_depot=0, **kwargs)
    return ProblemData(locations, clients, depots, [vehicle], [dur], [dur])


CASES = [
    # (name, duration matrix, windows, service, visits, max_daily_duty)
    # A far client: >9h driving each way forces breaks and daily rests.
    ("multi_day_far", [[0, 1200], [1200, 0]], {}, 0, [0], None),
    # Right at the daily-driving limit.
    ("daily_limit_boundary", [[0, 540], [540, 0]], {}, 0, [0], None),
    # Two clients out and back, service too.
    (
        "two_clients_multi_day",
        [[0, 400, 800], [400, 0, 400], [800, 400, 0]],
        {},
        20,
        [0, 1],
        None,
    ),
    # Working-day (duty) cap forces an earlier rest than driving alone would.
    ("duty_cap", [[0, 500], [500, 0]], {}, 0, [0], 600),
    # Overnight wait absorbs a daily rest: the client opens far in the future,
    # so the long wait hosts the 11h rest for free.
    ("overnight_wait_absorbs_rest", [[0, 300], [300, 0]], {1: (2000, None)}, 0, [0], None),
]


def _oracle(dur, windows, service, visits, duty):
    ref_data = _data(dur, windows, service, rules=False)
    rules = DriverRules(**_RULES, **({"max_daily_duty": duty} if duty else {}))
    schedule = plan_route_breaks(Route(ref_data, visits, 0), ref_data, rules)
    span = schedule.entries[-1].end_time - schedule.entries[0].start_time
    return span, schedule.time_warp, schedule.num_daily_rests


@pytest.mark.parametrize(
    ("name", "dur", "windows", "service", "visits", "duty"),
    CASES,
    ids=[c[0] for c in CASES],
)
def test_solution_route_matches_oracle(name, dur, windows, service, visits, duty):
    span, tw, num_rests = _oracle(dur, windows, service, visits, duty)

    data = _data(dur, windows, service, rules=True, duty=duty)
    route = Route(data, visits, 0)

    assert_equal(route.duration(), span)
    assert_equal(route.time_warp(), tw)
    assert num_rests > 0 or name == "overnight_wait_absorbs_rest"


def test_search_route_is_break_only():
    """
    The search-side Route is break-aware but NOT daily-rest aware by design:
    daily rests are handled exactly only on the finalised solution Route, so
    the search's incremental cost bookkeeping stays consistent. A multi-day
    route therefore has a smaller search-side duration (no overnight rest) than
    the exact solution-side duration.
    """
    dur = [[0, 1200], [1200, 0]]  # needs breaks and a daily rest each way
    data = _data(dur, {}, 0, rules=True)

    search = SearchRoute(data, 0)
    search.append(Node(ActivityType.CLIENT, 0))
    search.update()

    solution = Route(data, [0], 0)

    # The solution Route counts the overnight rests; the search Route does not.
    assert search.duration() < solution.duration()
