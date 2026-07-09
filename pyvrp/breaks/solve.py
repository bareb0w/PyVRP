from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvrp._pyvrp import Activity, ProblemData, Route
from pyvrp.constants import MAX_VALUE

from .rules import EU_RULES, DriverRules
from .schedule import CompliantSchedule, plan_route_breaks

if TYPE_CHECKING:
    from pyvrp.Model import Model
    from pyvrp.Result import Result
    from pyvrp.stop import StoppingCriterion


def break_aware_durations(
    data: ProblemData,
    rules: DriverRules = EU_RULES,
    *,
    reserve_daily_rest: bool = True,
    reserve_factor: float = 1.0,
) -> ProblemData:
    """
    Returns a copy of ``data`` whose duration matrices are inflated to reserve
    room for the rest time that driving-time rules will require. This lets the
    optimizer account (approximately) for breaks and daily rests while solving,
    even though the solver itself is not break-aware: a leg that takes ``t``
    units of driving is made to appear longer, in proportion to the rest that
    driving ``t`` units incurs on average.

    The inflation is deliberately approximate; :func:`solve_with_breaks`
    refines it with a feedback loop, and the exact rest placement is always
    computed afterwards by
    :func:`~pyvrp.breaks.schedule.plan_route_breaks`.

    Parameters
    ----------
    data
        The problem data to inflate.
    rules
        The driving-time rules to reserve room for. Defaults to
        :data:`~pyvrp.breaks.rules.EU_RULES`.
    reserve_daily_rest
        Whether to also reserve room for daily (overnight) rests, amortised
        over the daily driving limit. Set to ``False`` for single-day routes,
        where only breaks matter. Default ``True``.
    reserve_factor
        Additional multiplier applied to the reserved rest time. Values above
        one reserve more room; used by :func:`solve_with_breaks` to iterate
        towards feasibility. Default one.

    Returns
    -------
    ProblemData
        A copy of ``data`` with inflated duration matrices.
    """
    reserved = rules.break_duration / rules.max_continuous_driving
    if reserve_daily_rest:
        reserved += rules.daily_rest_duration / rules.max_daily_driving

    factor = 1.0 + reserve_factor * reserved

    matrices = []
    for profile in range(data.num_profiles):
        matrix = data.duration_matrix(profile).astype(np.float64)
        scaled = np.minimum(np.rint(matrix * factor), MAX_VALUE)
        scaled = scaled.astype(np.int64)
        np.fill_diagonal(scaled, 0)
        matrices.append(scaled)

    return data.replace(duration_matrices=matrices)


def solve_with_breaks(
    model_or_data: Model | ProblemData,
    stop: StoppingCriterion,
    rules: DriverRules = EU_RULES,
    *,
    seed: int = 0,
    reserve_daily_rest: bool = True,
    max_iters: int = 4,
    growth: float = 1.5,
    **kwargs,
) -> tuple[Result, list[CompliantSchedule]]:
    """
    Solves a routing problem and plans compliant driver breaks and rests for
    it. The problem is solved on duration data inflated by
    :func:`break_aware_durations` so the optimizer leaves room for rest, after
    which exact rests are inserted with
    :func:`~pyvrp.breaks.schedule.plan_route_breaks`. If inserting the required
    rest still pushes any stop past its time window, the reserved room is
    increased and the problem is re-solved, up to ``max_iters`` times.

    Parameters
    ----------
    model_or_data
        The :class:`~pyvrp.Model.Model` or
        :class:`~pyvrp._pyvrp.ProblemData` to solve.
    stop
        Stopping criterion to use for each solve.
    rules
        The driving-time rules to comply with. Defaults to
        :data:`~pyvrp.breaks.rules.EU_RULES`.
    seed
        Seed value for the random number stream. Default zero.
    reserve_daily_rest
        Passed through to :func:`break_aware_durations`. Default ``True``.
    max_iters
        Maximum number of solve attempts, increasing the reserved room each
        time the resulting schedules are not time-feasible. Default four.
    growth
        Factor by which the reserved room grows between attempts. Default 1.5.
    **kwargs
        Additional keyword arguments passed on to :func:`~pyvrp.solve.solve`.

    Returns
    -------
    tuple[Result, list[CompliantSchedule]]
        The best :class:`~pyvrp.Result.Result` found and, for each of its
        routes, the compliant schedule. When a fully feasible set of schedules
        is found, that result is returned immediately; otherwise the result of
        the final attempt is returned.
    """
    import copy

    from pyvrp.solve import solve

    if isinstance(model_or_data, ProblemData):
        data = model_or_data
    else:  # a Model; build its ProblemData
        data = model_or_data.data()

    best: tuple[Result, list[CompliantSchedule]] | None = None
    factor = 1.0
    for attempt in range(max_iters):
        inflated = break_aware_durations(
            data,
            rules,
            reserve_daily_rest=reserve_daily_rest,
            reserve_factor=factor,
        )

        # Stopping criteria are stateful: e.g. MaxRuntime starts its clock on
        # first call and stays expired forever after. Give every attempt a
        # fresh copy so retries get their full budget, rather than
        # terminating immediately after the first attempt used it up.
        result = solve(
            inflated, copy.deepcopy(stop), seed=seed + attempt, **kwargs
        )

        # Re-evaluate the found routes on the original (non-inflated) data so
        # that the planned schedule uses real travel times, then insert rests.
        schedules = []
        for route in result.best.routes():
            inner = list(route)[1:-1]  # drop implicit start and end depots
            activities = [Activity(act.type, act.idx) for act in inner]
            rebuilt = Route(data, activities, route.vehicle_type())
            schedules.append(plan_route_breaks(rebuilt, data, rules))

        best = (result, schedules)
        if all(schedule.is_feasible for schedule in schedules):
            break

        factor *= growth

    assert best is not None
    return best
