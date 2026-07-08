from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pyvrp._pyvrp import (
    Activity,
    ProblemData,
    Route,
    ScheduledActivity,
    Solution,
)

from .rules import EU_RULES, DriverRules

if TYPE_CHECKING:
    from pyvrp.Result import Result


class EntryType(Enum):
    """
    The type of a :class:`ScheduleEntry` in a compliant schedule.
    """

    DEPOT = "depot"
    CLIENT = "client"
    BREAK = "break"
    DAILY_REST = "daily_rest"


@dataclass(frozen=True)
class ScheduleEntry:
    """
    A single entry in a :class:`CompliantSchedule`. This is either a stop (a
    depot or client visit, mirroring PyVRP's
    :class:`~pyvrp._pyvrp.ScheduledActivity`) or an inserted rest (a break or a
    daily/overnight rest).

    Attributes
    ----------
    type
        The type of this entry.
    start_time
        Time at which this entry begins.
    end_time
        Time at which this entry completes.
    location
        Index of the physical location of this entry. For stops this is the
        depot or client location; for rests taken *en route* it is the location
        the vehicle most recently departed from.
    idx
        For stops, the depot or client index this entry refers to. ``None`` for
        rests.
    trip
        Index of the trip this entry belongs to.
    day
        Zero-based index of the day this entry belongs to. Incremented after
        every daily rest.
    wait_duration
        For stops, the time spent waiting for the location's time window to
        open. Zero for rests.
    time_warp
        For stops, the amount by which the vehicle arrives after the location's
        time window has closed. Non-zero time warp indicates the stop is served
        late, and thus that inserting rests made the route time-infeasible.
    """

    type: EntryType
    start_time: int
    end_time: int
    location: int
    idx: int | None
    trip: int
    day: int
    wait_duration: int = 0
    time_warp: int = 0

    @property
    def duration(self) -> int:
        """
        Duration of this entry, ``end_time - start_time``.
        """
        return self.end_time - self.start_time

    @property
    def is_rest(self) -> bool:
        """
        Whether this entry is an inserted break or daily rest.
        """
        return self.type in (EntryType.BREAK, EntryType.DAILY_REST)


class CompliantSchedule:
    """
    The break- and rest-augmented schedule of a single :class:`Route`. It
    interleaves the route's stops with the breaks and daily (overnight) rests
    needed to comply with the given :class:`~pyvrp.breaks.rules.DriverRules`.

    Use :func:`plan_route_breaks` to construct one.

    Parameters
    ----------
    entries
        The ordered timeline of stops and rests.
    rules
        The rules used to construct the schedule.
    total_driving
        Total driving (travel) time along the route.
    """

    def __init__(
        self,
        entries: list[ScheduleEntry],
        rules: DriverRules,
        total_driving: int,
    ):
        self._entries = entries
        self._rules = rules
        self._total_driving = total_driving

    @property
    def entries(self) -> list[ScheduleEntry]:
        """
        The ordered timeline of stops and rests.
        """
        return self._entries

    @property
    def rules(self) -> DriverRules:
        """
        The rules used to construct this schedule.
        """
        return self._rules

    @property
    def rests(self) -> list[ScheduleEntry]:
        """
        The inserted rest entries (breaks and daily rests), in order.
        """
        return [e for e in self._entries if e.is_rest]

    @property
    def num_breaks(self) -> int:
        """
        Number of (short) breaks inserted.
        """
        return sum(e.type == EntryType.BREAK for e in self._entries)

    @property
    def num_daily_rests(self) -> int:
        """
        Number of daily (overnight) rests inserted.
        """
        return sum(e.type == EntryType.DAILY_REST for e in self._entries)

    @property
    def num_days(self) -> int:
        """
        Number of days the route spans. Equals ``num_daily_rests + 1``.
        """
        return self.num_daily_rests + 1

    @property
    def total_driving(self) -> int:
        """
        Total driving (travel) time along the route.
        """
        return self._total_driving

    @property
    def total_break(self) -> int:
        """
        Total time spent on (short) breaks.
        """
        return sum(
            e.duration for e in self._entries if e.type == EntryType.BREAK
        )

    @property
    def total_rest(self) -> int:
        """
        Total time spent on daily (overnight) rests.
        """
        return sum(
            e.duration for e in self._entries if e.type == EntryType.DAILY_REST
        )

    @property
    def start_time(self) -> int:
        """
        Time at which the route departs its start depot.
        """
        return self._entries[0].start_time

    @property
    def end_time(self) -> int:
        """
        Time at which the route arrives at its end depot.
        """
        return self._entries[-1].end_time

    @property
    def time_warp(self) -> int:
        """
        Total time warp: the amount by which stops are served after their time
        windows close, once required rests are inserted.
        """
        return sum(e.time_warp for e in self._entries)

    @property
    def is_feasible(self) -> bool:
        """
        Whether the augmented schedule is time-feasible, that is, whether all
        stops are still served within their time windows after inserting the
        required breaks and rests. When this is ``False``, inserting the
        legally required rest pushed one or more stops past their time window.
        """
        return self.time_warp == 0

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __str__(self) -> str:
        lines = [
            f"Compliant schedule ({self.num_days} day(s), "
            f"{self.num_breaks} break(s), {self.num_daily_rests} rest(s))",
        ]
        for entry in self._entries:
            if entry.idx is None:
                label = entry.type.value
            else:
                label = f"{entry.type.value} {entry.idx}"

            line = (
                f"  [{entry.start_time:>6} - {entry.end_time:>6}] "
                f"day {entry.day} trip {entry.trip}: {label}"
            )
            if entry.wait_duration:
                line += f" (wait {entry.wait_duration})"
            if entry.time_warp:
                line += f" (LATE by {entry.time_warp})"

            lines.append(line)

        return "\n".join(lines)


def plan_route_breaks(
    route: Route,
    data: ProblemData,
    rules: DriverRules = EU_RULES,
) -> CompliantSchedule:
    """
    Plans breaks and daily (overnight) rests for a single solved route, so that
    it complies with the given driving-time rules.

    The route's stop sequence is kept fixed. Its timeline is simulated forward
    from the route's start time, accumulating driving time. A break is inserted
    once the maximum continuous driving time is reached, and a daily rest once
    the maximum daily driving time (or the optional working-day limit) is
    reached; see :class:`~pyvrp.breaks.rules.DriverRules`. Rests may be taken
    *anywhere en route*, including part-way along a leg at the exact point a
    limit is hit.

    Parameters
    ----------
    route
        A solved route to plan rests for. Its schedule and start time are read
        from ``data``, so ``route`` must have been created against ``data``.
    data
        The problem data the route was solved against. Its duration matrix
        (for the route's vehicle profile), service durations and time windows
        drive the simulation.
    rules
        The driving-time rules to comply with. Defaults to :data:`EU_RULES`.

    Returns
    -------
    CompliantSchedule
        The break- and rest-augmented schedule.
    """
    veh = data.vehicle_type(route.vehicle_type())
    durations = data.duration_matrix(veh.profile)
    schedule = route.schedule()

    entries: list[ScheduleEntry] = []

    driving_since_break = 0
    driving_today = 0
    day = 0
    day_start = route.start_time()
    now = route.start_time()

    # Emit the start depot stop.
    first = schedule[0]
    first_svc = _service_of(data, first)
    entries.append(_stop_entry(data, first, now, now + first_svc, day, 0, 0))
    now += first_svc
    prev_loc = _location_of(data, first)

    for act in schedule[1:]:
        cur_loc = _location_of(data, act)
        travel = int(durations[prev_loc, cur_loc])

        # Drive the leg, inserting breaks and daily rests as driving-time
        # limits are reached along the way.
        remaining = travel
        while remaining > 0:
            room_break = rules.max_continuous_driving - driving_since_break
            room_day = rules.max_daily_driving - driving_today
            rooms = [room_break, room_day]
            if rules.max_daily_duty is not None:
                rooms.append(rules.max_daily_duty - (now - day_start))

            room = min(rooms)
            if room <= 0:  # a rest is required before we can drive further
                needs_daily = room_day <= 0 or (
                    rules.max_daily_duty is not None
                    and now - day_start >= rules.max_daily_duty
                )
                if needs_daily:
                    end = now + rules.daily_rest_duration
                    entries.append(
                        _rest_entry(
                            EntryType.DAILY_REST,
                            now,
                            end,
                            prev_loc,
                            act.trip,
                            day,
                        )
                    )
                    now = end
                    driving_today = 0
                    driving_since_break = 0
                    day += 1
                    day_start = now
                else:
                    end = now + rules.break_duration
                    entries.append(
                        _rest_entry(
                            EntryType.BREAK, now, end, prev_loc, act.trip, day
                        )
                    )
                    now = end
                    driving_since_break = 0

                continue

            drive = min(remaining, room)
            now += drive
            driving_since_break += drive
            driving_today += drive
            remaining -= drive

        # Arrived at the stop. Account for its time window: wait if we are
        # early, and record time warp if we are late.
        tw_early, tw_late = _time_window_of(data, act)
        wait = max(tw_early - now, 0)
        now += wait

        # A sufficiently long wait lets the driver rest, satisfying a pending
        # break without adding further time. Absorb it so we do not double
        # count rest.
        if wait >= rules.break_duration:
            driving_since_break = 0

        time_warp = max(now - tw_late, 0)
        service = _service_of(data, act)
        entries.append(
            _stop_entry(data, act, now, now + service, day, wait, time_warp)
        )
        now += service
        prev_loc = cur_loc

    return CompliantSchedule(entries, rules, route.travel_duration())


def plan_breaks(
    solution: Result | Solution,
    data: ProblemData,
    rules: DriverRules = EU_RULES,
) -> list[CompliantSchedule]:
    """
    Plans breaks and daily rests for every route in a solution.

    Parameters
    ----------
    solution
        A :class:`~pyvrp.Result.Result` or :class:`~pyvrp._pyvrp.Solution` to
        plan rests for. When a ``Result`` is given, its best solution is used.
    data
        The problem data the solution was solved against.
    rules
        The driving-time rules to comply with. Defaults to :data:`EU_RULES`.

    Returns
    -------
    list[CompliantSchedule]
        One compliant schedule per route in the solution.
    """
    if not isinstance(solution, Solution):  # then it is a Result
        solution = solution.best

    return [plan_route_breaks(r, data, rules) for r in solution.routes()]


def _location_of(data: ProblemData, act: Activity) -> int:
    if act.is_depot():
        return data.depot(act.idx).location
    return data.client(act.idx).location


def _service_of(data: ProblemData, act: Activity) -> int:
    if act.is_depot():
        return data.depot(act.idx).service_duration
    return data.client(act.idx).service_duration


def _time_window_of(data: ProblemData, act: Activity) -> tuple[int, int]:
    if act.is_depot():
        depot = data.depot(act.idx)
        return depot.tw_early, depot.tw_late

    client = data.client(act.idx)
    return client.tw_early, client.tw_late


def _stop_entry(
    data: ProblemData,
    act: ScheduledActivity,
    start: int,
    end: int,
    day: int,
    wait: int,
    time_warp: int,
) -> ScheduleEntry:
    kind = EntryType.DEPOT if act.is_depot() else EntryType.CLIENT
    return ScheduleEntry(
        type=kind,
        start_time=start,
        end_time=end,
        location=_location_of(data, act),
        idx=act.idx,
        trip=act.trip,
        day=day,
        wait_duration=wait,
        time_warp=time_warp,
    )


def _rest_entry(
    kind: EntryType,
    start: int,
    end: int,
    location: int,
    trip: int,
    day: int,
) -> ScheduleEntry:
    return ScheduleEntry(
        type=kind,
        start_time=start,
        end_time=end,
        location=location,
        idx=None,
        trip=trip,
        day=day,
    )
