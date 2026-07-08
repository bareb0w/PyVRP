"""
Driver break and overnight-rest planning for PyVRP.

This subpackage adds the ability to make routes comply with driving-time
regulations such as the European Regulation (EC) No 561/2006, by assigning
breaks and daily (overnight) rests between a route's stops. See
:class:`~pyvrp.breaks.rules.DriverRules` for the configurable rules,
:func:`~pyvrp.breaks.schedule.plan_breaks` to annotate solved routes, and
:func:`~pyvrp.breaks.solve.solve_with_breaks` to solve and plan in one step.
"""

from .rules import EU_RULES as EU_RULES
from .rules import DriverRules as DriverRules
from .schedule import CompliantSchedule as CompliantSchedule
from .schedule import EntryType as EntryType
from .schedule import ScheduleEntry as ScheduleEntry
from .schedule import plan_breaks as plan_breaks
from .schedule import plan_route_breaks as plan_route_breaks
from .solve import break_aware_durations as break_aware_durations
from .solve import solve_with_breaks as solve_with_breaks
