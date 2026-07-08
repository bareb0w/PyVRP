.. module:: pyvrp.breaks
   :synopsis: Driver break and overnight-rest planning


Breaks
======

The :mod:`pyvrp.breaks` module makes routes comply with driving-time
regulations such as the European Regulation (EC) No 561/2006, by assigning
breaks and daily (overnight) rests *between* a route's stops. The rules are
fully configurable (with EU defaults), so other regimes or custom company
policies can be modelled as well.

Typical usage is to solve a problem as usual and then annotate the solution
with :func:`~pyvrp.breaks.schedule.plan_breaks`, or to use
:func:`~pyvrp.breaks.solve.solve_with_breaks`, which solves and plans in one
step while making the optimizer reserve room for the required rest.

.. automodule:: pyvrp.breaks.rules

   .. autoclass:: DriverRules
      :members:

   .. autodata:: EU_RULES

.. automodule:: pyvrp.breaks.schedule

   .. autoclass:: EntryType
      :members:

   .. autoclass:: ScheduleEntry
      :members:

   .. autoclass:: CompliantSchedule
      :members:
      :special-members: __len__, __iter__

   .. autofunction:: plan_route_breaks

   .. autofunction:: plan_breaks

.. automodule:: pyvrp.breaks.solve

   .. autofunction:: break_aware_durations

   .. autofunction:: solve_with_breaks
