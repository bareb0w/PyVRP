#ifndef PYVRP_BREAKTIMING_H
#define PYVRP_BREAKTIMING_H

#include "Matrix.h"
#include "Measure.h"

#include <algorithm>
#include <vector>

namespace pyvrp
{
/**
 * Break-aware total duration and time warp of a route, as computed by
 * :func:`simulateBreaks`.
 */
struct BreakTiming
{
    Duration duration;
    Duration timeWarp;
};

/**
 * Simulates a route's timeline forward from ``startTime``, inserting driving
 * breaks and daily (overnight) rests as the driving-time rules require, and
 * returns the resulting duration and time warp. This is a C++ port of
 * :func:`pyvrp.breaks.schedule.plan_route_breaks`.
 *
 * A break of ``breakDuration`` is taken once continuous driving would exceed
 * ``maxContinuous``; a daily rest of ``dailyRestDuration`` once daily driving
 * would exceed ``maxDailyDriving`` or the working-day span would exceed
 * ``maxDailyDuty``. A rest resets the daily and continuous counters and starts
 * a new day. Rests may fall part-way along a leg. Set a rule's limit to the
 * maximum ``Duration`` (and its rest length to zero) to disable it.
 *
 * The stop arrays (``locs``, ``twEarly``, ``twLate``, ``service``) are in
 * visiting order and include the start depot (index 0) and end depot (last).
 * ``startTime`` already respects the start depot's window, so no wait is taken
 * there. A wait of at least ``dailyRestDuration`` at a stop absorbs a daily
 * rest for free (placed at the end of the wait); a shorter wait of at least
 * ``breakDuration`` absorbs a pending break. Lateness cascades forward: unlike
 * the solver's usual time-warp model, ``now`` is not reset when a stop is
 * late, so the delay carries to later stops -- matching a real driver's clock.
 *
 * Any excess of the net route duration over ``maxDuration`` is folded into the
 * returned time warp, mirroring ``DurationSegment::timeWarp(maxDuration)``.
 */
inline BreakTiming simulateBreaks(Duration const startTime,
                                  std::vector<size_t> const &locs,
                                  std::vector<Duration> const &twEarly,
                                  std::vector<Duration> const &twLate,
                                  std::vector<Duration> const &service,
                                  Matrix<Duration> const &durations,
                                  Duration const maxContinuous,
                                  Duration const breakDuration,
                                  Duration const maxDailyDriving,
                                  Duration const dailyRestDuration,
                                  Duration const maxDailyDuty,
                                  Duration const maxDuration)
{
    auto const inf = std::numeric_limits<Duration>::max();
    bool const breaksOn = breakDuration > 0 && maxContinuous < inf;
    bool const dailyOn
        = dailyRestDuration > 0 && (maxDailyDriving < inf || maxDailyDuty < inf);

    Duration now = startTime;
    Duration drivingSinceBreak = 0;
    Duration drivingToday = 0;
    Duration dayStart = startTime;
    Duration travel = 0;
    Duration serviceTotal = service[0];
    Duration waitTotal = 0;
    Duration twTotal = 0;
    Duration breakTotal = 0;
    Duration restTotal = 0;

    // Start depot: startTime already respects its window; add its service.
    now += service[0];

    for (size_t i = 1; i != locs.size(); ++i)
    {
        // Drive from the previous stop, inserting breaks and daily rests as
        // the driving-time limits are reached along the way.
        Duration remaining = durations(locs[i - 1], locs[i]);
        travel += remaining;
        while (remaining > 0)
        {
            Duration const roomBreak
                = breaksOn ? maxContinuous - drivingSinceBreak : inf;
            Duration const roomDay
                = dailyOn ? maxDailyDriving - drivingToday : inf;
            Duration const roomDuty = (dailyOn && maxDailyDuty < inf)
                                          ? maxDailyDuty - (now - dayStart)
                                          : inf;
            Duration const room = std::min({roomBreak, roomDay, roomDuty});

            if (room <= 0)  // a rest is required before driving further
            {
                bool const needsDaily
                    = dailyOn
                      && (roomDay <= 0
                          || (maxDailyDuty < inf
                              && now - dayStart >= maxDailyDuty));

                if (needsDaily)
                {
                    now += dailyRestDuration;
                    restTotal += dailyRestDuration;
                    drivingToday = 0;
                    drivingSinceBreak = 0;
                    dayStart = now;
                }
                else
                {
                    now += breakDuration;
                    breakTotal += breakDuration;
                    drivingSinceBreak = 0;
                }
                continue;
            }

            Duration const drive = std::min(remaining, room);
            now += drive;
            drivingSinceBreak += drive;
            drivingToday += drive;
            remaining -= drive;
        }

        // Arrive: wait for the window. A wait long enough hosts a daily rest,
        // or (failing that) absorbs a pending break, at no extra cost. Record
        // cascade time warp, then service.
        Duration const wait = std::max<Duration>(twEarly[i] - now, 0);
        if (dailyOn && wait >= dailyRestDuration)
        {
            drivingToday = 0;
            drivingSinceBreak = 0;
            dayStart = twEarly[i];  // the new day starts when the rest ends
        }
        else if (breaksOn && wait >= breakDuration)
            drivingSinceBreak = 0;

        now = std::max(now, twEarly[i]);
        twTotal += std::max<Duration>(now - twLate[i], 0);
        waitTotal += wait;
        serviceTotal += service[i];
        now += service[i];
    }

    // Total time spent = driving + service + waiting + breaks + (forced) daily
    // rests. Absorbed rests are already counted within waiting. Time warp is
    // tracked separately; it does not extend the timeline.
    Duration const duration
        = travel + serviceTotal + waitTotal + breakTotal + restTotal;
    Duration timeWarp = twTotal;

    Duration const netDuration = duration - timeWarp;
    if (netDuration > maxDuration)
        timeWarp += netDuration - maxDuration;

    return {duration, timeWarp};
}
}  // namespace pyvrp

#endif  // PYVRP_BREAKTIMING_H
