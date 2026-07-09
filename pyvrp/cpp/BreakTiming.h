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
 * Simulates a route's timeline forward from ``startTime``, inserting a driving
 * break of ``breakDuration`` whenever continuous driving would exceed
 * ``maxContinuous`` (a break may fall part-way along a leg). Returns the
 * resulting break-aware total duration and time warp.
 *
 * The stop arrays (``locs``, ``twEarly``, ``twLate``, ``service``) are in
 * visiting order and include the start depot (index 0) and end depot (last).
 * ``startTime`` is assumed to already respect the start depot's window, so no
 * wait is taken there. A wait of at least ``breakDuration`` at a stop absorbs a
 * pending break at no extra cost. Lateness cascades forward: unlike the
 * solver's usual time-warp model, ``now`` is not reset when a stop is late, so
 * the delay carries to later stops -- matching a real driver's clock and
 * :func:`pyvrp.breaks.schedule.plan_route_breaks`.
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
                                  Duration const maxDuration)
{
    Duration now = startTime;
    Duration drivingSinceBreak = 0;
    Duration travel = 0;
    Duration serviceTotal = service[0];
    Duration waitTotal = 0;
    Duration twTotal = 0;
    Duration breakTotal = 0;

    // Start depot: startTime already respects its window; add its service.
    now += service[0];

    for (size_t i = 1; i != locs.size(); ++i)
    {
        // Drive from the previous stop, inserting breaks as the continuous
        // driving limit is reached along the way.
        Duration remaining = durations(locs[i - 1], locs[i]);
        travel += remaining;
        while (remaining > 0)
        {
            Duration const room = maxContinuous - drivingSinceBreak;
            if (room <= 0)  // a break is required before driving further
            {
                breakTotal += breakDuration;
                now += breakDuration;
                drivingSinceBreak = 0;
                continue;
            }

            Duration const drive = std::min(remaining, room);
            now += drive;
            drivingSinceBreak += drive;
            remaining -= drive;
        }

        // Arrive: wait for the window (a long wait absorbs a pending break),
        // record cascade time warp, then service.
        Duration const wait = std::max<Duration>(twEarly[i] - now, 0);
        if (wait >= breakDuration)
            drivingSinceBreak = 0;
        now = std::max(now, twEarly[i]);
        twTotal += std::max<Duration>(now - twLate[i], 0);
        waitTotal += wait;
        serviceTotal += service[i];
        now += service[i];
    }

    // Total time spent = driving + service + waiting + breaks. Time warp is
    // tracked separately; it does not extend the timeline.
    Duration const duration = travel + serviceTotal + waitTotal + breakTotal;
    Duration timeWarp = twTotal;

    Duration const netDuration = duration - timeWarp;
    if (netDuration > maxDuration)
        timeWarp += netDuration - maxDuration;

    return {duration, timeWarp};
}
}  // namespace pyvrp

#endif  // PYVRP_BREAKTIMING_H
