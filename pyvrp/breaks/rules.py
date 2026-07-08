from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriverRules:
    """
    Configurable driver break and rest rules, used to make routes comply with
    driving-time regulations such as the European Regulation (EC) No 561/2006.

    All durations are expressed in the **same unit as the model's duration
    matrix**. The default values below assume that unit is *minutes*, and
    encode the most commonly used EU limits (see :data:`EU_RULES`). If your
    instance uses a different unit (e.g. seconds), scale these values
    accordingly.

    Parameters
    ----------
    max_continuous_driving
        Maximum uninterrupted driving time before a break is required. Under
        EU rules this is 4.5 hours (270 minutes).
    break_duration
        Duration of a break taken once :attr:`max_continuous_driving` is
        reached. Under EU rules this is 45 minutes. Taking such a break resets
        the continuous-driving counter.
    max_daily_driving
        Maximum total driving time within a single day, after which a daily
        rest is required. Under EU rules this is 9 hours (540 minutes).
    daily_rest_duration
        Duration of the daily (overnight) rest taken once
        :attr:`max_daily_driving` is reached. Under EU rules this is (a
        reduced) 11 hours (660 minutes). Taking a daily rest resets both the
        daily and continuous-driving counters and begins a new day.
    max_daily_duty
        Optional maximum length of the working day (the span from the start of
        a day until the daily rest, including driving, service, breaks and
        waiting). When the working day would exceed this value, a daily rest is
        inserted. Unconstrained when ``None``.

    Raises
    ------
    ValueError
        When any duration is non-positive, when ``max_daily_driving`` is
        smaller than ``max_continuous_driving``, or when ``max_daily_duty`` is
        set but smaller than ``max_daily_driving``.
    """

    max_continuous_driving: int = 270
    break_duration: int = 45
    max_daily_driving: int = 540
    daily_rest_duration: int = 660
    max_daily_duty: int | None = None

    def __post_init__(self):
        if self.max_continuous_driving <= 0:
            raise ValueError("max_continuous_driving must be positive.")

        if self.break_duration <= 0:
            raise ValueError("break_duration must be positive.")

        if self.max_daily_driving <= 0:
            raise ValueError("max_daily_driving must be positive.")

        if self.daily_rest_duration <= 0:
            raise ValueError("daily_rest_duration must be positive.")

        if self.max_daily_driving < self.max_continuous_driving:
            msg = "max_daily_driving must be >= max_continuous_driving."
            raise ValueError(msg)

        if self.max_daily_duty is not None:
            if self.max_daily_duty <= 0:
                raise ValueError("max_daily_duty must be positive.")

            if self.max_daily_duty < self.max_daily_driving:
                msg = "max_daily_duty must be >= max_daily_driving."
                raise ValueError(msg)


#: Default rules encoding common limits of EU Regulation (EC) No 561/2006,
#: expressed in minutes: a 45 minute break after 4.5 hours of continuous
#: driving, and an 11 hour daily rest after 9 hours of daily driving.
EU_RULES = DriverRules()
