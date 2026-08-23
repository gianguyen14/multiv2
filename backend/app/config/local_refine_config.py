"""Configuration for query-time local dense frame refinement."""

from dataclasses import dataclass
import math
import os


@dataclass(frozen=True)
class LocalRefineConfig:
    """Configuration parameters for local candidate temporal window refinement."""

    enabled: bool = False
    window_seconds: float = 10.0
    interval_seconds: float = 0.5
    max_regions: int = 5

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")

        if (
            self.window_seconds is None
            or not isinstance(self.window_seconds, (int, float))
            or math.isnan(self.window_seconds)
            or math.isinf(self.window_seconds)
            or self.window_seconds <= 0
        ):
            raise ValueError("window_seconds must be a positive and finite number")

        if (
            self.interval_seconds is None
            or not isinstance(self.interval_seconds, (int, float))
            or math.isnan(self.interval_seconds)
            or math.isinf(self.interval_seconds)
            or self.interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be a positive and finite number")

        if (
            self.max_regions is None
            or not isinstance(self.max_regions, int)
            or isinstance(self.max_regions, bool)
            or self.max_regions < 1
        ):
            raise ValueError("max_regions must be an integer >= 1")

    @classmethod
    def from_env(cls) -> "LocalRefineConfig":
        enabled_str = os.getenv("LOCAL_REFINE_ENABLED", "false").lower()
        enabled = enabled_str in ("true", "1", "yes")

        win_val = float(os.getenv("LOCAL_REFINE_WINDOW_SECONDS", "10.0"))
        int_val = float(os.getenv("LOCAL_REFINE_INTERVAL_SECONDS", "0.5"))
        max_reg = int(os.getenv("LOCAL_REFINE_MAX_REGIONS", "5"))

        return cls(
            enabled=enabled,
            window_seconds=win_val,
            interval_seconds=int_val,
            max_regions=max_reg,
        )
