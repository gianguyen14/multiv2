from dataclasses import dataclass


@dataclass(frozen=True)
class FrameIdPolicy:
    mode: str = "zero_based"

    def __post_init__(self):
        if self.mode not in {"zero_based", "one_based"}:
            raise ValueError("frame ID policy must be zero_based or one_based")

    def to_submission_frame_id(self, source_frame_index_zero_based: int) -> int:
        if source_frame_index_zero_based < 0:
            raise ValueError("source frame index must be non-negative")
        return source_frame_index_zero_based + (self.mode == "one_based")
