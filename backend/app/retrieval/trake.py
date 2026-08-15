from dataclasses import dataclass

from backend.app.native import align_trake_events


@dataclass(frozen=True)
class EventCandidate:
    video_id: str
    frame_id: int
    score: float


@dataclass(frozen=True)
class TRAKEResult:
    video_id: str
    frame_ids: list[int]
    score: float


class TRAKEAligner:
    def __init__(self, transition_penalty=0.0, max_gap=None):
        self.transition_penalty = transition_penalty
        self.max_gap = max_gap

    def align(self, candidates_by_event):
        if not candidates_by_event or any(not candidates for candidates in candidates_by_event):
            return None

        videos = set.intersection(
            *(
                set(candidate.video_id for candidate in candidates)
                for candidates in candidates_by_event
            )
        )
        results = []

        for video_id in sorted(videos):
            events = [
                sorted(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.video_id == video_id
                    ),
                    key=lambda item: item.frame_id,
                )
                for candidates in candidates_by_event
            ]
            if any(not event for event in events):
                continue

            aligned = align_trake_events(
                [[candidate.frame_id for candidate in event] for event in events],
                [[candidate.score for candidate in event] for event in events],
                transition_penalty=float(self.transition_penalty),
                max_gap=self.max_gap,
            )
            if aligned is None:
                continue

            score, path = aligned
            results.append(TRAKEResult(video_id, path, score))

        # Preserve the original global tie-break: highest score, then highest video_id.
        return max(results, default=None, key=lambda item: (item.score, item.video_id))
