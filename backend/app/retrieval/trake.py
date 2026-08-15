from dataclasses import dataclass


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
        videos = set.intersection(*(set(candidate.video_id for candidate in candidates)
            for candidates in candidates_by_event))
        results = []
        for video_id in sorted(videos):
            events = [sorted((candidate for candidate in candidates if candidate.video_id == video_id),
                key=lambda item: item.frame_id) for candidates in candidates_by_event]
            if any(not ev for ev in events):
                continue
            states = [(candidate.score, [candidate.frame_id]) for candidate in events[0]]
            previous = events[0]
            for candidates in events[1:]:
                next_states = []
                for candidate in candidates:
                    choices = []
                    for prior, (score, path) in zip(previous, states):
                        if not path or score == float("-inf"):
                            continue
                        gap = candidate.frame_id - prior.frame_id
                        if gap > 0 and (self.max_gap is None or gap <= self.max_gap):
                            choices.append((score + candidate.score - self.transition_penalty * gap,
                                path + [candidate.frame_id]))
                    next_states.append(max(choices, default=(float("-inf"), []), key=lambda item: (item[0], [-v for v in item[1]])))
                previous, states = candidates, next_states
            valid_states = [(score, path) for score, path in states if path and len(path) == len(candidates_by_event) and score > float("-inf")]
            if valid_states:
                score, path = max(valid_states, key=lambda item: (item[0], [-v for v in item[1]]))
                results.append(TRAKEResult(video_id, path, score))
        return max(results, default=None, key=lambda item: (item.score, item.video_id))

