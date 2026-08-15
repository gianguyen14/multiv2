from dataclasses import dataclass

from backend.app.video.video_decoder import decode_frame_indices


@dataclass(frozen=True)
class KISResult:
    video_id: str
    source_frame_index_zero_based: int
    submission_frame_id: int
    score: float
    coarse_frame: int


class DenseTemporalRefiner:
    def __init__(self, encoder, frame_policy, radius_frames=30, stride=1, max_frames=240, batch_size=None):
        self.encoder = encoder
        self.frame_policy = frame_policy
        self.radius_frames = radius_frames
        self.stride = stride
        self.max_frames = max_frames
        self.batch_size = batch_size

    def refine(self, source_path, query, candidate):
        coarse = candidate.representative_frame
        indices = range(max(0, coarse - self.radius_frames), coarse + self.radius_frames + 1, self.stride)
        indices = list(indices)[:self.max_frames]
        try:
            frames = decode_frame_indices(source_path, indices)
            if not frames:
                raise ValueError("empty neighborhood")
            query_vector = self.encoder.encode_text([query])[0]
            images = [frame.image for frame in frames]
            image_vectors = self.encoder.encode_image(images, batch_size=self.batch_size) if self.batch_size is not None else self.encoder.encode_image(images)
            scores = image_vectors @ query_vector
            best = max(range(len(frames)), key=lambda index: (float(scores[index]), -frames[index].source_frame_index_zero_based))
            frame = frames[best].source_frame_index_zero_based
            return KISResult(candidate.video_id, frame, self.frame_policy.to_submission_frame_id(frame),
                float(scores[best]), coarse)
        except Exception:
            return KISResult(candidate.video_id, coarse, self.frame_policy.to_submission_frame_id(coarse),
                candidate.fused_score, coarse)


class KISPipeline:
    def __init__(self, retriever, refiner, source_paths):
        self.retriever = retriever
        self.refiner = refiner
        self.source_paths = source_paths

    def search(self, query, top_k=100, coarse_k=20):
        coarse = self.retriever.search(query, top_k=coarse_k)
        refined = [self.refiner.refine(self.source_paths[item.video_id], query, item) for item in coarse]
        return sorted(refined, key=lambda item: (-item.score, item.video_id,
            item.source_frame_index_zero_based))[:top_k]


def frame_interval_hit(result, ground_truth):
    return result.video_id == ground_truth["video_id"] and ground_truth["start_frame"] <= result.source_frame_index_zero_based <= ground_truth["end_frame"]
