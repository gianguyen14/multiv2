import pytest

from backend.app.video.frame_id_policy import FrameIdPolicy


def test_zero_and_one_based_submission_policies_preserve_internal_index():
    assert FrameIdPolicy("zero_based").to_submission_frame_id(1534) == 1534
    assert FrameIdPolicy("one_based").to_submission_frame_id(1534) == 1535


def test_rejects_invalid_frame_policy_and_index():
    with pytest.raises(ValueError):
        FrameIdPolicy("timestamp")
    with pytest.raises(ValueError):
        FrameIdPolicy().to_submission_frame_id(-1)
