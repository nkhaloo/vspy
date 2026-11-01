import numpy as np
import pytest

from vspy.align.textgrid import label_frames


@pytest.mark.parametrize("shift", [1, 5, 10, 0.5, 2.5])
def test_labels_follow_frame_times(shift):
    intervals = [(-5, 0.25, "a"), (2.25, 10, "b"), (10, 15, "c"),
                 (19, 100, "d")]
    times = np.arange(20) * shift
    expected = [next((label for start, end, label in intervals
                      if start <= time < end), None) for time in times]
    assert label_frames(intervals, 20, shift).tolist() == expected


def test_default_shift_preserves_one_millisecond_labels():
    assert label_frames([(1, 3, "a")], 4).tolist() == [None, "a", "a", None]
    assert label_frames([], 0).tolist() == []


@pytest.mark.parametrize("shift", [0, -1, np.nan, np.inf])
def test_invalid_shift_is_rejected(shift):
    with pytest.raises(ValueError, match="finite and positive"):
        label_frames([], 10, shift)
