import numpy as np

from libero.evaluation.mock_server import MockPolicy


def test_noop_mode_alternates_open_and_close_chunks():
    policy = MockPolicy(mode="noop", chunk_size=5)
    opened = policy.infer()["actions"]
    closed = policy.infer()["actions"]
    assert opened.dtype == np.float32
    assert opened.shape == (5, 7)
    np.testing.assert_array_equal(opened[:, :6], np.zeros((5, 6), np.float32))
    np.testing.assert_array_equal(opened[:, 6], np.ones(5, np.float32))
    np.testing.assert_array_equal(closed[:, 6], -np.ones(5, np.float32))
