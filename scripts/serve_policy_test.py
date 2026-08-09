from __future__ import annotations

import copy
import io
import threading

import msgpack
import numpy as np
import pytest
import zmq

from scripts import serve_policy


def _request(observation):
    return {
        "endpoint": "get_action",
        "data": {"observation": observation, "options": None},
    }


def test_codec_round_trip_uses_safe_npy_contract():
    source = {
        "float": np.arange(8, dtype=np.float32),
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "scalar": np.float32(1.25),
    }

    encoded = serve_policy.pack_message(source)
    raw = msgpack.unpackb(encoded, raw=False)
    assert raw["float"][serve_policy.NDARRAY_MARKER] is True
    npy = np.load(io.BytesIO(raw["float"]["as_npy"]), allow_pickle=False)
    np.testing.assert_array_equal(npy, source["float"])

    decoded = serve_policy.unpack_message(encoded)
    np.testing.assert_array_equal(decoded["float"], source["float"])
    np.testing.assert_array_equal(decoded["image"], source["image"])
    assert decoded["scalar"] == pytest.approx(1.25)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("observation.state", np.zeros(7, np.float32), "shape"),
        ("observation.state", np.zeros(8, np.float64), "dtype"),
        ("observation.wrench_compensated", np.zeros(5, np.float32), "shape"),
        ("observation.wrench_compensated", np.zeros(6, np.float64), "dtype"),
        ("observation.images.wrist", np.zeros((480, 640, 3), np.float32), "dtype"),
        ("observation.images.third_view", np.zeros((480, 640, 4), np.uint8), "shape"),
        ("task", "", "non-empty"),
    ],
)
def test_observation_shape_and_dtype_validation(key, value, error):
    observation = serve_policy.make_wire_observation()
    observation[key] = value
    with pytest.raises((TypeError, ValueError), match=error):
        serve_policy.validate_observation(observation)


def test_observation_rejects_uncompensated_or_missing_wrench():
    observation = serve_policy.make_wire_observation()
    observation["observation.wrench"] = observation.pop("observation.wrench_compensated")
    with pytest.raises(ValueError, match="wrench_compensated"):
        serve_policy.validate_observation(observation)


def test_fake_backend_returns_exact_wire_shape_and_dtype():
    backend = serve_policy.FakeBackend()
    server = serve_policy.PolicyServer(backend, host="127.0.0.1", port=0)

    actions, info = server.handle_request(_request(serve_policy.make_wire_observation()))

    assert actions.shape == (16, 8)
    assert actions.dtype == np.float32
    assert info["action_shape"] == [16, 8]
    assert info["model_name"] == "ForceVLA"
    assert "absolute" in info["action_representation"]


class _SizedBackend(serve_policy.FakeBackend):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def infer(self, policy_observation):
        return {"actions": np.zeros((self.horizon, 8), dtype=np.float32)}


def test_server_truncates_only_the_front_of_long_chunks():
    backend = _SizedBackend(20)
    expected = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)
    backend.infer = lambda _observation: {"actions": expected}
    server = serve_policy.PolicyServer(backend, host="127.0.0.1", port=0)

    actions, _ = server.handle_request(_request(serve_policy.make_wire_observation()))

    np.testing.assert_array_equal(actions, expected[:16])


def test_server_rejects_short_chunks_without_repeating_last_step():
    server = serve_policy.PolicyServer(_SizedBackend(15), host="127.0.0.1", port=0)
    with pytest.raises(ValueError, match="only 15 action steps"):
        server.handle_request(_request(serve_policy.make_wire_observation()))


class _FailsOnceBackend(serve_policy.FakeBackend):
    def __init__(self):
        super().__init__()
        self.fail = True

    def infer(self, policy_observation):
        if self.fail:
            self.fail = False
            raise RuntimeError("intentional first-request failure")
        return super().infer(policy_observation)


def test_rep_server_continues_after_request_exception():
    backend = _FailsOnceBackend()
    server = serve_policy.PolicyServer(backend, host="127.0.0.1", port=0)
    stop_event = threading.Event()
    ready_event = threading.Event()
    thread = threading.Thread(
        target=server.serve_forever,
        args=(stop_event,),
        kwargs={"ready_event": ready_event},
        daemon=True,
    )
    thread.start()
    assert ready_event.wait(timeout=3)
    assert server.bound_port is not None

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.connect(f"tcp://127.0.0.1:{server.bound_port}")
    request = _request(copy.deepcopy(serve_policy.make_wire_observation()))
    try:
        socket.send(serve_policy.pack_message(request))
        first_response = serve_policy.unpack_message(socket.recv())
        assert "intentional first-request failure" in first_response["error"]

        socket.send(serve_policy.pack_message(request))
        second_response = serve_policy.unpack_message(socket.recv())
        assert isinstance(second_response, list)
        assert second_response[0].shape == (16, 8)
        assert second_response[0].dtype == np.float32
    finally:
        socket.close(linger=0)
        context.term()
        stop_event.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
