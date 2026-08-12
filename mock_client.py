"""Standalone mock client for testing a remote VLA ZMQ policy server.

This file intentionally has no ROS 2, hardware, OpenCV, or cv_bridge imports.
Runtime dependencies are limited to NumPy, MessagePack, and pyzmq.
"""

from __future__ import annotations
from datetime import datetime
import argparse
import io
import json
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any

import msgpack
import numpy as np
import zmq

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
STATE_DIM = 8
WRENCH_DIM = 6
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "dataset-8Hz"

MODEL_ACTION_CHUNK_SIZES = {
    "forcevla": 16,
    "forceloop": 16,
    "openvla-oft": 8,
    "facte": 16,
    "pi0": 16,
}


class MsgSerializer:
    """MessagePack/NumPy codec shared by the real client contract."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            output = io.BytesIO()
            np.save(output, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot MessagePack-encode {type(value).__name__}")

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and "__ndarray_class__" in value:
            return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        return value


class ZmqTestClient:
    """Synchronous REQ client with real send/receive timeouts."""

    def __init__(self, host: str, port: int, timeout_ms: int) -> None:
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data

        assert self.socket is not None
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.ZMQError:
            # REQ sockets must be recreated after a timeout or interrupted cycle.
            self._connect()
            raise

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Policy server error: {response['error']}")
        return response

    def ping(self) -> Any:
        return self.call("ping")

    def reset(self) -> Any:
        return self.call("reset", {"options": None})

    def get_action(self, observation: dict[str, Any]) -> Any:
        return self.call(
            "get_action",
            {"observation": observation, "options": None},
        )

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()

    def __enter__(self) -> ZmqTestClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def parse_float_vector(text: str, expected_dim: int, name: str) -> np.ndarray:
    try:
        values = np.asarray(
            [float(item.strip()) for item in text.split(",")],
            dtype=np.float32,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} contains a non-numeric value") from exc
    if values.shape != (expected_dim,):
        raise argparse.ArgumentTypeError(f"{name} requires {expected_dim} comma-separated values, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(f"{name} must contain finite values")
    return values


def make_image(mode: str, camera: str, seed: int) -> np.ndarray:
    """Create a deterministic uint8 RGB image without OpenCV."""

    if mode == "zeros":
        return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

    if mode == "random":
        camera_offset = 0 if camera == "wrist" else 10000
        rng = np.random.default_rng(seed + camera_offset)
        return rng.integers(
            0,
            256,
            size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            dtype=np.uint8,
        )

    x = np.linspace(0, 255, IMAGE_WIDTH, dtype=np.uint8)
    y = np.linspace(0, 255, IMAGE_HEIGHT, dtype=np.uint8)
    red = np.broadcast_to(x[None, :], (IMAGE_HEIGHT, IMAGE_WIDTH))
    green = np.broadcast_to(y[:, None], (IMAGE_HEIGHT, IMAGE_WIDTH))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8) if camera == "wrist" else 255 - red
    return np.ascontiguousarray(np.stack((red, green, blue), axis=-1))


def make_observation(args: argparse.Namespace) -> dict[str, Any]:
    state = parse_float_vector(args.state, STATE_DIM, "--state")
    wrench = parse_float_vector(args.wrench, WRENCH_DIM, "--wrench")
    wrist = make_image(args.image_mode, "wrist", args.seed)
    third_view = make_image(args.image_mode, "third_view", args.seed)
    task = args.task.strip()
    if not task:
        raise ValueError("--task must not be empty")

    if args.schema == "lerobot":
        observation = {
            "observation.state": state,
            "observation.wrench_compensated": wrench,
            "observation.images.wrist": wrist,
            "observation.images.third_view": third_view,
            "task": task,
        }
    else:
        observation = {
            "observation/state": state,
            "observation/wrench_compensated": wrench,
            "observation/wrist_image": wrist,
            "observation/image": third_view,
            "prompt": task,
        }

    validate_observation(observation, args.schema)
    return observation


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _dataset_rgb(value: Any, key: str) -> np.ndarray:
    """Convert LeRobot's float CHW RGB tensor to the uint8 HWC wire contract."""

    image = _numpy(value)
    if image.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH):
        image = np.moveaxis(image, 0, -1)
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(f"dataset {key}: expected RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if image.dtype != np.uint8:
        raise TypeError(f"dataset {key}: expected uint8/float image, got {image.dtype}")
    return np.ascontiguousarray(image)


def load_dataset_samples(
    args: argparse.Namespace, chunk_size: int
) -> list[tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]]:
    """Load observations and their matching future absolute-action chunks."""

    if args.schema != "lerobot":
        raise ValueError("--dataset requires --schema=lerobot")
    dataset_dir = (Path(args.dataset_root) / args.dataset).resolve()
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text())
    fps = float(info["fps"])

    # Optional dependency: the synthetic/self-test path still only needs NumPy,
    # MessagePack and pyzmq.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        args.dataset,
        root=dataset_dir,
        delta_timestamps={"action": [step / fps for step in range(chunk_size)]},
        download_videos=False,
    )
    samples = []
    for offset in range(args.count):
        index = args.dataset_index + offset * args.dataset_stride
        if not 0 <= index < len(dataset):
            raise IndexError(f"dataset index {index} is outside [0, {len(dataset)})")
        item = dataset[index]
        obs = {
            "observation.state": np.ascontiguousarray(_numpy(item["observation.state"]), dtype=np.float32),
            "observation.wrench_compensated": np.ascontiguousarray(
                _numpy(item["observation.wrench_compensated"]), dtype=np.float32
            ),
            "observation.images.wrist": _dataset_rgb(item["observation.images.wrist"], "observation.images.wrist"),
            "observation.images.third_view": _dataset_rgb(
                item["observation.images.third_view"], "observation.images.third_view"
            ),
            "task": str(item["task"]),
        }
        validate_observation(obs, "lerobot")
        true_actions = np.ascontiguousarray(_numpy(item["action"]), dtype=np.float32)
        valid_steps = ~np.asarray(_numpy(item.get("action_is_pad", np.zeros(chunk_size, bool))), dtype=bool)
        if true_actions.shape != (chunk_size, STATE_DIM):
            raise ValueError(
                f"dataset action at index {index}: expected {(chunk_size, STATE_DIM)}, got {true_actions.shape}"
            )
        sample_info = {
            "dataset": args.dataset,
            "dataset_index": index,
            "episode_index": int(_numpy(item["episode_index"])),
            "frame_index": int(_numpy(item["frame_index"])),
            "task": item["task"],
            "valid_action_steps": int(np.count_nonzero(valid_steps)),
        }
        samples.append((obs, true_actions, valid_steps, sample_info))
    return samples


def validate_observation(observation: dict[str, Any], schema: str) -> None:
    if schema == "lerobot":
        state_key = "observation.state"
        wrench_key = "observation.wrench_compensated"
        wrist_key = "observation.images.wrist"
        third_key = "observation.images.third_view"
        task_key = "task"
    else:
        state_key = "observation/state"
        wrench_key = "observation/wrench_compensated"
        wrist_key = "observation/wrist_image"
        third_key = "observation/image"
        task_key = "prompt"

    expected = {
        state_key: ((STATE_DIM,), np.float32),
        wrench_key: ((WRENCH_DIM,), np.float32),
        wrist_key: ((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
        third_key: ((IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8),
    }
    for key, (shape, dtype) in expected.items():
        value = observation.get(key)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key} must be a NumPy array")
        if value.shape != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {value.shape}")
        if value.dtype != dtype:
            raise TypeError(f"{key}: expected dtype {dtype}, got {value.dtype}")
    if not isinstance(observation.get(task_key), str) or not observation[task_key]:
        raise ValueError(f"{task_key} must be a non-empty string")


def extract_action_response(response: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(response, list | tuple) and len(response) == 2:
        actions, info = response
        return actions, info if isinstance(info, dict) else {"value": info}
    if isinstance(response, dict) and "actions" in response:
        info = {key: value for key, value in response.items() if key != "actions"}
        return response["actions"], info
    return response, {}


def action_time_dimension(actions: Any) -> int | None:
    if isinstance(actions, dict):
        dimensions = []
        for value in actions.values():
            array = np.asarray(value)
            if array.ndim >= 3 and array.shape[0] == 1:
                dimensions.append(int(array.shape[1]))
            elif array.ndim >= 2:
                dimensions.append(int(array.shape[0]))
            else:
                dimensions.append(1)
        return min(dimensions) if dimensions else None

    array = np.asarray(actions)
    if array.ndim >= 3 and array.shape[0] == 1:
        return int(array.shape[1])
    if array.ndim >= 2:
        return int(array.shape[0])
    if array.ndim == 1:
        return 1
    return None


def summarize_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    summary: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size and np.issubdtype(array.dtype, np.number):
        summary.update(
            {
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
            }
        )
    return summary


def summarize_actions(actions: Any) -> Any:
    if isinstance(actions, dict):
        return {key: summarize_array(value) for key, value in actions.items()}
    return summarize_array(actions)


def first_action_step(actions: Any) -> Any:
    def first(value: Any) -> Any:
        array = np.asarray(value)
        if array.ndim >= 3 and array.shape[0] == 1:
            return array[0, 0].tolist()
        if array.ndim >= 2:
            return array[0].tolist()
        return array.tolist()

    if isinstance(actions, dict):
        return {key: first(value) for key, value in actions.items()}
    return first(actions)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    return value


def expected_chunk_size(args: argparse.Namespace) -> int:
    if args.chunk_size > 0:
        return args.chunk_size
    model = args.model.strip().lower()
    if model not in MODEL_ACTION_CHUNK_SIZES:
        supported = ", ".join(sorted(MODEL_ACTION_CHUNK_SIZES))
        raise ValueError(
            f"Unknown --model {args.model!r}; supported: {supported}. Use --chunk-size for a custom model."
        )
    return MODEL_ACTION_CHUNK_SIZES[model]


def print_json(value: Any) -> None:
    print(json.dumps(jsonable(value), ensure_ascii=False, indent=2))


def run_codec_test() -> None:
    source = {
        "state": np.arange(STATE_DIM, dtype=np.float32),
        "image": make_image("gradient", "wrist", 0),
        "task": "codec test",
    }
    restored = MsgSerializer.from_bytes(MsgSerializer.to_bytes(source))
    np.testing.assert_array_equal(restored["state"], source["state"])
    np.testing.assert_array_equal(restored["image"], source["image"])
    assert restored["task"] == source["task"]
    print("[PASS] MessagePack/NumPy codec round-trip")


def fake_server_worker(
    endpoint_queue: queue.Queue[str],
    stop_event: threading.Event,
    chunk_size: int,
    action_dim: int,
) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind("tcp://127.0.0.1:*")
    endpoint_queue.put(socket.getsockopt_string(zmq.LAST_ENDPOINT))

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    request_count = 0
    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if socket not in events:
                continue
            try:
                request = MsgSerializer.from_bytes(socket.recv())
                endpoint = request.get("endpoint", "get_action")
                if endpoint == "ping":
                    response = {"status": "ok", "server": "embedded_mock"}
                elif endpoint == "reset":
                    request_count = 0
                    response = {"status": "ok", "reset": True}
                elif endpoint == "get_action":
                    observation = request.get("data", {}).get("observation", {})
                    if not isinstance(observation, dict):
                        raise ValueError("observation must be a dictionary")
                    request_count += 1
                    actions = np.linspace(
                        0.0,
                        1.0,
                        chunk_size * action_dim,
                        dtype=np.float32,
                    ).reshape(chunk_size, action_dim)
                    response = [
                        actions,
                        {
                            "model_name": "embedded_mock",
                            "request_count": request_count,
                            "action_shape": list(actions.shape),
                        },
                    ]
                else:
                    response = {"error": f"Unknown endpoint: {endpoint}"}
            except Exception as exc:
                response = {"error": str(exc)}
            socket.send(MsgSerializer.to_bytes(response))
    finally:
        socket.close(linger=0)
        context.term()


def start_embedded_server(
    chunk_size: int,
    action_dim: int,
) -> tuple[threading.Thread, threading.Event, str, int]:
    endpoint_queue: queue.Queue[str] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=fake_server_worker,
        args=(endpoint_queue, stop_event, chunk_size, action_dim),
        daemon=True,
    )
    thread.start()
    endpoint = endpoint_queue.get(timeout=3.0)
    address = endpoint.removeprefix("tcp://")
    host, port_text = address.rsplit(":", 1)
    return thread, stop_event, host, int(port_text)


def call_with_retries(
    operation,
    retries: int,
    retry_delay: float,
    operation_name: str,
) -> Any:
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except (zmq.ZMQError, RuntimeError) as exc:
            if attempt >= retries:
                raise
            print(f"[WARN] {operation_name} failed ({attempt}/{retries}): {exc}; retrying in {retry_delay:.1f}s")
            time.sleep(retry_delay)
    raise AssertionError("unreachable")


def run_requests(args: argparse.Namespace) -> int:
    chunk_size = expected_chunk_size(args)
    dataset_samples = load_dataset_samples(args, chunk_size) if args.dataset else None
    observation = dataset_samples[0][0] if dataset_samples else make_observation(args)

    embedded_thread = None
    embedded_stop = None
    host = args.host
    port = args.port
    if args.self_test:
        embedded_thread, embedded_stop, host, port = start_embedded_server(
            chunk_size,
            args.mock_action_dim,
        )
        print(f"[INFO] Embedded mock server started at tcp://{host}:{port}")

    print("[INFO] Test configuration")
    print_json(
        {
            "server": f"tcp://{host}:{port}",
            "model": args.model,
            "expected_chunk_size": chunk_size,
            "schema": args.schema,
            "count": args.count,
            "timeout_ms": args.timeout_ms,
            "image_mode": args.image_mode,
            "task": observation["task" if args.schema == "lerobot" else "prompt"],
            "dataset": args.dataset or None,
            "dataset_index": args.dataset_index if args.dataset else None,
            "dataset_stride": args.dataset_stride if args.dataset else None,
            "observation_shapes": {
                key: list(value.shape) if isinstance(value, np.ndarray) else "string"
                for key, value in observation.items()
            },
        }
    )

    round_trip_times: list[float] = []
    comparison_errors: list[np.ndarray] = []
    comparison_records: list[dict[str, Any]] = []
    mismatches = 0
    try:
        with ZmqTestClient(host, port, args.timeout_ms) as client:
            if not args.skip_ping:
                ping = call_with_retries(
                    client.ping,
                    args.retries,
                    args.retry_delay,
                    "ping",
                )
                print("[PASS] ping")
                print_json(ping)

            if args.reset_first:
                reset = call_with_retries(
                    client.reset,
                    args.retries,
                    args.retry_delay,
                    "reset",
                )
                print("[PASS] reset")
                print_json(reset)

            for index in range(args.count):
                if dataset_samples:
                    observation, true_actions, valid_steps, sample_info = dataset_samples[index]
                else:
                    true_actions = valid_steps = sample_info = None
                started = time.perf_counter()
                response = call_with_retries(
                    lambda observation=observation: client.get_action(observation),
                    args.retries,
                    args.retry_delay,
                    "get_action",
                )
                round_trip_ms = (time.perf_counter() - started) * 1000.0
                round_trip_times.append(round_trip_ms)

                actions, info = extract_action_response(response)
                actual_chunk_size = action_time_dimension(actions)
                chunk_ok = actual_chunk_size == chunk_size
                if not chunk_ok:
                    mismatches += 1

                print(
                    f"[{'PASS' if chunk_ok else 'FAIL'}] request={index + 1}/{args.count} "
                    f"round_trip={round_trip_ms:.2f}ms "
                    f"chunk={actual_chunk_size} expected={chunk_size}"
                )
                print_json(
                    {
                        "actions": summarize_actions(actions),
                        "first_action_step": first_action_step(actions),
                        "info": info,
                    }
                )
                if true_actions is not None and valid_steps is not None:
                    predicted = np.asarray(actions, dtype=np.float32)
                    if predicted.shape != true_actions.shape:
                        mismatches += 1
                        comparison_records.append(
                            {
                                **sample_info,
                                "status": "shape_mismatch",
                                "predicted_shape": list(predicted.shape),
                                "ground_truth_shape": list(true_actions.shape),
                            }
                        )
                    elif not np.any(valid_steps):
                        mismatches += 1
                        comparison_records.append(
                            {
                                **sample_info,
                                "status": "no_valid_action_targets",
                            }
                        )
                    else:
                        errors = predicted[valid_steps] - true_actions[valid_steps]
                        comparison_errors.append(errors)
                        comparison_records.append(
                            {
                                **sample_info,
                                "status": "ok",
                                "valid_step_mask": valid_steps,
                                "ground_truth_actions": true_actions,
                                "predicted_actions": predicted,
                                "errors": predicted - true_actions,
                                "mae": float(np.mean(np.abs(errors))),
                                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                                "per_dim_mae": np.mean(np.abs(errors), axis=0),
                                "per_dim_rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
                                "bias_per_dim": np.mean(errors, axis=0),
                            }
                        )
                if args.print_actions:
                    print("[INFO] Full actions")
                    print_json(actions)

                if index + 1 < args.count and args.interval > 0.0:
                    time.sleep(args.interval)
    finally:
        if embedded_stop is not None:
            embedded_stop.set()
        if embedded_thread is not None:
            embedded_thread.join(timeout=2.0)

    sorted_times = sorted(round_trip_times)
    p95_index = max(0, int(np.ceil(len(sorted_times) * 0.95)) - 1)
    print("[INFO] Summary")
    summary = {
        "requests": len(round_trip_times),
        "chunk_mismatches": mismatches,
        "round_trip_ms": {
            "min": min(round_trip_times),
            "mean": statistics.fmean(round_trip_times),
            "median": statistics.median(round_trip_times),
            "p95": sorted_times[p95_index],
            "max": max(round_trip_times),
        },
    }
    if dataset_samples:
        comparison_report: dict[str, Any] = {
            "dataset": args.dataset,
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "server": f"tcp://{host}:{port}",
            "action_representation": "absolute joint positions (7) + absolute gripper command (1)",
            "action_shape": [chunk_size, STATE_DIM],
            "samples": comparison_records,
        }
    if comparison_errors:
        errors = np.concatenate(comparison_errors, axis=0)
        comparison_report["aggregate"] = {
            "compared_steps": int(errors.shape[0]),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "per_dim_mae": np.mean(np.abs(errors), axis=0),
            "per_dim_rmse": np.sqrt(np.mean(np.square(errors), axis=0)),
            "bias_per_dim": np.mean(errors, axis=0),
        }
    if dataset_samples:
        output_path = Path(args.comparison_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(jsonable(comparison_report), ensure_ascii=False, indent=2) + "\n")
        print(f"[INFO] Action comparison saved to {output_path}")
    print_json(summary)
    return 0 if mismatches == 0 else 2


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS-free mock client for the generic VLA ZMQ protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--model", default="pi0")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="positive value overrides the chunk size selected by --model",
    )
    parser.add_argument(
        "--schema",
        choices=("lerobot", "openpi"),
        default="lerobot",
    )
    parser.add_argument("--task", default="pick up the test object")
    parser.add_argument(
        "--state",
        default="0,0,0,0,0,0,0,0",
        help="8 comma-separated values: 7 left-arm joints and 1 gripper value",
    )
    parser.add_argument(
        "--wrench",
        default="0,0,0,0,0,0",
        help="6 comma-separated values: Fx,Fy,Fz,Mx,My,Mz",
    )
    parser.add_argument(
        "--image-mode",
        choices=("gradient", "zeros", "random"),
        default="gradient",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        choices=("flip_box", "insert_plug", "press_button", "wipe_board"),
        help="load real observations/action targets from this local LeRobot dataset",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=0,
        help="first global frame index used in --dataset mode",
    )
    parser.add_argument(
        "--dataset-stride",
        type=int,
        default=16,
        help="global frame stride between requests in --dataset mode",
    )
    parser.add_argument(
        "--comparison-output",
        default=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_action_comparison.json",
        help="JSON output for per-sample and aggregate dataset action errors",
    )
    parser.add_argument("--skip-ping", action="store_true")
    parser.add_argument("--reset-first", action="store_true")
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument(
        "--codec-test",
        action="store_true",
        help="run the local serialization test before network requests",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="start an embedded fake server and test without an external server",
    )
    parser.add_argument(
        "--mock-action-dim",
        type=int,
        default=8,
        help="action dimension used only by --self-test",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if args.retries <= 0:
        raise ValueError("--retries must be positive")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.interval < 0.0 or args.retry_delay < 0.0:
        raise ValueError("--interval and --retry-delay must be non-negative")
    if args.chunk_size < 0:
        raise ValueError("--chunk-size must be zero or positive")
    if args.mock_action_dim <= 0:
        raise ValueError("--mock-action-dim must be positive")
    if args.dataset_index < 0:
        raise ValueError("--dataset-index must be non-negative")
    if args.dataset_stride <= 0:
        raise ValueError("--dataset-stride must be positive")
    if args.dataset and args.self_test:
        raise ValueError("--dataset compares a real policy response and cannot be combined with --self-test")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
        if args.codec_test or args.self_test:
            run_codec_test()
        return run_requests(args)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
        return 130
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
