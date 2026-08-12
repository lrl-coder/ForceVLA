"""Serve the ForceVLA policy over the robot-side ZeroMQ protocol.

The wire format intentionally matches ``mock_client.py``.  NumPy arrays are
stored as safe ``.npy`` byte strings inside MessagePack; pickle is never used.
The real OpenPI/JAX imports are lazy so ``--dry-run`` and unit tests do not need
a GPU, JAX, or a checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import io
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any, Protocol

import msgpack
import numpy as np
import zmq


LOGGER = logging.getLogger("forcevla.serve_policy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "forcevla_8hz_all_lora"
    / "dataset_8hz_all_lora_v1"
    / "29999"
)
DEFAULT_CONFIG = "forcevla_8hz_all_lora"

ACTION_HORIZON = 16
ACTION_DIM = 8
MODEL_ACTION_DIM = 32
STATE_DIM = 8
WRENCH_DIM = 6
IMAGE_SHAPE = (480, 640, 3)
NDARRAY_MARKER = "__ndarray_class__"

ACTION_NAMES = [
    "Left_Arm_Joint1",
    "Left_Arm_Joint2",
    "Left_Arm_Joint3",
    "Left_Arm_Joint4",
    "Left_Arm_Joint5",
    "Left_Arm_Joint6",
    "Left_Arm_Joint7",
    "left_gripper_command",
]
ACTION_REPRESENTATION = "absolute joint angles (7) + absolute normalized gripper command (1)"
WRENCH_REPRESENTATION = (
    "gravity- and zero-bias-compensated FT300S [Fx,Fy,Fz,Tx,Ty,Tz] in "
    "left_ft300s_link sensor-native axes; units [N,N,N,N.m,N.m,N.m]"
)


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object-dtype NumPy arrays are not supported")
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return {NDARRAY_MARKER: True, "as_npy": output.getvalue()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot MessagePack-encode {type(value).__name__}")


def _msgpack_object_hook(value: dict[str, Any]) -> Any:
    if value.get(NDARRAY_MARKER) is not True:
        return value
    if set(value) != {NDARRAY_MARKER, "as_npy"}:
        raise ValueError(f"invalid NumPy MessagePack object keys: {sorted(value)}")
    payload = value["as_npy"]
    if not isinstance(payload, bytes):
        raise TypeError("NumPy MessagePack as_npy value must be bytes")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise TypeError("decoded value is not a safe NumPy array")
    return array


def pack_message(value: Any) -> bytes:
    """Encode a protocol value without pickle."""

    return msgpack.packb(value, default=_msgpack_default, use_bin_type=True)


def unpack_message(payload: bytes) -> Any:
    """Decode a protocol value without pickle."""

    return msgpack.unpackb(payload, object_hook=_msgpack_object_hook, raw=False, strict_map_key=True)


def _require_array(observation: Mapping[str, Any], key: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    if key not in observation:
        raise ValueError(f"observation is missing required field {key!r}")
    value = observation[key]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{key} must be a NumPy array, got {type(value).__name__}")
    if value.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise TypeError(f"{key} must have dtype {np.dtype(dtype).name}, got {value.dtype}")
    if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains NaN or infinity")
    return np.ascontiguousarray(value)


def validate_observation(observation: Any) -> dict[str, Any]:
    """Validate the exact trained 8 Hz observation contract and adapt it to the policy input."""

    if not isinstance(observation, Mapping):
        raise TypeError(f"observation must be a dictionary, got {type(observation).__name__}")

    state = _require_array(observation, "observation.state", (STATE_DIM,), np.dtype(np.float32))
    wrench = _require_array(
        observation,
        "observation.wrench_compensated",
        (WRENCH_DIM,),
        np.dtype(np.float32),
    )
    wrist = _require_array(observation, "observation.images.wrist", IMAGE_SHAPE, np.dtype(np.uint8))
    third_view = _require_array(
        observation,
        "observation.images.third_view",
        IMAGE_SHAPE,
        np.dtype(np.uint8),
    )
    task = observation.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if not 0.0 <= float(state[-1]) <= 1.0:
        raise ValueError(
            "observation.state[7] must be the normalized gripper opening in [0, 1], "
            f"got {float(state[-1])}"
        )

    # These names are the inference-side inputs consumed by Forcevla_inputs.  The
    # training-only RepackTransform is deliberately not duplicated here.
    return {
        "state": state,
        "force": wrench,
        "image": third_view,
        "wrist_image": wrist,
        "prompt": task,
    }


def make_wire_observation(*, task: str = "warmup") -> dict[str, Any]:
    """Create a valid observation for warmup and tests."""

    return {
        "observation.state": np.zeros((STATE_DIM,), dtype=np.float32),
        "observation.wrench_compensated": np.zeros((WRENCH_DIM,), dtype=np.float32),
        "observation.images.wrist": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "observation.images.third_view": np.zeros(IMAGE_SHAPE, dtype=np.uint8),
        "task": task,
    }


class Backend(Protocol):
    model_name: str
    checkpoint: str
    device: str
    dtype: str
    parameter_count: int
    action_dim: int

    def infer(self, policy_observation: dict[str, Any]) -> Any: ...

    def reset(self) -> None: ...


class FakeBackend:
    """Deterministic CPU-only backend used by ``--dry-run`` and tests."""

    model_name = "ForceVLA"
    checkpoint = "dry-run"
    device = "cpu"
    dtype = "float32"
    parameter_count = 0
    action_dim = ACTION_DIM

    def __init__(self) -> None:
        self.request_count = 0

    def infer(self, policy_observation: dict[str, Any]) -> dict[str, np.ndarray]:
        self.request_count += 1
        state = np.asarray(policy_observation["state"], dtype=np.float32)
        # The trained output contract is absolute, so a stationary chunk is a
        # useful and semantically valid dry-run response.
        return {"actions": np.repeat(state[None, :], ACTION_HORIZON, axis=0)}

    def reset(self) -> None:
        self.request_count = 0


def _configure_jax_environment(device: str) -> None:
    """Select the JAX platform before importing JAX/OpenPI."""

    normalized = device.strip().lower()
    if normalized == "auto":
        return
    if normalized == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        return
    if normalized in {"cuda", "gpu"}:
        os.environ["JAX_PLATFORMS"] = "cuda"
        return
    if normalized.startswith("cuda:") or normalized.startswith("gpu:"):
        try:
            index = int(normalized.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"invalid --device {device!r}; expected auto, cpu, cuda, or cuda:N") from exc
        if index < 0:
            raise ValueError("CUDA device index must be non-negative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        os.environ["JAX_PLATFORMS"] = "cuda"
        return
    raise ValueError(f"invalid --device {device!r}; expected auto, cpu, cuda, or cuda:N")


def _parse_checkpoint_metadata(checkpoint: Path) -> tuple[int, dict[str, tuple[int, ...]]]:
    metadata_path = checkpoint / "params" / "_METADATA"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint parameter metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    shapes: dict[str, tuple[int, ...]] = {}
    parameter_count = 0
    for key, entry in metadata.get("tree_metadata", {}).items():
        shape = tuple(entry.get("value_metadata", {}).get("write_shape", ()))
        if not shape:
            continue
        shapes[key] = shape
        parameter_count += int(np.prod(shape, dtype=np.int64))
    if parameter_count <= 0:
        raise ValueError(f"no parameter arrays found in {metadata_path}")
    return parameter_count, shapes


def _find_shape(shapes: Mapping[str, tuple[int, ...]], *path_parts: str) -> tuple[int, ...]:
    matches = [shape for key, shape in shapes.items() if all(part in key for part in path_parts)]
    if len(matches) != 1:
        raise ValueError(f"checkpoint expected one parameter matching {path_parts}, found {len(matches)}")
    return matches[0]


def _validate_real_contract(train_config: Any, checkpoint: Path) -> None:
    """Fail before loading weights if config/checkpoint do not match this robot contract."""

    from openpi.policies import forcevla_policy
    from openpi.shared import normalize
    from openpi.training import config as training_config

    model = train_config.model
    expected_model_values = {
        "action_dim": MODEL_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "proprio_dim": STATE_DIM,
        "force_dim": WRENCH_DIM,
    }
    for name, expected in expected_model_values.items():
        actual = getattr(model, name, None)
        if actual != expected:
            raise ValueError(f"config model.{name} must be {expected}, got {actual}")

    factory = train_config.data
    if not isinstance(factory, training_config.LeRobotForcevlaDataConfig):
        raise TypeError(f"config data factory must be LeRobotForcevlaDataConfig, got {type(factory).__name__}")
    if factory.output_action_dim != ACTION_DIM:
        raise ValueError(f"config output_action_dim must be {ACTION_DIM}, got {factory.output_action_dim}")
    if tuple(factory.delta_action_mask or ()) != (True,) * 7 + (False,):
        raise ValueError(
            "config delta_action_mask must convert the seven joints to delta and keep the gripper absolute"
        )

    expected_repack = {
        "image": "observation.images.third_view",
        "wrist_image": "observation.images.wrist",
        "state": "observation.state",
        "force": "observation.wrench_compensated",
        "actions": "action",
        "prompt": "prompt",
    }
    repack_inputs = tuple(factory.repack_transforms.inputs)
    if len(repack_inputs) != 1 or getattr(repack_inputs[0], "structure", None) != expected_repack:
        raise ValueError("config repack transform does not match the trained ForceVLA 8 Hz fields")

    data_config = factory.create(train_config.assets_dirs, model)
    if data_config.use_quantile_norm:
        raise ValueError("ForceVLA 8 Hz checkpoint was trained with z-score normalization, not quantile normalization")
    if len(data_config.data_transforms.inputs) < 1 or not isinstance(
        data_config.data_transforms.inputs[0], forcevla_policy.Forcevla_inputs
    ):
        raise ValueError("config does not use the ForceVLA input processor")
    if len(data_config.data_transforms.outputs) < 1 or not isinstance(
        data_config.data_transforms.outputs[-1], forcevla_policy.Forcevla_outputs
    ):
        raise ValueError("config does not use the ForceVLA output processor")

    asset_id = data_config.asset_id
    if not asset_id:
        raise ValueError("config must define an asset_id for normalization statistics")
    stats_dir = checkpoint / "assets" / asset_id
    stats = normalize.load(stats_dir)
    if set(stats) != {"state", "actions"}:
        raise ValueError(f"checkpoint norm stats must contain exactly state/actions, got {sorted(stats)}")
    for key in ("state", "actions"):
        for field in ("mean", "std"):
            values = np.asarray(getattr(stats[key], field))
            if values.shape != (MODEL_ACTION_DIM,):
                raise ValueError(f"checkpoint {key}.{field} must have shape ({MODEL_ACTION_DIM},), got {values.shape}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"checkpoint {key}.{field} contains NaN or infinity")
    if np.any(np.asarray(stats["state"].std)[STATE_DIM : STATE_DIM + WRENCH_DIM] <= 0):
        raise ValueError("checkpoint has missing/constant force normalization statistics")
    if np.any(np.asarray(stats["actions"].std)[:ACTION_DIM] <= 0):
        raise ValueError("checkpoint has missing/constant action normalization statistics")

    _, shapes = _parse_checkpoint_metadata(checkpoint)
    if _find_shape(shapes, "'action_out_proj'", "'bias'") != (MODEL_ACTION_DIM,):
        raise ValueError("checkpoint action head does not output 32 model dimensions")
    force_kernel = _find_shape(shapes, "'force_in_proj'", "'kernel'")
    if force_kernel[0] != WRENCH_DIM:
        raise ValueError(f"checkpoint force head expects {force_kernel[0]} values, not {WRENCH_DIM}")


class RealBackend:
    """One-time-loaded JAX ForceVLA backend."""

    model_name = "ForceVLA"
    action_dim = ACTION_DIM

    def __init__(self, *, checkpoint: Path, config_name: str, device: str, dtype: str) -> None:
        _configure_jax_environment(device)

        # Import only after platform environment selection.  JAX has no PyTorch
        # train/eval mode or autocast context: inference uses train=False in the
        # model, JIT, and the requested parameter dtype as its mixed-precision
        # equivalent.
        import jax
        import jax.numpy as jnp
        from openpi.policies import policy_config
        from openpi.training import config as training_config

        dtype_aliases = {
            "bfloat16": (jnp.bfloat16, "bfloat16"),
            "bf16": (jnp.bfloat16, "bfloat16"),
            "float16": (jnp.float16, "float16"),
            "fp16": (jnp.float16, "float16"),
            "float32": (jnp.float32, "float32"),
            "fp32": (jnp.float32, "float32"),
        }
        dtype_key = dtype.strip().lower()
        if dtype_key not in dtype_aliases:
            raise ValueError(f"unsupported --dtype {dtype!r}; choose bfloat16, float16, or float32")

        self._jax = jax
        self._device_object = jax.devices()[0]
        if device.strip().lower() not in {"auto", "cpu"} and self._device_object.platform not in {"gpu", "cuda"}:
            raise RuntimeError(f"CUDA was requested but JAX selected {self._device_object}")

        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory not found: {checkpoint}")
        train_config = training_config.get_config(config_name)
        _validate_real_contract(train_config, checkpoint)
        self.parameter_count, _ = _parse_checkpoint_metadata(checkpoint)

        param_dtype, compute_dtype = dtype_aliases[dtype_key]
        if hasattr(train_config.model, "dtype"):
            # ForceVLA layers use config.dtype as JAX's autocast-equivalent
            # matrix-multiplication/activation dtype.
            train_config = dataclasses.replace(
                train_config,
                model=dataclasses.replace(train_config.model, dtype=compute_dtype),
            )

        with jax.default_device(self._device_object):
            self._policy = policy_config.create_trained_policy(
                train_config,
                checkpoint,
                param_dtype=param_dtype,
            )

        self.checkpoint = str(checkpoint)
        self.device = str(self._device_object)
        self.dtype = np.dtype(param_dtype).name

    def infer(self, policy_observation: dict[str, Any]) -> Any:
        with self._jax.default_device(self._device_object):
            return self._policy.infer(policy_observation)

    def reset(self) -> None:
        # A reset starts the diffusion sampling PRNG sequence over; the model has
        # no recurrent hidden state.
        self._policy.reset_rng()


class PolicyServer:
    """Robust REP loop: every successful receive is followed by exactly one send."""

    def __init__(self, backend: Backend, *, host: str, port: int) -> None:
        self.backend = backend
        self.host = host
        self.port = port
        self.bound_port: int | None = None
        self._request_sequence = 0

    def _next_request_id(self, request: Any) -> str:
        self._request_sequence += 1
        if isinstance(request, Mapping) and request.get("request_id") is not None:
            return str(request["request_id"])
        return str(self._request_sequence)

    def handle_request(self, request: Any, *, request_id: str | None = None) -> Any:
        request_id = request_id or self._next_request_id(request)
        if not isinstance(request, Mapping):
            raise TypeError("request must be a MessagePack dictionary")
        endpoint = request.get("endpoint")
        if endpoint == "ping":
            return {
                "status": "ok",
                "model_name": self.backend.model_name,
                "request_id": request_id,
            }
        if endpoint == "reset":
            self.backend.reset()
            return {"status": "ok", "reset": True, "request_id": request_id}
        if endpoint != "get_action":
            raise ValueError(f"unknown endpoint: {endpoint!r}")

        data = request.get("data")
        if not isinstance(data, Mapping):
            raise TypeError("get_action data must be a dictionary")
        if data.get("options") is not None:
            raise ValueError("get_action options must be null")
        policy_observation = validate_observation(data.get("observation"))

        started = time.perf_counter()
        result = self.backend.infer(policy_observation)
        inference_ms = (time.perf_counter() - started) * 1000.0
        actions_value = result.get("actions") if isinstance(result, Mapping) else result
        actions = np.asarray(actions_value)
        if actions.ndim != 2:
            raise ValueError(f"backend actions must have rank 2, got shape {actions.shape}")
        if actions.shape[0] < ACTION_HORIZON:
            raise ValueError(
                f"backend returned only {actions.shape[0]} action steps; at least {ACTION_HORIZON} are required"
            )
        if actions.shape[1] != self.backend.action_dim or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"backend action dimension must be {ACTION_DIM}, got {actions.shape[1]}")
        actions = np.ascontiguousarray(actions[:ACTION_HORIZON], dtype=np.float32)
        if not np.all(np.isfinite(actions)):
            raise ValueError("backend returned NaN or infinite actions")

        info = {
            "model_name": "ForceVLA",
            "request_id": request_id,
            "inference_ms": inference_ms,
            "action_shape": list(actions.shape),
            "action_representation": ACTION_REPRESENTATION,
            "action_names": ACTION_NAMES,
            "checkpoint": self.backend.checkpoint,
        }
        LOGGER.info(
            "request_id=%s inference_ms=%.2f action_shape=%s",
            request_id,
            inference_ms,
            tuple(actions.shape),
        )
        return [actions, info]

    def serve_forever(
        self,
        stop_event: threading.Event,
        *,
        ready_event: threading.Event | None = None,
    ) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        poller = zmq.Poller()
        try:
            if self.port == 0:
                self.bound_port = socket.bind_to_random_port(f"tcp://{self.host}")
            else:
                socket.bind(f"tcp://{self.host}:{self.port}")
                self.bound_port = self.port
            poller.register(socket, zmq.POLLIN)
            LOGGER.info("ForceVLA REP server listening on tcp://%s:%d", self.host, self.bound_port)
            if ready_event is not None:
                ready_event.set()

            while not stop_event.is_set():
                events = dict(poller.poll(timeout=250))
                if socket not in events:
                    continue
                # Allocate an ID immediately after receipt so malformed payloads
                # are still traceable without logging their contents.
                request_id = self._next_request_id(None)
                try:
                    request = unpack_message(socket.recv())
                    if isinstance(request, Mapping) and request.get("request_id") is not None:
                        request_id = str(request["request_id"])
                    response = self.handle_request(request, request_id=request_id)
                except Exception as exc:  # Keep REP state and process alive for all request errors.
                    LOGGER.exception("request_id=%s request failed", request_id)
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                try:
                    socket.send(pack_message(response))
                except Exception:
                    # A send failure means the REP state cannot safely continue.
                    LOGGER.exception("fatal response send failure; stopping server")
                    stop_event.set()
        finally:
            if ready_event is not None:
                ready_event.set()
            try:
                poller.unregister(socket)
            except KeyError:
                pass
            socket.close(linger=0)
            context.term()
            LOGGER.info("ForceVLA REP server closed")


def warmup_backend(backend: Backend, count: int) -> None:
    if count < 0:
        raise ValueError("--warmup must be non-negative")
    policy_observation = validate_observation(make_wire_observation())
    for index in range(count):
        started = time.perf_counter()
        result = backend.infer(policy_observation)
        actions = result.get("actions") if isinstance(result, Mapping) else result
        array = np.asarray(actions)
        if array.ndim != 2 or array.shape[0] < ACTION_HORIZON or array.shape[1] != ACTION_DIM:
            raise ValueError(f"warmup returned invalid action shape {array.shape}")
        LOGGER.info("warmup=%d/%d elapsed_ms=%.2f", index + 1, count, (time.perf_counter() - started) * 1000)
    backend.reset()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the trained ForceVLA 8 Hz policy over ZeroMQ REP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", default="bfloat16", help="bfloat16, float16, or float32")
    parser.add_argument("--warmup", type=int, default=1, help="number of startup inference calls")
    parser.add_argument("--dry-run", action="store_true", help="use a deterministic fake backend")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.host:
        raise ValueError("--host must not be empty")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    _validate_args(args)

    load_started = time.perf_counter()
    if args.dry_run:
        backend: Backend = FakeBackend()
    else:
        backend = RealBackend(
            checkpoint=args.checkpoint,
            config_name=args.config,
            device=args.device,
            dtype=args.dtype,
        )
    load_seconds = time.perf_counter() - load_started

    LOGGER.info(
        "loaded model=%s seconds=%.2f device=%s dtype=%s parameters=%d checkpoint=%s",
        backend.model_name,
        load_seconds,
        backend.device,
        backend.dtype,
        backend.parameter_count,
        backend.checkpoint,
    )
    LOGGER.info(
        "observation_contract=state float32(8), wrench_compensated float32(6), "
        "wrist/third_view uint8 RGB(480,640,3), non-empty task"
    )
    LOGGER.info("wrench_contract=%s", WRENCH_REPRESENTATION)
    LOGGER.info(
        "action_contract=float32(%d,%d), step 0 nearest observation, %s; no interpolation/control",
        ACTION_HORIZON,
        ACTION_DIM,
        ACTION_REPRESENTATION,
    )
    warmup_backend(backend, args.warmup)

    stop_event = threading.Event()

    def request_shutdown(signum: int, _frame: Any) -> None:
        LOGGER.info("received signal=%s; shutting down", signal.Signals(signum).name)
        stop_event.set()

    previous_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_shutdown)
    try:
        PolicyServer(backend, host=args.host, port=args.port).serve_forever(stop_event)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
