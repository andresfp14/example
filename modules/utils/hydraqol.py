"""Portable Hydra run lifecycle and resolvers (contract version 2).

Modes: base skips completed runs; check only reports; clean removes incomplete
runs and stops; force removes and reruns. Tasks return their own results unchanged.
Multi-rank calls require an initialized process group before entering the decorator.
"""

import hashlib
import importlib.metadata
import inspect
import itertools
import json
import logging
import multiprocessing
import operator
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from functools import reduce, wraps
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


def run_decorator(func):
    """Manage cfg.save_dir; keep task returns separate from lifecycle metadata."""

    @wraps(func)
    def wrapper(cfg, *args, **kwargs):
        # 1. Select the interpreter before touching this run's files.
        script = Path(inspect.getfile(func)).resolve()
        switched, result = venv_force_check(cfg, script)
        if switched:
            return result
        root = Path(get_original_cwd() if HydraConfig.initialized() else Path.cwd()).resolve()
        group = sys.modules.get("torch.distributed")
        if group is None or not group.is_available() or not group.is_initialized():
            group = None
            if rank_info()[1] > 1:
                raise RuntimeError("Initialize the process group before entering run_decorator")
        rank = group.get_rank() if group is not None else rank_info()[0]
        mode = cfg.get("mode", "base")
        if mode not in ("base", "check", "clean", "force"):
            raise ValueError(f"Unknown run mode: {mode}")
        retries = max(0, int(cfg.get("max_retries", cfg.get("retry", {}).get("max_retries", 0))))
        delay = cfg.get("retry_delay", cfg.get("retry", {}).get("delay", 5))
        if group is not None and retries:
            raise ValueError("Use the distributed launcher's restart policy, not task retries")

        # 2. Let rank zero prepare the run and share its decision with every worker.
        logger = logging.getLogger("hydraqol")
        message = [None]
        if group is None or rank == 0:
            try:
                folder = (root / cfg.save_dir).resolve()
                cfg.save_dir = str(folder)
                info = read_run_info(folder)
                completed = info["state"] == "completed"
                run = mode in ("base", "force") and (mode == "force" or not completed)
                record = info if mode == "check" else None
                if mode == "check":
                    logger.info("%s: %s", folder, info["state"])

                # 3. Only delete marked run folders, never the checkout or its parents.
                if mode == "force" or (mode == "clean" and not completed):
                    if folder.exists():
                        if root.is_relative_to(folder) or not (folder / "config.yaml").is_file():
                            raise ValueError(f"Not a removable run directory: {folder}")
                        shutil.rmtree(folder)

                # 4. Save the resolved configuration, environment and provenance together.
                if run:
                    folder.mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(cfg, folder / "config.yaml", resolve=True)
                    if (root / "uv.lock").exists():
                        shutil.copyfile(root / "uv.lock", folder / "uv.lock")
                    provenance = {}
                    for key, command in (
                        ("git_commit", ["rev-parse", "HEAD"]),
                        ("git_status", ["status", "--porcelain"]),
                    ):
                        try:
                            provenance[key] = subprocess.check_output(
                                ["git", *command], cwd=root, stderr=subprocess.DEVNULL, text=True
                            ).strip()
                        except (OSError, subprocess.CalledProcessError):
                            provenance[key] = None
                    arguments = [sys.executable, str(script), *sys.argv[1:]]
                    rerun = [
                        sys.executable,
                        str(script),
                        f"--config-path={folder.as_posix()}",
                        "--config-name=config",
                        "mode=force",
                    ]
                    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
                    record = {
                        "schema_version": 2,
                        "state": "running",
                        "mode": mode,
                        "executed_file": script.name,
                        "save_dir": str(folder),
                        "start_time": datetime.now(UTC).isoformat(),
                        "command": quote(arguments),
                        "argv": arguments,
                        "relative_command": quote(rerun),
                        "config_hash": config_hash(cfg),
                        "python_version": sys.version,
                        "hostname": platform.node(),
                        **provenance,
                        "packages": {
                            d.metadata["Name"]: d.version
                            for d in importlib.metadata.distributions()
                        },
                        "retry_count": 0,
                        "error_files": [],
                    }
                    write_json(folder / "run_info.json", record)
                message[0] = {"folder": str(folder), "run": run, "record": record}
            except Exception:
                if group is None:
                    raise
                message[0] = {"error": traceback.format_exc()}
        if group is not None:
            group.broadcast_object_list(message, src=0)
        if "error" in message[0]:
            raise RuntimeError(message[0]["error"])
        folder, run, record = Path(message[0]["folder"]), message[0]["run"], message[0]["record"]
        cfg.save_dir = str(folder)
        if rank == 0 and not cfg.get("wrapper_quiet", False):
            logger.info("%s %s: %s", mode, "run" if run else "skip", folder)
        if not run:
            if path := os.environ.get("HYDRA_VENV_RESULT"):
                write_json(Path(path), record)
            return record
        if rank == 0 and cfg.get("printcfg", False):
            logger.info("%s", OmegaConf.to_yaml(cfg, resolve=True))
        started = time.perf_counter()

        # 5. Run with optional single-process retries and one error file per rank/attempt.
        for attempt in range(retries + 1):
            error, result = None, None
            error_path = folder / f"error_p{rank}_attempt{attempt + 1}.txt"
            try:
                result = func(cfg, *args, **kwargs)
            except Exception as caught:
                error = caught
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
            failures = [str(error_path) if error is not None else None]
            if group is not None:
                failures = [None] * group.get_world_size()
                group.all_gather_object(failures, str(error_path) if error is not None else None)
            failed = any(failures)

            # 6. Publish rank-zero results only after all participating ranks return.
            record.update(
                state="failed" if failed else "completed",
                retry_count=attempt,
                end_time=datetime.now(UTC).isoformat(),
                total_time_seconds=time.perf_counter() - started,
                result=result,
            )
            record["error_files"].extend(path for path in failures if path is not None)
            message = [None]
            if group is None or rank == 0:
                try:
                    write_json(folder / "run_info.json", record)
                except Exception:
                    if group is None:
                        raise
                    message[0] = traceback.format_exc()
            if group is not None:
                group.broadcast_object_list(message, src=0)
            if message[0] is not None:
                raise RuntimeError(message[0])
            if not failed:
                if path := os.environ.get("HYDRA_VENV_RESULT"):
                    write_json(Path(path), result)
                return result
            if attempt < retries:
                logger.warning("Retrying %s in %s seconds", script.name, delay)
                time.sleep(delay)
            elif error is not None:
                raise error
            else:
                raise RuntimeError(f"Another rank failed; see {record['error_files']}")

    return wrapper


def write_json(path: Path, value) -> None:
    # 1. Replace complete records so interrupted writes do not look like finished runs.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_run_info(folder):
    """Read the common record, accepting older example and TSFoundation records."""
    folder = Path(folder)
    for name in ("run_info.json", "run.json"):
        path = folder / name
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            record.setdefault(
                "state", "completed" if record.get("status") == "success" else "incomplete"
            )
            return record
    return {"state": "incomplete" if folder.exists() else "not run"}


def rank_info(env=None):
    """Read global rank/world; torchrun takes precedence inside a Slurm allocation."""
    env = os.environ if env is None else env
    if "RANK" in env:
        return int(env["RANK"]), int(env.get("WORLD_SIZE", 1))
    if "SLURM_PROCID" in env:
        return int(env["SLURM_PROCID"]), int(env.get("SLURM_NTASKS", 1))
    return 0, int(env.get("WORLD_SIZE", 1))


def rank_zero(env=None):
    group = sys.modules.get("torch.distributed") if env is None else None
    if group is not None and group.is_available() and group.is_initialized():
        return group.get_rank() == 0
    return rank_info(env)[0] == 0


def venv_force_check(cfg, script=None):
    """Return (switched, result); legacy cfg.venv may name a parent containing .venv."""
    # 1. uv normally selects the environment; switching is an opt-in compatibility path.
    if not cfg.get("venv_force", False):
        return False, None
    root = Path(get_original_cwd() if HydraConfig.initialized() else Path.cwd()).resolve()
    environment = (root / cfg.get("venv", ".")).resolve()
    if not (environment / "pyvenv.cfg").is_file():
        environment = environment / ".venv"
    if Path(sys.prefix).resolve() == environment:
        return False, None
    target = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not target.is_file():
        raise FileNotFoundError(f"venv_force: interpreter not found: {target}")
    if os.environ.get("HYDRA_VENV_FORCED") == str(environment):
        raise RuntimeError("venv_force: child did not enter the requested environment")
    group = sys.modules.get("torch.distributed")
    if group is not None and group.is_initialized():
        raise RuntimeError("Select the environment before initializing distributed communication")

    # 2. Pass the current resolved job, never the original multirun command line.
    script = Path(script or sys.argv[0]).resolve()
    staging = root / "data" / ".hydraqol"
    staging.mkdir(parents=True, exist_ok=True)
    child_env = dict(os.environ, HYDRA_VENV_FORCED=str(environment))
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), child_env.get("PYTHONPATH", "")]
    )
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        OmegaConf.save(cfg, Path(temporary) / "config.yaml", resolve=True)
        result_path = Path(temporary) / "result.json"
        child_env["HYDRA_VENV_RESULT"] = str(result_path)
        command = [
            str(target),
            str(script),
            f"--config-path={Path(temporary).as_posix()}",
            "--config-name=config",
            "hydra.mode=RUN",
        ]
        logs = root / "data" / "outputs" / ".venv_logs" / Path(temporary).name
        command.append(f"hydra.run.dir={logs.as_posix()}")
        subprocess.run(command, cwd=root, env=child_env, check=True)
        # 3. Transfer this invocation's result, including None when a job was skipped.
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return True, result


# Configuration resolvers


def default(val, default=1):
    """Use a fallback for null values: ${default:${value},1}."""
    return default if val is None else val


def math(operation: str, *values):
    """Arithmetic on floats; power, modulo and floor division take two operands."""
    values = list(map(float, values))
    if not values or (operation in ("**", "%", "//") and len(values) != 2):
        raise ValueError("math needs operands; **, % and // require exactly two")
    operations = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
        "min": min,
        "max": max,
    }
    return reduce(operations[operation], values)


def config_hash(*values):
    """Stable short identity for explicit configuration fields, excluding runtime state."""
    text = json.dumps(
        [
            OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            for value in values
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def concat(*lists):
    """Join plain lists and OmegaConf lists without nesting them."""
    lists = [
        OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for value in lists
    ]
    return [item for values in lists for item in values]


def cpu_count(fraction=1.0):
    """Use available CPU affinity and the Slurm per-task allocation when present."""
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    )
    available = min(available, int(os.environ.get("SLURM_CPUS_PER_TASK", available)))
    return max(1, int(available * float(fraction)))


def get_mp_start_method():
    """Expose the platform/Python default; choose spawn explicitly for CUDA workers."""
    return multiprocessing.get_context().get_start_method()


def _flatten(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, (list, tuple)):
        return [float(value)], None
    flat, shape = [], []
    for item in value:
        values, child_shape = _flatten(item)
        flat.extend(values)
        shape.append((len(values), child_shape))
    return flat, shape


def _unflatten(values, shape, offset=0):
    if shape is None:
        return values[offset]
    output = []
    for size, child_shape in shape:
        output.append(_unflatten(values, child_shape, offset))
        offset += size
    return output


def grid_range(mins, maxs, n_steps):
    """Cartesian grid retaining the nested shape; equal bounds yield one value."""
    # 1. Construct each scalar range, retaining the structure for reconstruction.
    low, shape = _flatten(mins)
    high, high_shape = _flatten(maxs)
    steps = int(n_steps)
    if shape != high_shape or steps < 1:
        raise ValueError("grid_range requires matching shapes and positive n_steps")
    ranges = [
        [a] if a == b or steps == 1 else [a + (b - a) * i / (steps - 1) for i in range(steps)]
        for a, b in zip(low, high)
    ]
    # 2. Expand combinations and reconstruct each nested value.
    return [_unflatten(values, shape) for values in itertools.product(*ranges)]


def n_patches_overlap(length, patch, stride):
    """Count patches after rounding the context down to a patch multiple."""
    length, patch, stride = int(length), int(patch), int(stride)
    return ((length // patch * patch - patch) // stride) + 1


def n_patches_padded(length, patch, stride):
    """Count patches after rounding the context up to a patch multiple."""
    length, patch, stride = int(length), int(patch), int(stride)
    return (((-(-length // patch)) * patch - patch) // stride) + 1


def register_resolvers():
    """Register once; importing another copy does not overwrite existing resolvers."""
    resolvers = {
        "default": default,
        "default_if_missing": default,
        "math": math,
        "config_hash": config_hash,
        "concat": concat,
        "grid_range": grid_range,
        "ceildiv": lambda a, b: -(-int(a) // int(b)),
        "percentagestring": lambda value, width=3: f"{round(float(value) * 100):0{int(width)}d}",
        "cpu_count": cpu_count,
        "mp_start_method": get_mp_start_method,
        "device_to_backend": lambda device: "gpu" if str(device).split(":")[0] == "cuda" else "cpu",
        "n_patches_overlap": n_patches_overlap,
        "n_patches_padded": n_patches_padded,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


register_resolvers()
