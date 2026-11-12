"""taskset 输入文件的结构校验。

与 AutoEval-Benchmark-Adaptor 保持一致的严格风格：未声明的字段直接报错，
避免拼写错误被静默忽略。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

TOP_LEVEL_KEYS = ("target", "throughput", "resources", "tasks", "simulation")
TARGET_KEYS = ("base_url", "model")
THROUGHPUT_KEYS = ("mode", "path", "inline")
RESOURCE_KEYS = ("mode", "server", "headroom")
TASK_KEYS = ("task_id", "config", "config_object", "estimate", "case_count")
ESTIMATE_KEYS = (
    "case_count",
    "input_tokens_median",
    "input_tokens_p90",
    "output_tokens_median",
    "output_tokens_p90",
    "overhead_sec",
)
SIMULATION_KEYS = ("runs", "seed", "throughput_cv")


@dataclass(frozen=True)
class Target:
    base_url: str
    model: str


@dataclass(frozen=True)
class ThroughputSpec:
    mode: str
    path: str | None = None
    inline: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResourceSpec:
    mode: str = "infinite"
    server: str | None = None
    headroom: float = 0.20


@dataclass(frozen=True)
class Estimate:
    case_count: int
    input_tokens_median: float
    input_tokens_p90: float
    output_tokens_median: float
    output_tokens_p90: float
    overhead_sec: float = 0.0


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    config_path: str | None = None
    config_object: dict[str, Any] | None = None
    estimate: Estimate | None = None
    case_count: int | None = None


@dataclass(frozen=True)
class SimulationSpec:
    runs: int = 100
    seed: int = 20261111
    throughput_cv: float = 0.10


@dataclass(frozen=True)
class TaskSet:
    path: Path
    base_dir: Path
    target: Target
    throughput: ThroughputSpec
    resources: ResourceSpec
    tasks: tuple[TaskSpec, ...]
    simulation: SimulationSpec


def load_taskset(path: str | Path) -> TaskSet:
    taskset_path = Path(path)
    if not taskset_path.is_file():
        raise ConfigError(f"taskset 文件不存在: {taskset_path}")
    try:
        raw = json.loads(taskset_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"taskset 文件 {taskset_path} 不是合法 JSON: {exc}") from exc
    return parse_taskset(raw, base_dir=taskset_path.parent, source=str(taskset_path), path=taskset_path)


def parse_taskset(
    raw: Any,
    *,
    base_dir: Path | None = None,
    source: str = "<json>",
    path: Path | None = None,
) -> TaskSet:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} 顶层必须是 JSON 对象，实际为 {type(raw).__name__}")
    _reject_unknown(raw, TOP_LEVEL_KEYS, source)
    for key in ("target", "throughput", "tasks"):
        if key not in raw:
            raise ConfigError(f"{source} 缺少顶层字段: {key}")

    target = _parse_target(raw["target"], f"{source}.target")
    throughput = _parse_throughput(raw["throughput"], f"{source}.throughput")
    resources = _parse_resources(raw.get("resources"), f"{source}.resources")
    tasks = _parse_tasks(raw["tasks"], f"{source}.tasks")
    simulation = _parse_simulation(raw.get("simulation"), f"{source}.simulation")

    resolved_base = base_dir if base_dir is not None else Path.cwd()
    return TaskSet(
        path=path if path is not None else resolved_base / "<inline>",
        base_dir=resolved_base,
        target=target,
        throughput=throughput,
        resources=resources,
        tasks=tasks,
        simulation=simulation,
    )


def _parse_target(raw: Any, where: str) -> Target:
    obj = _expect_object(raw, where)
    _reject_unknown(obj, TARGET_KEYS, where)
    base_url = _require_str(obj, "base_url", where)
    model = _require_str(obj, "model", where)
    return Target(base_url=base_url, model=model)


def _parse_throughput(raw: Any, where: str) -> ThroughputSpec:
    obj = _expect_object(raw, where)
    _reject_unknown(obj, THROUGHPUT_KEYS, where)
    mode = _require_str(obj, "mode", where, choices=("file", "inline"))
    if mode == "file":
        path = _require_str(obj, "path", where)
        if "inline" in obj:
            raise ConfigError(f"{where} mode=file 时不允许再提供 inline")
        return ThroughputSpec(mode=mode, path=path)
    inline = obj.get("inline")
    if not isinstance(inline, dict):
        raise ConfigError(f"{where}.inline 在 mode=inline 时必须是 JSON 对象")
    if "path" in obj:
        raise ConfigError(f"{where} mode=inline 时不允许再提供 path")
    return ThroughputSpec(mode=mode, inline=dict(inline))


def _parse_resources(raw: Any, where: str) -> ResourceSpec:
    if raw is None:
        return ResourceSpec()
    obj = _expect_object(raw, where)
    _reject_unknown(obj, RESOURCE_KEYS, where)
    mode = _require_str(obj, "mode", where, choices=("infinite", "server"))
    server = obj.get("server")
    if mode == "server":
        if not isinstance(server, str) or not server.strip():
            raise ConfigError(f"{where}.server 在 mode=server 时必填")
        server = server.strip()
    elif server is not None:
        raise ConfigError(f"{where}.server 只能与 mode=server 一起使用")
    headroom = _optional_number(obj, "headroom", where, default=0.20)
    if not 0 <= headroom <= 0.9:
        raise ConfigError(f"{where}.headroom 必须在 0 到 0.9 之间")
    return ResourceSpec(mode=mode, server=server, headroom=headroom)


def _parse_tasks(raw: Any, where: str) -> tuple[TaskSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where} 必须是非空数组")
    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        item_where = f"{where}[{index}]"
        obj = _expect_object(item, item_where)
        _reject_unknown(obj, TASK_KEYS, item_where)
        task_id = _require_str(obj, "task_id", item_where)
        if task_id in seen:
            raise ConfigError(f"{item_where}.task_id 重复: {task_id}")
        seen.add(task_id)

        has_path = "config" in obj
        has_object = "config_object" in obj
        if has_path == has_object:
            raise ConfigError(f"{item_where} 必须且只能提供 config 或 config_object 之一")

        config_path = _require_str(obj, "config", item_where) if has_path else None
        config_object = None
        if has_object:
            if not isinstance(obj["config_object"], dict):
                raise ConfigError(f"{item_where}.config_object 必须是 JSON 对象")
            config_object = dict(obj["config_object"])

        estimate = _parse_estimate(obj.get("estimate"), f"{item_where}.estimate")
        case_count = _optional_int(obj, "case_count", item_where, default=0) or None
        if case_count is not None and case_count <= 0:
            raise ConfigError(f"{item_where}.case_count 必须大于 0")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                config_path=config_path,
                config_object=config_object,
                estimate=estimate,
                case_count=case_count,
            )
        )
    return tuple(tasks)


def _parse_estimate(raw: Any, where: str) -> Estimate | None:
    if raw is None:
        return None
    obj = _expect_object(raw, where)
    _reject_unknown(obj, ESTIMATE_KEYS, where)
    case_count = _require_int(obj, "case_count", where)
    if case_count <= 0:
        raise ConfigError(f"{where}.case_count 必须大于 0")
    values = {
        "input_tokens_median": _require_number(obj, "input_tokens_median", where),
        "input_tokens_p90": _require_number(obj, "input_tokens_p90", where),
        "output_tokens_median": _require_number(obj, "output_tokens_median", where),
        "output_tokens_p90": _require_number(obj, "output_tokens_p90", where),
    }
    for prefix in ("input_tokens", "output_tokens"):
        median = values[f"{prefix}_median"]
        p90 = values[f"{prefix}_p90"]
        if median <= 0:
            raise ConfigError(f"{where}.{prefix}_median 必须大于 0")
        if p90 < median:
            raise ConfigError(f"{where}.{prefix}_p90 不能小于 {prefix}_median")
    overhead = _optional_number(obj, "overhead_sec", where, default=0.0)
    if overhead < 0:
        raise ConfigError(f"{where}.overhead_sec 不能为负数")
    return Estimate(case_count=case_count, overhead_sec=overhead, **values)


def _parse_simulation(raw: Any, where: str) -> SimulationSpec:
    if raw is None:
        return SimulationSpec()
    obj = _expect_object(raw, where)
    _reject_unknown(obj, SIMULATION_KEYS, where)
    runs = _optional_int(obj, "runs", where, default=100)
    if runs < 1:
        raise ConfigError(f"{where}.runs 必须大于等于 1")
    if runs > 5000:
        raise ConfigError(f"{where}.runs 过大（>5000），请缩小仿真次数")
    seed = _optional_int(obj, "seed", where, default=20261111)
    cv = _optional_number(obj, "throughput_cv", where, default=0.10)
    if cv < 0:
        raise ConfigError(f"{where}.throughput_cv 不能为负数")
    return SimulationSpec(runs=runs, seed=seed, throughput_cv=cv)


def _expect_object(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} 必须是 JSON 对象，实际为 {type(raw).__name__}")
    return raw


def _reject_unknown(obj: dict[str, Any], allowed: tuple[str, ...], where: str) -> None:
    extra = [key for key in obj if key not in allowed]
    if extra:
        raise ConfigError(
            f"{where} 存在未知字段: {', '.join(map(str, extra))}；允许的字段: {', '.join(allowed)}"
        )


def _require_str(
    obj: dict[str, Any],
    key: str,
    where: str,
    *,
    choices: tuple[str, ...] | None = None,
) -> str:
    if key not in obj:
        raise ConfigError(f"{where} 缺少必填字段: {key}")
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}.{key} 必须是非空字符串")
    if choices is not None and value not in choices:
        raise ConfigError(f"{where}.{key} 取值必须是 {', '.join(choices)} 之一，实际为 {value!r}")
    return value.strip()


def _require_int(obj: dict[str, Any], key: str, where: str) -> int:
    if key not in obj:
        raise ConfigError(f"{where} 缺少必填字段: {key}")
    value = obj[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是整数")
    return value


def _optional_int(obj: dict[str, Any], key: str, where: str, *, default: int) -> int:
    if key not in obj or obj[key] is None:
        return default
    value = obj[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是整数")
    return value


def _require_number(obj: dict[str, Any], key: str, where: str) -> float:
    if key not in obj:
        raise ConfigError(f"{where} 缺少必填字段: {key}")
    value = obj[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是数字")
    return float(value)


def _optional_number(obj: dict[str, Any], key: str, where: str, *, default: float) -> float:
    if key not in obj or obj[key] is None:
        return default
    value = obj[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是数字")
    return float(value)
