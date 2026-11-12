"""把 Adaptor 任务配置转换成规划用的任务模型。"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Estimate, TaskSet, TaskSpec
from .errors import ConfigError, InsufficientDataError
from .library.db import Library
from .library.profile import (
    QUALITY_HEURISTIC,
    CaseSpec,
    DatasetProfile,
    heuristic_profile,
    profile_from_estimate,
    profile_from_rows,
    reference_profile,
)

TEST_RANGE = re.compile(r"^\[\s*(\d+)\s*:\s*(\d+)\s*\]$")
CONFIG_KEYS = ("tool", "subcommand", "params")
SUPPORTED = {
    ("ais_bench", "llm_accuracy"): "ais_bench/llm_accuracy",
    ("ais_bench", "harbor_agent"): "ais_bench/harbor_agent",
}


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    group_id: str
    config_path: str | None
    task_type: str
    dataset_alias: str
    dataset_key: str | None
    case_limit: int | None
    attempts: int
    declared_case_concurrency: int | None
    estimate: Estimate | None
    display_name: str
    case_count_override: int | None = None


@dataclass
class TaskModel:
    task_id: str
    group_id: str
    config_path: str | None
    task_type: str
    dataset_key: str
    dataset_alias: str
    cases: list[CaseSpec]
    pool: list[CaseSpec]
    quality: str
    declared_case_concurrency: int | None
    source_runs: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        return len(self.cases)


def plan_tasks(taskset: TaskSet, library: Library) -> list[PlannedTask]:
    planned: list[PlannedTask] = []
    for spec in taskset.tasks:
        config = _load_task_config(spec, taskset.base_dir)
        tool = config["tool"]
        subcommand = config["subcommand"]
        params = config["params"]
        task_type = SUPPORTED.get((tool, subcommand))
        if task_type is None:
            raise ConfigError(
                f"task {spec.task_id}: 暂不支持 tool={tool!r} subcommand={subcommand!r}，"
                f"当前支持: {', '.join(sorted(f'{t}/{s}' for t, s in SUPPORTED))}"
            )

        if subcommand == "llm_accuracy":
            planned.extend(_plan_accuracy(spec, task_type, params, library))
        else:
            planned.append(_plan_harbor(spec, task_type, params, library))
    return planned


def build_task_models(
    planned: list[PlannedTask],
    library: Library,
    *,
    seed: int,
    strict: bool,
) -> tuple[list[TaskModel], list[str]]:
    warnings: list[str] = []
    history: dict[str, DatasetProfile] = {}
    for task in planned:
        key = task.dataset_key
        if not key or key in history:
            continue
        rows = library.load_cases(key)
        if not rows:
            continue
        history[key] = profile_from_rows(
            key, task.task_type, rows, library.dataset_runs(key)
        )

    models: list[TaskModel] = []
    for task in planned:
        key = task.dataset_key or task.dataset_alias
        profile = history.get(task.dataset_key or "")
        if profile is None:
            if task.estimate is not None:
                profile = profile_from_estimate(key, task.task_type, task.estimate, seed)
                warnings.append(f"{task.task_id}: 无历史数据，使用输入估算（case_count={task.estimate.case_count}）")
            else:
                profile = reference_profile(
                    key, task.task_type, list(history.values()), seed
                )
                if profile is None:
                    profile = heuristic_profile(key, task.task_type, seed)
                    warnings.append(
                        f"{task.task_id}: 无历史数据且未提供估算，使用启发式默认分布（结果置信度低）"
                    )
                else:
                    warnings.append(
                        f"{task.task_id}: 无历史数据，按同类任务参考分布估算（结果置信度低）"
                    )
        if strict and profile.quality == QUALITY_HEURISTIC:
            raise InsufficientDataError(
                f"task {task.task_id}: 缺少历史数据（--strict 模式要求全部数据集都有历史记录或显式估算）"
            )

        cases = list(profile.cases)
        resampled = False
        if task.case_count_override and task.case_count_override != len(cases):
            # 历史 run 的任务集合可能和当前数据集不同（例如含重试或多版本），
            # 用历史分布重采样到目标 case 数。
            rng = random.Random(f"{seed}:{task.task_id}")
            cases = profile.sample(task.case_count_override, rng)
            resampled = True
            warnings.append(
                f"{task.task_id}: 按 case_count={task.case_count_override} 重采样历史分布"
                f"（历史唯一 case 数={profile.case_count}）"
            )
        elif task.case_limit and task.case_limit < len(cases):
            cases = cases[: task.case_limit]
        if not cases:
            raise ConfigError(f"task {task.task_id}: 有效 case 数为 0，无法规划")

        pool = list(profile.cases) if resampled else list(cases)
        if task.attempts > 1:
            cases = cases * task.attempts
            pool = pool * task.attempts

        models.append(
            TaskModel(
                task_id=task.task_id,
                group_id=task.group_id,
                config_path=task.config_path,
                task_type=task.task_type,
                dataset_key=key,
                dataset_alias=task.dataset_alias,
                cases=cases,
                pool=pool,
                quality=profile.quality,
                declared_case_concurrency=task.declared_case_concurrency,
                source_runs=list(profile.source_runs),
            )
        )
    return models, warnings


# -- 配置加载 -------------------------------------------------------------


def _load_task_config(spec: TaskSpec, base_dir: Path) -> dict[str, Any]:
    if spec.config_object is not None:
        raw = spec.config_object
        source = f"task {spec.task_id} 的 config_object"
    else:
        path = Path(str(spec.config_path)).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        if not path.is_file():
            raise ConfigError(f"task {spec.task_id}: 配置文件不存在: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"task {spec.task_id}: {path} 不是合法 JSON: {exc}") from exc
        source = str(path)

    if not isinstance(raw, dict):
        raise ConfigError(f"{source} 顶层必须是 JSON 对象")
    extra = [key for key in raw if key not in CONFIG_KEYS]
    if extra:
        raise ConfigError(f"{source} 顶层只允许 tool、subcommand、params，发现多余字段: {', '.join(extra)}")
    missing = [key for key in CONFIG_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"{source} 缺少顶层字段: {', '.join(missing)}")
    if not isinstance(raw["params"], dict):
        raise ConfigError(f"{source} 的 params 必须是 JSON 对象")
    if not isinstance(raw["tool"], str) or not isinstance(raw["subcommand"], str):
        raise ConfigError(f"{source} 的 tool / subcommand 必须是字符串")
    return raw


# -- accuracy --------------------------------------------------------------


def _plan_accuracy(
    spec: TaskSpec,
    task_type: str,
    params: dict[str, Any],
    library: Library,
) -> list[PlannedTask]:
    datasets = params.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(
        isinstance(item, str) and item for item in datasets
    ):
        raise ConfigError(f"task {spec.task_id}: params.datasets 必须是非空字符串数组")

    case_limit = _case_limit(params, spec.task_id)
    declared = _positive_int(params.get("batch_size"), f"task {spec.task_id} 的 params.batch_size")

    tasks: list[PlannedTask] = []
    multiple = len(datasets) > 1
    for alias in datasets:
        tasks.append(
            PlannedTask(
                task_id=f"{spec.task_id}:{alias}" if multiple else spec.task_id,
                group_id=spec.task_id,
                config_path=spec.config_path,
                task_type=task_type,
                dataset_alias=alias,
                dataset_key=library.lookup_alias(alias, task_type),
                case_limit=case_limit,
                attempts=1,
                declared_case_concurrency=declared,
                estimate=spec.estimate,
                display_name=alias,
                case_count_override=spec.case_count,
            )
        )
    return tasks


def _case_limit(params: dict[str, Any], task_id: str) -> int | None:
    num_prompts = _positive_int(params.get("num_prompts"), f"task {task_id} 的 params.num_prompts")
    if num_prompts is not None:
        return num_prompts

    test_range = params.get("dataset_test_range")
    if test_range is None:
        return None
    if not isinstance(test_range, str):
        raise ConfigError(f"task {task_id}: params.dataset_test_range 必须是字符串")
    match = TEST_RANGE.match(test_range.strip())
    if match is None:
        raise ConfigError(
            f"task {task_id}: 无法解析 dataset_test_range={test_range!r}，期望形如 \"[0:100]\""
        )
    start, end = int(match.group(1)), int(match.group(2))
    if end <= start:
        raise ConfigError(f"task {task_id}: dataset_test_range 的结束位置必须大于起始位置")
    return end - start


# -- harbor ----------------------------------------------------------------


def _plan_harbor(
    spec: TaskSpec,
    task_type: str,
    params: dict[str, Any],
    library: Library,
) -> PlannedTask:
    alias = params.get("dataset_abbr")
    if not isinstance(alias, str) or not alias.strip():
        path = params.get("path")
        if isinstance(path, str) and path.strip():
            alias = Path(path.rstrip("/\\")).name
        else:
            raise ConfigError(
                f"task {spec.task_id}: harbor_agent 需要 params.dataset_abbr 或 params.path"
            )
    alias = alias.strip()

    attempts = _positive_int(params.get("n_attempts"), f"task {spec.task_id} 的 params.n_attempts") or 1
    case_limit = _positive_int(params.get("n_tasks"), f"task {spec.task_id} 的 params.n_tasks")
    declared = _positive_int(
        params.get("n_concurrent_trials"),
        f"task {spec.task_id} 的 params.n_concurrent_trials",
    )

    return PlannedTask(
        task_id=spec.task_id,
        group_id=spec.task_id,
        config_path=spec.config_path,
        task_type=task_type,
        dataset_alias=alias,
        dataset_key=library.lookup_alias(alias, task_type),
        case_limit=case_limit,
        attempts=attempts,
        declared_case_concurrency=declared,
        estimate=spec.estimate,
        display_name=alias,
        case_count_override=spec.case_count,
    )


def _positive_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{where} 必须是整数")
    if value <= 0:
        raise ConfigError(f"{where} 必须大于 0")
    return value
