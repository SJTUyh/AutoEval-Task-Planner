"""规划结果 JSON 组装与控制台摘要。"""

from __future__ import annotations

from datetime import datetime

from .config import TaskSet
from .optimize import Candidate, OptimizationResult
from .resources import INFINITE, ResourceProvider, resource_caps
from .taskset import TaskModel
from .throughput import ThroughputProfile

SCHEMA_VERSION = 1
QUALITY_SCORE = {"history": 2, "estimate": 1, "heuristic": 0}


def build_report(
    *,
    taskset: TaskSet,
    throughput: ThroughputProfile,
    tasks: list[TaskModel],
    result: OptimizationResult,
    warnings: list[str],
    provider: ResourceProvider | None = None,
) -> dict:
    best = result.best
    case_concurrency = {
        task_id: value
        for task_id, value in best.plan.case_concurrency.items()
    }
    resource_assumption, resources = _resource_sections(
        taskset, tasks, best, provider, warnings
    )
    timeline = []
    for item in best.timeline:
        entry = dict(item)
        entry["duration_sec"] = round(
            float(item["end_sec"]) - float(item["start_sec"]), 2
        )
        timeline.append(entry)
    timeline_map = {item["task_id"]: item for item in timeline}

    task_entries = []
    for task in tasks:
        concurrency = case_concurrency.get(task.task_id, 1)
        standalone = best.standalone_durations.get(task.task_id, 0.0)
        task_entries.append(
            _task_entry(
                task,
                concurrency,
                timeline_map.get(task.task_id),
                standalone,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "target_service": {
            "base_url": taskset.target.base_url,
            "model": taskset.target.model,
        },
        "throughput_source": {
            "type": throughput.source,
            "measured_at": throughput.measured_at,
            "model": throughput.model,
            "effort": throughput.effort,
            "decode_tps": round(throughput.peak_decode_tps, 3),
            "prefill_tps": (
                round(throughput.peak_prefill_tps, 3)
                if throughput.peak_prefill_tps
                else None
            ),
            "max_concurrency": throughput.max_concurrency,
            "curve": [
                {
                    "concurrency": level.concurrency,
                    "decode_tps": round(level.decode_tps, 3),
                    "ttft_sec": round(level.ttft_sec, 4) if level.ttft_sec is not None else None,
                    "error_rate": round(level.error_rate, 4),
                }
                for level in throughput.levels
            ],
        },
        "resource_assumption": resource_assumption,
        "resources": resources,
        "tasks": task_entries,
        "estimate": {
            "total_duration_sec": round(best.p50_sec, 1),
            "range": {
                "p10_sec": round(best.p10_sec, 1),
                "p50_sec": round(best.p50_sec, 1),
                "p90_sec": round(best.p90_sec, 1),
            },
            "utilization": round(best.utilization, 4),
        },
        "plan": {
            "dataset_order": list(best.plan.order),
            "dataset_concurrency": best.plan.dataset_concurrency,
            "case_concurrency": case_concurrency,
            "timeline": timeline,
            "suggested_adaptor_overrides": _suggested_overrides(tasks, case_concurrency),
            "execution_groups": _execution_groups(tasks, best),
        },
        "alternatives": [_candidate_entry(candidate) for candidate in result.alternatives],
        "simulation": {
            "runs": result.simulation_runs,
            "evaluated_candidates": result.evaluated_candidates,
            "simulated_candidates": result.simulated_candidates,
            "throughput_cv": taskset.simulation.throughput_cv,
            "seed": taskset.simulation.seed,
        },
        "warnings": list(warnings),
        "confidence": _confidence(tasks, throughput),
    }


def _task_entry(
    task: TaskModel,
    case_concurrency: int | None,
    timeline: dict | None,
    standalone_sec: float,
) -> dict:
    input_total = sum(case.input_tokens for case in task.cases)
    output_total = sum(case.output_tokens for case in task.cases)
    scheduled = timeline or {}
    return {
        "task_id": task.task_id,
        "group_id": task.group_id,
        "config_path": task.config_path,
        "task_type": task.task_type,
        "dataset_key": task.dataset_key,
        "dataset_alias": task.dataset_alias,
        "case_count": task.case_count,
        "input_tokens_total": input_total,
        "output_tokens_total": output_total,
        "data_quality": task.quality,
        "source_runs": list(task.source_runs),
        "declared_case_concurrency": task.declared_case_concurrency,
        "case_concurrency": case_concurrency,
        "standalone_duration_sec": round(standalone_sec, 1),
        "scheduled_start_sec": scheduled.get("start_sec"),
        "scheduled_end_sec": scheduled.get("end_sec"),
        "scheduled_duration_sec": scheduled.get("duration_sec"),
    }


def _candidate_entry(candidate: Candidate) -> dict:
    return {
        "dataset_order": list(candidate.plan.order),
        "dataset_concurrency": candidate.plan.dataset_concurrency,
        "case_concurrency": dict(candidate.plan.case_concurrency),
        "p10_sec": round(candidate.p10_sec, 1),
        "p50_sec": round(candidate.p50_sec, 1),
        "p90_sec": round(candidate.p90_sec, 1),
        "utilization": round(candidate.utilization, 4),
        "resource_peaks": candidate.resource_peaks,
        "binding_resource": candidate.binding_resource,
        "resource_limited": candidate.resource_limited,
    }


def _resource_sections(
    taskset: TaskSet,
    tasks: list[TaskModel],
    best: Candidate,
    provider: ResourceProvider | None,
    warnings: list[str],
) -> tuple[dict, dict]:
    if provider is None or taskset.resources.mode == "infinite":
        return (
            {"mode": "infinite"},
            {
                "binding_resource": None,
                "peak": {},
                "resource_limited": False,
                "max_feasible_case_concurrency": {},
                "per_task": {},
            },
        )

    capacity = provider.capacity()
    assumption = {
        "mode": "server",
        "server": capacity.name,
        "captured_at": capacity.captured_at,
        "source": capacity.source,
        "headroom": capacity.headroom,
        "cpu_cores": round(capacity.cpu_cores, 2),
        "mem_mb": round(capacity.mem_mb, 1),
        "disk_gb": round(capacity.disk_gb, 1) if capacity.disk_gb < INFINITE else None,
        "networks": capacity.networks if capacity.networks < INFINITE else None,
        "network_pool": capacity.network_pool,
    }

    caps = resource_caps(tasks, provider, max_case_concurrency=1_000_000)
    per_task: dict[str, dict] = {}
    for task in tasks:
        case = provider.case_resources(task.task_id)
        per_task[task.task_id] = {
            "cpu_base": case.cpu_base,
            "mem_mb_base": case.mem_mb_base,
            "cpu_per_case": case.cpu,
            "mem_mb_per_case": case.mem_mb,
            "disk_gb_per_case": case.disk_gb,
            "networks_per_case": case.network,
            "source": case.source,
            "max_feasible_case_concurrency": caps[task.task_id],
        }

    resources = {
        "binding_resource": best.binding_resource,
        "peak": best.resource_peaks,
        "resource_limited": best.resource_limited,
        "max_feasible_case_concurrency": caps,
        "per_task": per_task,
    }

    if best.resource_limited and best.binding_resource:
        warnings.append(
            f"资源受限：{best.binding_resource} 达到可用容量上限"
            f"（已预留 {capacity.headroom:.0%} 裕量），case 并发被压制"
        )
    if capacity.captured_at:
        try:
            captured = datetime.fromisoformat(capacity.captured_at)
            age_days = (
                datetime.now().astimezone() - captured
            ).total_seconds() / 86400
            if age_days > 7:
                warnings.append(
                    f"服务器资源快照已 {age_days:.1f} 天未更新，建议重新执行 resource probe"
                )
        except (TypeError, ValueError):
            pass
    return assumption, resources


def _suggested_overrides(tasks: list[TaskModel], case_concurrency: dict[str, int]) -> dict:
    overrides: dict[str, dict] = {}
    for task in tasks:
        value = case_concurrency.get(task.task_id)
        if value is None:
            continue
        if task.task_type.endswith("harbor_agent"):
            overrides[task.task_id] = {"n_concurrent_trials": value}
        else:
            overrides[task.task_id] = {"batch_size": value}
    return overrides


def _execution_groups(tasks: list[TaskModel], best: Candidate) -> list[dict]:
    groups: dict[str, list[str]] = {}
    for task in tasks:
        key = task.config_path or f"<inline:{task.group_id}>"
        groups.setdefault(key, []).append(task.task_id)

    intervals = {
        item["task_id"]: (float(item["start_sec"]), float(item["end_sec"]))
        for item in best.timeline
    }
    result: list[dict] = []
    for key, task_ids in groups.items():
        windows = [intervals[task_id] for task_id in task_ids if task_id in intervals]
        result.append(
            {
                "config_path": None if key.startswith("<inline:") else key,
                "tasks": task_ids,
                "max_num_workers": _max_overlap(windows),
            }
        )
    return result


def _max_overlap(windows: list[tuple[float, float]]) -> int:
    if not windows:
        return 1
    events: list[tuple[float, int]] = []
    for start, end in windows:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    maximum = 0
    for _, delta in events:
        current += delta
        maximum = max(maximum, current)
    return max(maximum, 1)


def _confidence(tasks: list[TaskModel], throughput: ThroughputProfile) -> str:
    score = min((QUALITY_SCORE.get(task.quality, 0) for task in tasks), default=0)
    if throughput.source != "probe":
        score = max(0, score - 1)
    return {2: "high", 1: "medium", 0: "low"}[score]


def format_summary(report: dict) -> str:
    estimate = report["estimate"]
    plan = report["plan"]
    lines = [
        "规划结果",
        f"  目标服务: {report['target_service']['base_url']} ({report['target_service']['model']})",
        f"  吞吐来源: {report['throughput_source']['type']} "
        f"(decode {report['throughput_source']['decode_tps']} tok/s @ "
        f"c={report['throughput_source']['max_concurrency']})",
        f"  最优总时长: {_duration(estimate['total_duration_sec'])} "
        f"(p10 {_duration(estimate['range']['p10_sec'])}, "
        f"p90 {_duration(estimate['range']['p90_sec'])})",
        f"  数据集顺序: {' -> '.join(plan['dataset_order'])}",
        f"  数据集并发: {plan['dataset_concurrency']}",
        "  case 并发: "
        + ", ".join(f"{key}={value}" for key, value in plan["case_concurrency"].items()),
        f"  服务利用率: {estimate['utilization']:.1%}",
        f"  置信度: {report['confidence']}",
        "  各数据集时长:",
    ]
    for task in report["tasks"]:
        scheduled = (
            f"，计划内 {_duration(task['scheduled_duration_sec'])}"
            f"（第 {_duration(task['scheduled_start_sec'])} 开始）"
            if task.get("scheduled_duration_sec") is not None
            else ""
        )
        lines.append(
            f"    - {task['task_id']}: 单独 {_duration(task['standalone_duration_sec'])}{scheduled}"
        )
    if report["warnings"]:
        lines.append("  告警:")
        lines.extend(f"    - {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def _duration(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{rest:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h{minutes:02d}m{int(rest):02d}s"
