"""计划枚举、粗筛与蒙特卡洛寻优。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import permutations, product

from .errors import ConfigError
from .library.profile import DatasetProfile
from .resources import ResourceProvider, dataset_cap, resource_caps
from .simulate import Plan, percentile, simulate
from .taskset import TaskModel
from .throughput import ThroughputProfile

MAX_CANDIDATES = 600
TOP_K = 5
MAX_CASE_CONCURRENCY = 256


@dataclass
class Candidate:
    plan: Plan
    p10_sec: float
    p50_sec: float
    p90_sec: float
    deterministic_sec: float
    utilization: float
    timeline: list[dict]
    resource_peaks: dict = field(default_factory=dict)
    binding_resource: str | None = None
    resource_limited: bool = False
    standalone_durations: dict = field(default_factory=dict)


@dataclass
class OptimizationResult:
    best: Candidate
    alternatives: list[Candidate]
    evaluated_candidates: int
    simulated_candidates: int
    simulation_runs: int


def optimize(
    tasks: list[TaskModel],
    throughput: ThroughputProfile,
    provider: ResourceProvider,
    *,
    runs: int,
    seed: int,
    throughput_cv: float,
) -> OptimizationResult:
    if not tasks:
        raise ConfigError("没有可规划的任务")

    capacity = provider.capacity()
    max_concurrency = max(
        1, min(capacity.max_case_concurrency, throughput.max_concurrency)
    )
    # accuracy：每个 case 就是一次在线请求，case 并发不能超过服务并发上限；
    # harbor：case 并发是 trial 并发，允许高于服务并发（大量时间花在工具执行上）。
    direct_task_ids = {
        task.task_id for task in tasks if not task.task_type.endswith("harbor_agent")
    }
    agent_task_ids = {
        task.task_id for task in tasks if task.task_type.endswith("harbor_agent")
    }

    resource_case_caps = resource_caps(
        tasks, provider, max_case_concurrency=MAX_CASE_CONCURRENCY
    )
    option_lists: list[list[int]] = []
    for task in tasks:
        cap = min(resource_case_caps[task.task_id], task.case_count)
        if task.declared_case_concurrency:
            cap = min(cap, task.declared_case_concurrency)
        cap = max(1, cap)
        if task.task_id in direct_task_ids:
            # 并发越高对该任务越有利，直接取服务饱和值与资源上限的较小者。
            options = [min(max_concurrency, cap)]
        else:
            options = sorted(
                {
                    value
                    for value in {cap, max(1, cap // 2), 64, 128, 256}
                    if 1 <= value <= cap
                }
            )
        option_lists.append(options or [1])

    orders = _enumerate_orders(tasks)
    max_dataset_concurrency = min(
        len(tasks), capacity.max_parallel_tasks, dataset_cap(tasks, provider)
    )
    dataset_options = list(range(1, max(1, max_dataset_concurrency) + 1))

    candidates: list[Plan] = []
    seen: set[tuple] = set()
    for order in orders:
        for dataset_concurrency in dataset_options:
            for combo in product(*option_lists):
                case_map = {}
                for task, value in zip(tasks, combo):
                    # 在途 case 数不可能超过实际 case 数。
                    value = min(value, task.case_count)
                    if task.declared_case_concurrency:
                        value = min(value, task.declared_case_concurrency)
                    case_map[task.task_id] = max(1, value)
                in_flight = sum(
                    sorted(case_map.values(), reverse=True)[:dataset_concurrency]
                )
                if in_flight > capacity.max_case_concurrency:
                    continue
                plan = Plan(
                    order=list(order),
                    dataset_concurrency=dataset_concurrency,
                    case_concurrency=case_map,
                )
                key = plan.key()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(plan)

    if not candidates:
        raise ConfigError(
            "没有可行的并发组合：请检查配置中的并发上限或吞吐曲线的 max_concurrency"
        )

    candidates = _thin(candidates, MAX_CANDIDATES)
    cases = {task.task_id: task.cases for task in tasks}

    ranked: list[tuple[float, Plan, object]] = []
    for plan in candidates:
        result = simulate(
            cases, plan, throughput, provider, ignore_input_prefill=agent_task_ids
        )
        ranked.append((result.makespan_sec, plan, result))
    ranked.sort(key=lambda item: item[0])

    shortlist = ranked[:TOP_K]
    evaluated: list[Candidate] = []
    for index, (deterministic_sec, plan, deterministic_result) in enumerate(shortlist):
        results = _monte_carlo(
            tasks,
            plan,
            throughput,
            provider,
            runs=runs,
            seed=seed + index * 7919,
            throughput_cv=throughput_cv,
            ignore_input_prefill=agent_task_ids,
        )
        values = [item[0].makespan_sec for item in results]
        p50 = percentile(values, 0.50)
        # 用最接近 p50 的那次仿真作为代表性时间线，保证 per-task 时长与总时长自洽。
        representative, sampled, scaled = min(
            results, key=lambda item: abs(item[0].makespan_sec - p50)
        )
        standalone = {
            task.task_id: simulate(
                {task.task_id: sampled[task.task_id]},
                Plan(
                    [task.task_id],
                    1,
                    {task.task_id: plan.case_concurrency.get(task.task_id, 1)},
                ),
                scaled,
                provider,
                ignore_input_prefill=agent_task_ids,
            ).makespan_sec
            for task in tasks
        }
        evaluated.append(
            Candidate(
                plan=plan,
                p10_sec=percentile(values, 0.10),
                p50_sec=p50,
                p90_sec=percentile(values, 0.90),
                deterministic_sec=deterministic_sec,
                utilization=representative.utilization,
                timeline=representative.timeline_dict(),
                resource_peaks=representative.resource_peaks,
                binding_resource=representative.binding_resource,
                resource_limited=representative.resource_limited,
                standalone_durations=standalone,
            )
        )

    evaluated.sort(key=_candidate_sort_key)
    best = evaluated[0]
    alternatives = evaluated[1:3]
    return OptimizationResult(
        best=best,
        alternatives=alternatives,
        evaluated_candidates=len(candidates),
        simulated_candidates=len(evaluated),
        simulation_runs=runs,
    )


def _candidate_sort_key(candidate: Candidate) -> tuple:
    return (
        candidate.p50_sec,
        candidate.plan.dataset_concurrency,
        -sum(candidate.plan.case_concurrency.values()),
        tuple(candidate.plan.order),
    )


def _monte_carlo(
    tasks: list[TaskModel],
    plan: Plan,
    throughput: ThroughputProfile,
    provider: ResourceProvider,
    *,
    runs: int,
    seed: int,
    throughput_cv: float,
    ignore_input_prefill: set[str],
) -> list:
    rng = random.Random(seed)
    profiles = {
        task.task_id: DatasetProfile(
            dataset_key=task.dataset_key,
            task_type=task.task_type,
            cases=task.pool,
        )
        for task in tasks
    }
    overhead_cv = {
        task.task_id: (0.15 if any(case.overhead_sec > 0 for case in task.pool) else 0.0)
        for task in tasks
    }
    results: list = []
    for _ in range(runs):
        factor = math.exp(rng.gauss(0.0, throughput_cv)) if throughput_cv > 0 else 1.0
        scaled = throughput.scaled(factor)
        sampled = {
            task.task_id: profiles[task.task_id].sample(
                len(task.cases), rng, overhead_cv=overhead_cv[task.task_id]
            )
            for task in tasks
        }
        results.append(
            (
                simulate(
                    sampled,
                    plan,
                    scaled,
                    provider,
                    ignore_input_prefill=ignore_input_prefill,
                ),
                sampled,
                scaled,
            )
        )
    return results


def _enumerate_orders(tasks: list[TaskModel]) -> list[list[str]]:
    ids = [task.task_id for task in tasks]
    if len(ids) <= 5:
        return [list(order) for order in permutations(ids)]

    def workload(task: TaskModel) -> int:
        return sum(case.output_tokens for case in task.cases)

    heaviest = sorted(tasks, key=workload, reverse=True)
    orders = [[task.task_id for task in heaviest]]
    orders.append([task.task_id for task in reversed(heaviest)])
    for order in permutations(ids):
        orders.append(list(order))
        if len(orders) >= 20:
            break
    return orders


def _thin(candidates: list[Plan], limit: int) -> list[Plan]:
    if len(candidates) <= limit:
        return candidates
    step = len(candidates) / limit
    return [candidates[int(index * step)] for index in range(limit)]
