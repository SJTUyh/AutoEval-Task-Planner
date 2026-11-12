"""case 粒度的离散事件仿真。

服务模型：全局在途 n 个请求时，聚合 decode 吞吐取吞吐曲线的 rate_at(n)，
每个请求均分该吞吐；首 token 延迟取 ttft_at(n) + 输入 token 的 prefill 时间。

模型假设：
- 数据集内部按 FIFO 取 case（与 AISBench 行为一致）；
- `overhead_sec` 表示非 LLM 固定开销（如 agent 容器/工具执行），占用该任务的 case
  并发名额但不消耗服务吞吐；
- 服务器资源无限（由 ResourceProvider 提供 case 总并发上限）。
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .errors import ConfigError
from .library.profile import CaseSpec
from .resources import CaseResources, ResourceProvider
from .throughput import ThroughputProfile

DEFAULT_PREFILL_TPS = 8000.0
EPS = 1e-9

_sequence = itertools.count()


@dataclass(frozen=True)
class Plan:
    order: list[str]
    dataset_concurrency: int
    case_concurrency: dict[str, int]

    def key(self) -> tuple:
        return (
            tuple(self.order),
            self.dataset_concurrency,
            tuple(sorted(self.case_concurrency.items())),
        )

    def as_dict(self) -> dict:
        return {
            "dataset_order": list(self.order),
            "dataset_concurrency": self.dataset_concurrency,
            "case_concurrency": dict(self.case_concurrency),
        }


@dataclass(frozen=True)
class TaskTimeline:
    task_id: str
    start_sec: float
    end_sec: float


@dataclass
class SimResult:
    makespan_sec: float
    timeline: list[TaskTimeline]
    utilization: float
    tokens_done: float
    resource_peaks: dict[str, float] = field(default_factory=dict)
    binding_resource: str | None = None
    resource_limited: bool = False

    def timeline_dict(self) -> list[dict]:
        return [
            {
                "task_id": item.task_id,
                "start_sec": round(item.start_sec, 2),
                "end_sec": round(item.end_sec, 2),
            }
            for item in self.timeline
        ]


class _Request:
    __slots__ = ("task_id", "remaining", "overhead")

    def __init__(self, task_id: str, output_tokens: int, overhead: float) -> None:
        self.task_id = task_id
        self.remaining = float(output_tokens)
        self.overhead = overhead


def simulate(
    tasks: Mapping[str, Sequence[CaseSpec]],
    plan: Plan,
    throughput: ThroughputProfile,
    provider: ResourceProvider,
    *,
    ignore_input_prefill: set[str] | None = None,
) -> SimResult:
    """服务吞吐按当前在途请求数取吞吐曲线并公平分享；服务器资源由 ResourceProvider
    限制 case 总并发（本期无限）。

    ignore_input_prefill 中的任务（harbor_agent）不把输入 token 计入 TTFT：
    agent 的输入是跨多轮累计值且大量命中缓存，无法当作一次 prefill。
    """
    skip_input = ignore_input_prefill or set()
    order = [task_id for task_id in plan.order if tasks.get(task_id)]
    if not order:
        raise ConfigError("计划中没有可执行的 case")

    capacity = provider.capacity()
    case_cap = max(1, capacity.max_case_concurrency)
    dataset_concurrency = max(1, min(plan.dataset_concurrency, len(order)))

    resources = {task_id: _case_resources(provider, task_id) for task_id in order}
    cap_cpu = capacity.cpu_cores
    cap_mem = capacity.mem_mb
    cap_disk = capacity.disk_gb
    cap_net = capacity.networks
    used_cpu = 0.0
    used_mem = 0.0
    used_disk = 0.0
    used_net = 0
    peaks = {"cpu": 0.0, "mem_mb": 0.0, "disk_gb": 0.0, "networks": 0.0}
    base_held: set[str] = set()
    resource_limited = False

    for task_id in order:
        need = resources[task_id]
        if (
            need.cpu > cap_cpu
            or need.mem_mb > cap_mem
            or need.disk_gb > cap_disk
            or need.network > cap_net
        ):
            raise ConfigError(
                f"资源不足：任务 {task_id} 单个 case 需要 "
                f"{need.cpu} 核 / {need.mem_mb} MB / {need.disk_gb} GB / {need.network} 网段，"
                "超过服务器可用容量"
            )

    def update_peaks() -> None:
        peaks["cpu"] = max(peaks["cpu"], used_cpu)
        peaks["mem_mb"] = max(peaks["mem_mb"], used_mem)
        peaks["disk_gb"] = max(peaks["disk_gb"], used_disk)
        peaks["networks"] = max(peaks["networks"], float(used_net))

    state = {
        task_id: {
            "cases": list(tasks[task_id]),
            "next": 0,
            "active": 0,
            "start": 0.0,
        }
        for task_id in order
    }
    concurrency = {
        task_id: max(1, int(plan.case_concurrency.get(task_id, 1))) for task_id in order
    }

    active_tasks: list[str] = []
    timeline: list[TaskTimeline] = []
    ttft_heap: list[tuple[float, int, _Request]] = []
    decode_heap: list[tuple[float, int, _Request]] = []
    overhead_heap: list[tuple[float, int, _Request]] = []
    active_total = 0

    order_index = 0
    now = 0.0
    ticks = 0.0
    tokens_done = 0.0
    prefill_tps = throughput.prefill_tps() or DEFAULT_PREFILL_TPS

    def finish(request: _Request) -> None:
        nonlocal active_total, used_cpu, used_mem, used_disk, used_net
        task_state = state[request.task_id]
        task_state["active"] -= 1
        active_total -= 1
        need = resources[request.task_id]
        used_cpu -= need.cpu
        used_mem -= need.mem_mb
        used_disk -= need.disk_gb
        used_net -= need.network

    def close_tasks() -> None:
        nonlocal used_cpu, used_mem
        for task_id in _close_finished_tasks(active_tasks, state, timeline, now):
            if task_id in base_held:
                base = resources[task_id]
                used_cpu -= base.cpu_base
                used_mem -= base.mem_mb_base
                base_held.discard(task_id)

    guard = 0
    while order_index < len(order) or active_total > 0 or active_tasks:
        guard += 1
        if guard > 5_000_000:
            raise ConfigError("仿真步数异常，请检查吞吐曲线或并发设置")

        while len(active_tasks) < dataset_concurrency and order_index < len(order):
            task_id = order[order_index]
            order_index += 1
            if not state[task_id]["cases"]:
                timeline.append(TaskTimeline(task_id, now, now))
                continue
            base = resources[task_id]
            if used_cpu + base.cpu_base > cap_cpu or used_mem + base.mem_mb_base > cap_mem:
                raise ConfigError(
                    f"资源不足：无法为任务 {task_id} 分配基础资源 "
                    f"({base.cpu_base} 核 / {base.mem_mb_base} MB)"
                )
            used_cpu += base.cpu_base
            used_mem += base.mem_mb_base
            base_held.add(task_id)
            state[task_id]["start"] = now
            active_tasks.append(task_id)
            update_peaks()

        service_active = len(ttft_heap) + len(decode_heap)
        for task_id in list(active_tasks):
            task_state = state[task_id]
            need = resources[task_id]
            while (
                task_state["active"] < concurrency[task_id]
                and task_state["next"] < len(task_state["cases"])
                and active_total < case_cap
            ):
                if (
                    used_cpu + need.cpu > cap_cpu
                    or used_mem + need.mem_mb > cap_mem
                    or used_disk + need.disk_gb > cap_disk
                    or used_net + need.network > cap_net
                ):
                    resource_limited = True
                    break
                used_cpu += need.cpu
                used_mem += need.mem_mb
                used_disk += need.disk_gb
                used_net += need.network
                update_peaks()
                case = task_state["cases"][task_state["next"]]
                task_state["next"] += 1
                task_state["active"] += 1
                active_total += 1
                request = _Request(task_id, case.output_tokens, case.overhead_sec)
                input_tokens = 0 if task_id in skip_input else case.input_tokens
                ttft = _ttft(throughput, service_active + 1, input_tokens, prefill_tps)
                if ttft > EPS:
                    heapq.heappush(ttft_heap, (now + ttft, next(_sequence), request))
                    service_active += 1
                elif request.remaining > 0:
                    heapq.heappush(
                        decode_heap,
                        (ticks + request.remaining, next(_sequence), request),
                    )
                    service_active += 1
                else:
                    heapq.heappush(
                        overhead_heap,
                        (now + max(request.overhead, EPS), next(_sequence), request),
                    )

        if active_total == 0:
            close_tasks()
            if not active_tasks and order_index >= len(order):
                break
            continue

        # 吞吐曲线的 decode_tps 是纯 decode 吞吐，因此只按真正在解码的请求数取曲线；
        # TTFT 阶段不摊薄其它请求的解码速率（prefill/思考开销已体现在 ttft 上）。
        n = max(1, len(decode_heap))
        per_request_rate = throughput.rate_at(n) / n
        if per_request_rate <= 0:
            raise ConfigError("吞吐曲线在并发 %d 下为 0，无法仿真" % n)

        delta = math.inf
        if ttft_heap:
            delta = min(delta, ttft_heap[0][0] - now)
        if overhead_heap:
            delta = min(delta, overhead_heap[0][0] - now)
        if decode_heap:
            delta = min(delta, (decode_heap[0][0] - ticks) / per_request_rate)
        if delta <= EPS:
            delta = 0.0
        elif delta is math.inf:
            raise ConfigError("仿真无法推进：请检查 case token 与吞吐设置")

        decoding = len(decode_heap)
        if delta > 0:
            now += delta
            ticks += per_request_rate * delta
            tokens_done += per_request_rate * delta * decoding

        while ttft_heap and ttft_heap[0][0] <= now + EPS:
            _, _, request = heapq.heappop(ttft_heap)
            if request.remaining > 0:
                heapq.heappush(
                    decode_heap, (ticks + request.remaining, next(_sequence), request)
                )
            elif request.overhead > EPS:
                heapq.heappush(
                    overhead_heap, (now + request.overhead, next(_sequence), request)
                )
            else:
                finish(request)

        while decode_heap and decode_heap[0][0] <= ticks + EPS:
            _, _, request = heapq.heappop(decode_heap)
            if request.overhead > EPS:
                heapq.heappush(
                    overhead_heap, (now + request.overhead, next(_sequence), request)
                )
            else:
                finish(request)

        while overhead_heap and overhead_heap[0][0] <= now + EPS:
            _, _, request = heapq.heappop(overhead_heap)
            finish(request)

        close_tasks()

    makespan = max((item.end_sec for item in timeline), default=0.0)
    utilization = 0.0
    if makespan > 0 and throughput.peak_decode_tps > 0:
        utilization = min(1.0, tokens_done / (throughput.peak_decode_tps * makespan))

    binding: str | None = None
    if capacity.name != "infinite":
        ratios = {
            "cpu": peaks["cpu"] / cap_cpu if cap_cpu > 0 else 0.0,
            "mem_mb": peaks["mem_mb"] / cap_mem if cap_mem > 0 else 0.0,
            "disk_gb": peaks["disk_gb"] / cap_disk if cap_disk > 0 else 0.0,
            "networks": peaks["networks"] / cap_net if cap_net > 0 else 0.0,
        }
        binding = max(ratios, key=ratios.get)

    return SimResult(
        makespan_sec=makespan,
        timeline=sorted(timeline, key=lambda item: (item.start_sec, item.task_id)),
        utilization=utilization,
        tokens_done=tokens_done,
        resource_peaks={key: round(value, 3) for key, value in peaks.items()},
        binding_resource=binding,
        resource_limited=resource_limited,
    )


def _case_resources(provider: ResourceProvider, task_id: str) -> CaseResources:
    getter = getattr(provider, "case_resources", None)
    return getter(task_id) if callable(getter) else CaseResources()


def _close_finished_tasks(
    active_tasks: list[str],
    state: dict,
    timeline: list[TaskTimeline],
    now: float,
) -> list[str]:
    removed: list[str] = []
    for task_id in list(active_tasks):
        task_state = state[task_id]
        if task_state["next"] >= len(task_state["cases"]) and task_state["active"] == 0:
            timeline.append(TaskTimeline(task_id, task_state["start"], now))
            active_tasks.remove(task_id)
            removed.append(task_id)
    return removed


def _ttft(
    throughput: ThroughputProfile,
    concurrency: int,
    input_tokens: int,
    prefill_tps: float,
) -> float:
    """首 token 延迟 = 实测基线 + 输入 token 的 prefill 时间。"""
    prefill_sec = input_tokens / prefill_tps if prefill_tps > 0 else 0.0
    measured = throughput.ttft_at(concurrency)
    if measured is not None:
        return measured + prefill_sec
    return prefill_sec


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1 - ratio) + ordered[upper] * ratio
