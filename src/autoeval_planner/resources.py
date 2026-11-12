"""服务器资源模型。

资源来源分三层：

1. `servers`：目标服务器的 CPU / 内存 / 磁盘 / 可用 Docker 网段快照；
2. `tool_resource_defaults`：按 (tool, subcommand) 的默认资源画像；
3. `dataset_resources`：按数据集的资源画像（优先于工具默认）。

`mode=infinite` 时行为与旧版完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol

from .config import ResourceSpec
from .errors import ConfigError
from .library.db import Library, ResourceProfile, ServerSnapshot

INFINITE = 1_000_000.0
INFINITE_INT = 1_000_000

TOOL_DEFAULTS: dict[tuple[str, str], ResourceProfile] = {
    ("ais_bench", "llm_accuracy"): ResourceProfile(
        cpu_base=1.0,
        mem_mb_base=1024,
        cpu_per_case=0.02,
        mem_mb_per_case=16,
        networks_per_case=0,
        source="default",
    ),
    ("ais_bench", "harbor_agent"): ResourceProfile(
        cpu_base=0.5,
        mem_mb_base=256,
        cpu_per_case=2.0,
        mem_mb_per_case=8192,
        networks_per_case=1,
        source="default",
    ),
}


@dataclass(frozen=True)
class ResourceCapacity:
    cpu_cores: float = INFINITE
    mem_mb: float = INFINITE
    disk_gb: float = INFINITE
    networks: int = INFINITE_INT
    max_parallel_tasks: int = INFINITE_INT
    max_case_concurrency: int = INFINITE_INT
    headroom: float = 0.0
    name: str = "infinite"
    captured_at: str | None = None
    source: str | None = None
    network_pool: str | None = None


@dataclass(frozen=True)
class CaseResources:
    cpu: float = 0.0
    mem_mb: float = 0.0
    disk_gb: float = 0.0
    network: int = 0
    cpu_base: float = 0.0
    mem_mb_base: float = 0.0
    source: str = "default"


class TaskRef(Protocol):
    task_id: str
    dataset_key: str
    task_type: str


class ResourceProvider(Protocol):
    def capacity(self) -> ResourceCapacity: ...

    def case_resources(self, task_id: str) -> CaseResources: ...


class InfiniteResourceProvider:
    def capacity(self) -> ResourceCapacity:
        return ResourceCapacity()

    def case_resources(self, task_id: str) -> CaseResources:
        return CaseResources()


class ServerResourceProvider:
    def __init__(
        self,
        capacity: ResourceCapacity,
        profiles: Mapping[str, CaseResources],
    ) -> None:
        self._capacity = capacity
        self._profiles = dict(profiles)

    def capacity(self) -> ResourceCapacity:
        return self._capacity

    def case_resources(self, task_id: str) -> CaseResources:
        return self._profiles.get(task_id, CaseResources())


def build_provider(
    spec: ResourceSpec,
    library: Library,
    tasks: Iterable[TaskRef],
) -> ResourceProvider:
    if spec.mode == "infinite":
        return InfiniteResourceProvider()
    if spec.mode != "server":
        raise ConfigError(f"暂不支持 resources.mode={spec.mode}")

    name = spec.server or ""
    snapshot = library.get_server(name)
    if snapshot is None:
        raise ConfigError(
            f"未找到服务器资源快照: {name}；请先执行 "
            f"`autoeval-planner resource probe --name {name} ...` 或 `resource set`"
        )

    capacity = _capacity_from_snapshot(snapshot, spec.headroom)
    profiles: dict[str, CaseResources] = {}
    for task in tasks:
        tool, subcommand = _split_task_type(task.task_type)
        default = library.get_tool_resources(tool, subcommand)
        if default is None:
            default = TOOL_DEFAULTS.get((tool, subcommand)) or ResourceProfile(
                cpu_base=0.0,
                mem_mb_base=0,
                cpu_per_case=0.0,
                mem_mb_per_case=0,
                networks_per_case=0,
                source="default",
            )
            library.upsert_tool_resources(tool, subcommand, default)

        profile = library.get_dataset_resources(task.dataset_key, task.task_type)
        if profile is None:
            profile = ResourceProfile(
                cpu_base=default.cpu_base,
                mem_mb_base=default.mem_mb_base,
                cpu_per_case=default.cpu_per_case,
                mem_mb_per_case=default.mem_mb_per_case,
                networks_per_case=default.networks_per_case,
                source="default",
            )
            library.upsert_dataset_resources(task.dataset_key, task.task_type, profile)

        profiles[task.task_id] = CaseResources(
            cpu=profile.cpu_per_case,
            mem_mb=profile.mem_mb_per_case,
            disk_gb=profile.disk_gb_per_case,
            network=profile.networks_per_case,
            cpu_base=profile.cpu_base,
            mem_mb_base=profile.mem_mb_base,
            source=profile.source,
        )
    return ServerResourceProvider(capacity, profiles)


def _capacity_from_snapshot(snapshot: ServerSnapshot, headroom: float) -> ResourceCapacity:
    factor = 1.0 - headroom
    return ResourceCapacity(
        cpu_cores=snapshot.cpu_cores * factor,
        mem_mb=snapshot.mem_mb * factor,
        disk_gb=(
            snapshot.disk_gb * factor if snapshot.disk_gb is not None else INFINITE
        ),
        networks=(
            snapshot.network_available if snapshot.network_available is not None else INFINITE_INT
        ),
        headroom=headroom,
        name=snapshot.name,
        captured_at=snapshot.captured_at,
        source=snapshot.source,
        network_pool=snapshot.network_pool,
    )


def _split_task_type(task_type: str) -> tuple[str, str]:
    if "/" in task_type:
        tool, subcommand = task_type.split("/", 1)
        return tool, subcommand
    return task_type, ""


def resource_caps(
    tasks: Iterable[TaskRef],
    provider: ResourceProvider,
    *,
    max_case_concurrency: int = INFINITE_INT,
) -> dict[str, int]:
    """每个任务受资源限制的 case 并发上限。"""
    capacity = provider.capacity()
    task_list = list(tasks)
    # 所有任务的基础资源会同时占用，先从容量里扣除，避免 case 并发贴边。
    base_cpu = sum(provider.case_resources(task.task_id).cpu_base for task in task_list)
    base_mem = sum(provider.case_resources(task.task_id).mem_mb_base for task in task_list)
    cpu_budget = max(0.0, capacity.cpu_cores - base_cpu)
    mem_budget = max(0.0, capacity.mem_mb - base_mem)

    caps: dict[str, int] = {}
    for task in task_list:
        case = provider.case_resources(task.task_id)
        limits: list[int] = [max_case_concurrency]
        if case.cpu > 0:
            limits.append(int(cpu_budget // case.cpu))
        if case.mem_mb > 0:
            limits.append(int(mem_budget // case.mem_mb))
        if case.disk_gb > 0:
            limits.append(int(capacity.disk_gb // case.disk_gb))
        if case.network > 0:
            limits.append(int(capacity.networks // case.network))
        caps[task.task_id] = max(1, min(limits))
    return caps


def dataset_cap(
    tasks: Iterable[TaskRef],
    provider: ResourceProvider,
) -> int:
    """数据集级并发上限（基础资源 × 资源容量）。"""
    capacity = provider.capacity()
    limit = capacity.max_parallel_tasks
    for task in tasks:
        case = provider.case_resources(task.task_id)
        if case.cpu_base > 0:
            limit = min(limit, int(capacity.cpu_cores // case.cpu_base))
        if case.mem_mb_base > 0:
            limit = min(limit, int(capacity.mem_mb // case.mem_mb_base))
    return max(1, limit)
