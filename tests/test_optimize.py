from autoeval_planner.library.profile import CaseSpec
from autoeval_planner.optimize import optimize
from autoeval_planner.resources import (
    CaseResources,
    InfiniteResourceProvider,
    ResourceCapacity,
)
from autoeval_planner.taskset import TaskModel
from autoeval_planner.throughput import Level, ThroughputProfile


def _profile() -> ThroughputProfile:
    return ThroughputProfile(
        source="manual",
        levels=[Level(concurrency=1, decode_tps=100.0), Level(concurrency=8, decode_tps=800.0)],
        peak_prefill_tps=8000.0,
    )


def _task(task_id: str, count: int, output_tokens: int, declared: int | None = None) -> TaskModel:
    cases = [CaseSpec(input_tokens=0, output_tokens=output_tokens) for _ in range(count)]
    return TaskModel(
        task_id=task_id,
        group_id=task_id,
        config_path=None,
        task_type="ais_bench/llm_accuracy",
        dataset_key=task_id,
        dataset_alias=task_id,
        cases=cases,
        pool=cases,
        quality="history",
        declared_case_concurrency=declared,
    )


def test_optimize_produces_valid_plan():
    tasks = [_task("big", 40, 1000), _task("small", 5, 500)]
    result = optimize(
        tasks,
        _profile(),
        InfiniteResourceProvider(),
        runs=10,
        seed=7,
        throughput_cv=0.05,
    )

    assert sorted(result.best.plan.order) == ["big", "small"]
    assert 1 <= result.best.plan.dataset_concurrency <= 2
    assert all(value >= 1 for value in result.best.plan.case_concurrency.values())
    assert result.best.p10_sec <= result.best.p50_sec <= result.best.p90_sec
    assert len(result.alternatives) <= 2
    assert result.evaluated_candidates > 0


def test_case_concurrency_never_exceeds_case_count():
    tasks = [_task("tiny", 3, 1000)]
    result = optimize(
        tasks,
        _profile(),
        InfiniteResourceProvider(),
        runs=5,
        seed=5,
        throughput_cv=0.0,
    )
    assert result.best.plan.case_concurrency["tiny"] <= 3


class _ResourceProvider:
    def __init__(self, capacity: ResourceCapacity, case: CaseResources) -> None:
        self._capacity = capacity
        self._case = case

    def capacity(self) -> ResourceCapacity:
        return self._capacity

    def case_resources(self, task_id: str) -> CaseResources:
        return self._case


def _harbor_task(task_id: str, count: int, output_tokens: int = 1000) -> TaskModel:
    cases = [CaseSpec(input_tokens=0, output_tokens=output_tokens) for _ in range(count)]
    return TaskModel(
        task_id=task_id,
        group_id=task_id,
        config_path=None,
        task_type="ais_bench/harbor_agent",
        dataset_key=task_id,
        dataset_alias=task_id,
        cases=cases,
        pool=cases,
        quality="history",
        declared_case_concurrency=None,
    )


def test_resource_cap_limits_case_concurrency():
    provider = _ResourceProvider(
        ResourceCapacity(
            cpu_cores=8, mem_mb=10**6, disk_gb=10**6, networks=100, name="server"
        ),
        CaseResources(cpu=2, mem_mb=1024, network=1),
    )
    result = optimize(
        [_harbor_task("harbor", 20)],
        _profile(),
        provider,
        runs=5,
        seed=1,
        throughput_cv=0.0,
    )
    assert result.best.plan.case_concurrency["harbor"] <= 4


def test_base_resources_limit_dataset_concurrency():
    provider = _ResourceProvider(
        ResourceCapacity(cpu_cores=8, mem_mb=10**6, networks=100, name="server"),
        CaseResources(cpu=1, mem_mb=1024, cpu_base=6),
    )
    result = optimize(
        [_harbor_task("a", 4), _harbor_task("b", 4)],
        _profile(),
        provider,
        runs=5,
        seed=1,
        throughput_cv=0.0,
    )
    assert result.best.plan.dataset_concurrency == 1


def test_declared_concurrency_is_respected():
    tasks = [_task("capped", 20, 1000, declared=2)]
    result = optimize(
        tasks,
        _profile(),
        InfiniteResourceProvider(),
        runs=5,
        seed=3,
        throughput_cv=0.0,
    )
    assert result.best.plan.case_concurrency["capped"] <= 2
