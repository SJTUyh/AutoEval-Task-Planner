import pytest

from autoeval_planner.errors import ConfigError
from autoeval_planner.library.profile import CaseSpec
from autoeval_planner.resources import CaseResources, ResourceCapacity
from autoeval_planner.simulate import Plan, percentile, simulate
from autoeval_planner.throughput import Level, ThroughputProfile


def _profile() -> ThroughputProfile:
    return ThroughputProfile(
        source="manual",
        levels=[Level(concurrency=1, decode_tps=100.0), Level(concurrency=4, decode_tps=400.0)],
        peak_prefill_tps=8000.0,
    )


class _CappedProvider:
    def __init__(self, cap: int) -> None:
        self._cap = cap

    def capacity(self) -> ResourceCapacity:
        return ResourceCapacity(max_case_concurrency=self._cap)


def _cases(count: int, output_tokens: int = 100) -> list[CaseSpec]:
    return [CaseSpec(input_tokens=0, output_tokens=output_tokens) for _ in range(count)]


def test_single_task_scales_with_concurrency():
    from autoeval_planner.resources import InfiniteResourceProvider

    provider = InfiniteResourceProvider()
    cases = {"t": _cases(8)}

    serial = simulate(cases, Plan(["t"], 1, {"t": 1}), _profile(), provider)
    parallel = simulate(cases, Plan(["t"], 1, {"t": 4}), _profile(), provider)

    assert serial.makespan_sec == pytest.approx(8.0, rel=1e-6)
    assert parallel.makespan_sec == pytest.approx(2.0, rel=1e-6)
    assert parallel.utilization == pytest.approx(1.0, rel=1e-6)


def test_global_concurrency_cap_is_respected():
    cases = {"t": _cases(8)}
    result = simulate(cases, Plan(["t"], 1, {"t": 4}), _profile(), _CappedProvider(2))
    assert result.makespan_sec == pytest.approx(4.0, rel=1e-6)


def test_dataset_concurrency_parallelizes_tasks():
    cases = {"a": _cases(1), "b": _cases(1)}
    serial = simulate(cases, Plan(["a", "b"], 1, {"a": 1, "b": 1}), _profile(), _CappedProvider(2))
    both = simulate(cases, Plan(["a", "b"], 2, {"a": 1, "b": 1}), _profile(), _CappedProvider(2))

    assert serial.makespan_sec == pytest.approx(2.0, rel=1e-6)
    assert both.makespan_sec == pytest.approx(1.0, rel=1e-6)


def test_overhead_does_not_consume_service_throughput():
    from autoeval_planner.resources import InfiniteResourceProvider

    cases = {"t": [CaseSpec(input_tokens=0, output_tokens=0, overhead_sec=5.0)]}
    result = simulate(cases, Plan(["t"], 1, {"t": 1}), _profile(), InfiniteResourceProvider())
    assert result.makespan_sec == pytest.approx(5.0, rel=1e-6)


class _ResourceProvider:
    def __init__(self, capacity: ResourceCapacity, case: CaseResources) -> None:
        self._capacity = capacity
        self._case = case

    def capacity(self) -> ResourceCapacity:
        return self._capacity

    def case_resources(self, task_id: str) -> CaseResources:
        return self._case


def test_cpu_capacity_limits_concurrency():
    cases = {"t": _cases(4)}
    provider = _ResourceProvider(
        ResourceCapacity(cpu_cores=4, name="server"),
        CaseResources(cpu=2, mem_mb=1024),
    )
    result = simulate(cases, Plan(["t"], 1, {"t": 4}), _profile(), provider)

    # 4 核 / 每 case 2 核 => 最多 2 并发，4 个 case 需要 2 波
    assert result.makespan_sec == pytest.approx(2.0, rel=1e-6)
    assert result.resource_limited is True
    assert result.binding_resource == "cpu"
    assert result.resource_peaks["cpu"] == 4.0


def test_network_capacity_limits_concurrency():
    cases = {"t": _cases(4)}
    provider = _ResourceProvider(
        ResourceCapacity(networks=1, name="server"),
        CaseResources(network=1),
    )
    result = simulate(cases, Plan(["t"], 1, {"t": 4}), _profile(), provider)

    assert result.makespan_sec == pytest.approx(4.0, rel=1e-6)
    assert result.binding_resource == "networks"


def test_single_case_larger_than_capacity_is_rejected():
    cases = {"t": _cases(1)}
    provider = _ResourceProvider(
        ResourceCapacity(cpu_cores=1, name="server"),
        CaseResources(cpu=2),
    )
    with pytest.raises(ConfigError, match="资源不足"):
        simulate(cases, Plan(["t"], 1, {"t": 1}), _profile(), provider)


def test_base_resources_are_held_while_task_runs():
    cases = {"a": _cases(1), "b": _cases(1)}
    provider = _ResourceProvider(
        ResourceCapacity(cpu_cores=2, name="server"),
        CaseResources(cpu_base=1.5),
    )
    with pytest.raises(ConfigError, match="基础资源"):
        simulate(cases, Plan(["a", "b"], 2, {"a": 1, "b": 1}), _profile(), provider)


def test_percentile():
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([5], 0.9) == 5.0
