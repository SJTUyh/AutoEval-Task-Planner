import pytest

from autoeval_planner.config import ResourceSpec
from autoeval_planner.errors import ConfigError
from autoeval_planner.library.db import Library, ResourceProfile, ServerSnapshot
from autoeval_planner.resource import parse_probe_output, snapshot_from_values
from autoeval_planner.resources import (
    CaseResources,
    ResourceCapacity,
    ServerResourceProvider,
    build_provider,
    dataset_cap,
    resource_caps,
)


class _Task:
    def __init__(self, task_id: str, dataset_key: str, task_type: str) -> None:
        self.task_id = task_id
        self.dataset_key = dataset_key
        self.task_type = task_type


def test_parse_probe_output_english():
    text = """__CPU__
72
__MEM__
              total        used        free      shared  buff/cache   available
Mem:         771953       22577       44961         657      704414      744384
Swap:          4095         537        3558
__DISK__
Avail
3300G
__NET__
pool: 172.28.0.0/14 size=24 (cap=1024 used=0 free=1024)
existing_networks=4
available=1024
"""
    snapshot = parse_probe_output(text, name="s1")
    assert snapshot.cpu_cores == 72
    assert snapshot.mem_mb == 744384
    assert snapshot.disk_gb == 3300
    assert snapshot.network_available == 1024
    assert snapshot.docker_networks == 4
    assert "172.28.0.0/14" in (snapshot.network_pool or "")


def test_parse_probe_output_chinese():
    text = """__CPU__
64
__MEM__
Mem:  1000  100  200  0  500  800
__DISK__
Avail
100G
__NET__
池配置: 172.17.0.0/12 size=16 (容量1, 已占0, 可用1)
现存网络总数: 3
★ 当前可用网络数: 1
"""
    snapshot = parse_probe_output(text, name="s2")
    assert snapshot.cpu_cores == 64
    assert snapshot.mem_mb == 800
    assert snapshot.network_available == 1
    assert snapshot.docker_networks == 3


def test_snapshot_values_validation():
    with pytest.raises(ConfigError):
        snapshot_from_values(name="x", cpu_cores=0, mem_mb=1)
    snapshot = snapshot_from_values(
        name="x", cpu_cores=8, mem_mb=1024, disk_gb=100, network_available=10
    )
    assert snapshot.source == "manual"


def test_resource_caps_and_dataset_cap():
    capacity = ResourceCapacity(
        cpu_cores=8,
        mem_mb=16384,
        disk_gb=1000,
        networks=10,
        name="test",
    )
    provider = ServerResourceProvider(
        capacity,
        {
            "harbor": CaseResources(cpu=2, mem_mb=4096, disk_gb=20, network=1),
            "llm": CaseResources(cpu=0.02, mem_mb=16, cpu_base=1.0, mem_mb_base=1024),
        },
    )
    tasks = [
        _Task("harbor", "d1", "ais_bench/harbor_agent"),
        _Task("llm", "d2", "ais_bench/llm_accuracy"),
    ]
    caps = resource_caps(tasks, provider, max_case_concurrency=256)
    # 8 核 - 1 核基础资源 = 7 核预算，每 trial 2 核 => 3
    assert caps["harbor"] == 3
    assert dataset_cap([_Task("llm", "d2", "ais_bench/llm_accuracy")], provider) == 8


def test_build_provider_applies_headroom_and_defaults(tmp_path):
    with Library(tmp_path / "lib.sqlite3") as library:
        library.upsert_server(
            ServerSnapshot(
                name="srv",
                cpu_cores=8,
                mem_mb=16000,
                disk_gb=1000,
                network_available=10,
                network_pool=None,
                docker_networks=0,
                captured_at="2026-01-01T00:00:00+08:00",
                source="manual",
            )
        )
        tasks = [_Task("harbor", "deepswe", "ais_bench/harbor_agent")]

        provider = build_provider(
            ResourceSpec(mode="server", server="srv", headroom=0.5), library, tasks
        )
        capacity = provider.capacity()
        assert capacity.cpu_cores == 4
        assert capacity.mem_mb == 8000

        profile = library.get_dataset_resources("deepswe", "ais_bench/harbor_agent")
        assert profile is not None
        assert profile.cpu_per_case == 2
        assert profile.mem_mb_per_case == 8192
        assert profile.source == "default"

        library.upsert_dataset_resources(
            "deepswe",
            "ais_bench/harbor_agent",
            ResourceProfile(
                cpu_base=0.5,
                mem_mb_base=256,
                cpu_per_case=4,
                mem_mb_per_case=2048,
                disk_gb_per_case=10,
                networks_per_case=1,
                source="manual",
            ),
        )
        provider = build_provider(
            ResourceSpec(mode="server", server="srv", headroom=0.5), library, tasks
        )
        case = provider.case_resources("harbor")
        assert case.cpu == 4
        assert case.mem_mb == 2048
        assert case.disk_gb == 10
        assert case.source == "manual"


def test_build_provider_requires_server_snapshot(tmp_path):
    with Library(tmp_path / "lib.sqlite3") as library:
        with pytest.raises(ConfigError, match="资源快照"):
            build_provider(
                ResourceSpec(mode="server", server="missing"), library, []
            )
