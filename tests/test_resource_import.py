from autoeval_planner.cli import main
from autoeval_planner.library.db import Library
from autoeval_planner.resource import (
    load_tasks_dir,
    resource_profile_from_tasks,
    upsert_harbor_resources,
)

TASK_A = """[environment]
cpus = 2
memory_mb = 8192
storage_mb = 20480

[verifier.environment]
cpus = 4
memory_mb = 8192

[agent]
network_mode = "no-network"
"""

TASK_B = """[environment]
cpus = 4
memory_mb = 16384
storage_mb = 40960

[verifier.environment]
cpus = 4
memory_mb = 16384

[agent]
network_mode = "no-network"
"""


def _write_tasks(root):
    for name, content in (("task-a", TASK_A), ("task-b", TASK_B)):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "task.toml").write_text(content, encoding="utf-8")
    return root


def test_load_tasks_dir_and_aggregate(tmp_path):
    root = _write_tasks(tmp_path / "tasks")

    documents = load_tasks_dir(root)
    profile, detail = resource_profile_from_tasks(documents, "tasks-dir:test")

    assert profile.task_count == 2
    assert profile.cpu_per_case == 3.0
    assert profile.max_cpu_per_case == 4.0
    assert profile.mem_mb_per_case == 12288
    assert profile.disk_gb_per_case == 30.0
    assert profile.networks_per_case == 1
    assert detail["verifier_cpu_median"] == 4.0
    assert detail["network_modes"] == ["no-network"]


def test_upsert_harbor_resources(tmp_path):
    root = _write_tasks(tmp_path / "tasks")
    documents = load_tasks_dir(root)

    with Library(tmp_path / "lib.sqlite3") as library:
        profile = upsert_harbor_resources(
            library,
            dataset_key="harbor_deepswe-v1.1",
            task_type="ais_bench/harbor_agent",
            documents=documents,
            source="tasks-dir:test",
        )
        stored = library.get_dataset_resources(
            "harbor_deepswe-v1.1", "ais_bench/harbor_agent"
        )
        assert stored is not None
        assert stored.cpu_per_case == 3.0
        assert stored.task_count == 2
        assert profile.source == "tasks-dir:test"


def test_resource_cli_set_import_list_show(tmp_path, capsys):
    root = _write_tasks(tmp_path / "tasks")
    db_path = tmp_path / "lib.sqlite3"

    assert (
        main(
            [
                "resource",
                "set",
                "--name",
                "srv",
                "--cpu",
                "8",
                "--mem-mb",
                "16384",
                "--disk-gb",
                "100",
                "--network-available",
                "10",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "srv" in output
    assert "8 核" in output

    assert (
        main(
            [
                "resource",
                "import-harbor",
                "--dataset-key",
                "harbor_deepswe-v1.1",
                "--tasks-dir",
                str(root),
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "已导入 2 个 task" in output
    assert '"cpu_per_case": 3.0' in output

    assert main(["resource", "show", "--dataset-key", "harbor_deepswe-v1.1", "--db", str(db_path)]) == 0
    output = capsys.readouterr().out
    assert '"task_count": 2' in output

    assert main(["resource", "list", "--db", str(db_path)]) == 0
    output = capsys.readouterr().out
    assert "srv" in output
    assert "harbor_deepswe-v1.1" in output
