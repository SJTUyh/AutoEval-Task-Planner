import json

import pytest

from autoeval_planner.config import Estimate, TaskSet, TaskSpec, ThroughputSpec, Target, SimulationSpec, ResourceSpec
from autoeval_planner.errors import InsufficientDataError
from autoeval_planner.library.db import Library
from autoeval_planner.library.importer import import_source
from autoeval_planner.taskset import build_task_models, plan_tasks


def _taskset(tmp_path, tasks, config_path=None) -> TaskSet:
    return TaskSet(
        path=tmp_path / "tasks.json",
        base_dir=tmp_path,
        target=Target(base_url="http://x/v1", model="m"),
        throughput=ThroughputSpec(mode="inline", inline={"decode_tps": 100, "max_concurrency": 4}),
        resources=ResourceSpec(),
        tasks=tuple(tasks),
        simulation=SimulationSpec(runs=3, seed=1),
    )


def _write_config(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return name


def _seed_library(tmp_path) -> tuple[Library, str]:
    run_dir = tmp_path / "run" / "20260101_000000"
    predictions = run_dir / "predictions" / "model"
    predictions.mkdir(parents=True)
    rows = [
        {
            "data_abbr": "GPQA_diamond",
            "id": index,
            "success": True,
            "input_tokens": 100,
            "output_tokens": 1000,
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:10+00:00",
        }
        for index in range(10)
    ]
    (predictions / "GPQA_diamond.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    library = Library(tmp_path / "lib.sqlite3")
    import_source(
        library,
        run_dir,
        task_type="ais_bench/llm_accuracy",
        alias="gpqa_gen_0_shot_cot_chat_prompt",
    )
    return library, "gpqa_gen_0_shot_cot_chat_prompt"


def test_plan_accuracy_partial_config(tmp_path):
    library, alias = _seed_library(tmp_path)
    config = _write_config(
        tmp_path,
        "gpqa.json",
        {
            "tool": "ais_bench",
            "subcommand": "llm_accuracy",
            "params": {"datasets": [alias], "dataset_test_range": "[0:6]", "batch_size": 2},
        },
    )
    taskset = _taskset(tmp_path, [TaskSpec(task_id="gpqa", config_path=config)])

    planned = plan_tasks(taskset, library)
    assert len(planned) == 1
    assert planned[0].dataset_key == "GPQA_diamond"
    assert planned[0].case_limit == 6
    assert planned[0].declared_case_concurrency == 2

    models, warnings = build_task_models(planned, library, seed=1, strict=False)
    assert warnings == []
    assert models[0].quality == "history"
    assert models[0].case_count == 6

    library.close()


def test_case_count_override_resamples_history(tmp_path):
    library, alias = _seed_library(tmp_path)
    config = _write_config(
        tmp_path,
        "gpqa.json",
        {
            "tool": "ais_bench",
            "subcommand": "llm_accuracy",
            "params": {"datasets": [alias]},
        },
    )
    taskset = _taskset(
        tmp_path,
        [TaskSpec(task_id="gpqa", config_path=config, case_count=4)],
    )
    planned = plan_tasks(taskset, library)
    models, warnings = build_task_models(planned, library, seed=1, strict=False)

    assert models[0].case_count == 4
    assert any("重采样" in warning for warning in warnings)
    library.close()


def test_multi_dataset_config_splits_tasks(tmp_path):
    library, alias = _seed_library(tmp_path)
    config = _write_config(
        tmp_path,
        "multi.json",
        {
            "tool": "ais_bench",
            "subcommand": "llm_accuracy",
            "params": {"datasets": [alias, "another_alias"]},
        },
    )
    taskset = _taskset(tmp_path, [TaskSpec(task_id="multi", config_path=config)])
    planned = plan_tasks(taskset, library)

    assert [task.task_id for task in planned] == [f"multi:{alias}", "multi:another_alias"]
    assert planned[1].dataset_key is None
    library.close()


def test_estimate_used_when_no_history(tmp_path):
    config = _write_config(
        tmp_path,
        "mmmu.json",
        {
            "tool": "ais_bench",
            "subcommand": "llm_accuracy",
            "params": {"datasets": ["mmmu_pro_options10_gen"]},
        },
    )
    estimate = Estimate(
        case_count=5,
        input_tokens_median=100,
        input_tokens_p90=200,
        output_tokens_median=500,
        output_tokens_p90=1000,
        overhead_sec=0.0,
    )
    taskset = _taskset(
        tmp_path,
        [TaskSpec(task_id="mmmu", config_path=config, estimate=estimate)],
    )
    with Library(tmp_path / "lib.sqlite3") as library:
        planned = plan_tasks(taskset, library)
        models, warnings = build_task_models(planned, library, seed=1, strict=False)

    assert models[0].quality == "estimate"
    assert models[0].case_count == 5
    assert any("估算" in warning for warning in warnings)


def test_harbor_attempts_multiply_cases(tmp_path):
    config = _write_config(
        tmp_path,
        "deepswe.json",
        {
            "tool": "ais_bench",
            "subcommand": "harbor_agent",
            "params": {
                "dataset_abbr": "harbor_deepswe-v1.1",
                "n_attempts": 3,
                "n_tasks": 4,
                "n_concurrent_trials": 8,
            },
        },
    )
    estimate = Estimate(
        case_count=4,
        input_tokens_median=100,
        input_tokens_p90=200,
        output_tokens_median=500,
        output_tokens_p90=1000,
        overhead_sec=10.0,
    )
    taskset = _taskset(
        tmp_path,
        [TaskSpec(task_id="deepswe", config_path=config, estimate=estimate)],
    )
    with Library(tmp_path / "lib.sqlite3") as library:
        planned = plan_tasks(taskset, library)
        models, _ = build_task_models(planned, library, seed=1, strict=False)

    assert planned[0].attempts == 3
    assert models[0].case_count == 4 * 3
    assert models[0].declared_case_concurrency == 8


def test_strict_mode_rejects_heuristic(tmp_path):
    config = _write_config(
        tmp_path,
        "unknown.json",
        {
            "tool": "ais_bench",
            "subcommand": "llm_accuracy",
            "params": {"datasets": ["no-history-alias"]},
        },
    )
    taskset = _taskset(tmp_path, [TaskSpec(task_id="unknown", config_path=config)])
    with Library(tmp_path / "lib.sqlite3") as library:
        planned = plan_tasks(taskset, library)
        with pytest.raises(InsufficientDataError):
            build_task_models(planned, library, seed=1, strict=True)
