import json

from autoeval_planner.library.db import Library
from autoeval_planner.library.importer import detect_source, import_source


def _write_predictions(tmp_path, dataset_abbr="GPQA_diamond", rows=None):
    run_dir = tmp_path / "gpqa_run" / "20260101_000000"
    predictions = run_dir / "predictions" / "vllm-api-general-chat"
    predictions.mkdir(parents=True)
    rows = rows or [
        {
            "data_abbr": dataset_abbr,
            "id": 1,
            "success": True,
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:02+00:00",
            "input_tokens": 100,
            "output_tokens": 200,
        },
        {
            "data_abbr": dataset_abbr,
            "id": 2,
            "success": False,
            "start_time": "2026-01-01T00:00:03+00:00",
            "end_time": "2026-01-01T00:00:05+00:00",
            "input_tokens": 50,
            "output_tokens": 300,
        },
    ]
    path = predictions / f"{dataset_abbr}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return run_dir


def test_detect_ais_bench_run(tmp_path):
    run_dir = _write_predictions(tmp_path)
    info = detect_source(run_dir)
    assert info.kind == "ais_bench_run"
    assert info.run_id == "gpqa_run_20260101_000000"


def test_import_run_registers_dataset_and_is_idempotent(tmp_path):
    run_dir = _write_predictions(tmp_path)
    db_path = tmp_path / "library.sqlite3"

    with Library(db_path) as library:
        summary = import_source(
            library,
            run_dir,
            task_type="ais_bench/llm_accuracy",
            alias="gpqa_gen_0_shot_cot_chat_prompt",
            observed_concurrency=8,
            decode_tps=1000.0,
        )
        assert summary.datasets[0].dataset_key == "GPQA_diamond"
        assert summary.datasets[0].case_count == 2
        assert summary.datasets[0].input_tokens_total == 150
        assert summary.datasets[0].output_tokens_total == 500
        assert summary.datasets[0].error_cases == 1

        assert library.lookup_alias("gpqa_gen_0_shot_cot_chat_prompt", "ais_bench/llm_accuracy") == "GPQA_diamond"
        cases = library.load_cases("GPQA_diamond")
        assert len(cases) == 2
        # 200 tokens * 8 / 1000 = 1.6s，实测 2s，剩余 0.4s 计入 overhead
        assert abs(cases[0]["overhead_sec"] - 0.4) < 1e-6

        import_source(
            library,
            run_dir,
            task_type="ais_bench/llm_accuracy",
            alias="gpqa_gen_0_shot_cot_chat_prompt",
            observed_concurrency=8,
            decode_tps=1000.0,
        )
        assert library.count_cases("GPQA_diamond") == 2

        datasets = library.list_datasets()
        assert len(datasets) == 1
        assert datasets[0].run_count == 1
        assert datasets[0].aliases == (
            "GPQA_diamond",
            "gpqa_gen_0_shot_cot_chat_prompt",
        )


def test_import_unified_result(tmp_path):
    path = tmp_path / "run" / "unified_result.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool": "ais_bench",
                "subcommand": "harbor_agent",
                "case_count": 1,
                "cases": [
                    {
                        "case_id": "task-1",
                        "attempt": 1,
                        "status": "success",
                        "duration_sec": 120.0,
                        "input_tokens": 5000,
                        "output_tokens": 800,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:02:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "library.sqlite3"

    with Library(db_path) as library:
        summary = import_source(
            library,
            path,
            task_type="ais_bench/harbor_agent",
            alias="harbor_deepswe-v1.1",
        )
        assert summary.source_type == "unified_result"
        assert summary.datasets[0].dataset_key == "harbor_deepswe-v1.1"
        assert library.load_cases("harbor_deepswe-v1.1")[0]["output_tokens"] == 800


def test_profile_dedupes_case_ids_across_runs():
    from autoeval_planner.library.profile import profile_from_rows

    rows = [
        {"case_id": "a", "input_tokens": 100, "output_tokens": 1000, "overhead_sec": 0.0, "status": "success"},
        {"case_id": "a", "input_tokens": 300, "output_tokens": 3000, "overhead_sec": 60.0, "status": "success"},
        {"case_id": "b", "input_tokens": 200, "output_tokens": 2000, "overhead_sec": 30.0, "status": "success"},
    ]
    profile = profile_from_rows("d", "ais_bench/harbor_agent", rows, ["r1", "r2"])

    assert profile.case_count == 2
    assert profile.cases[0].output_tokens == 2000
    assert profile.cases[0].overhead_sec == 30.0


def test_import_harbor_run(tmp_path):
    run_dir = tmp_path / "deepswe_run" / "20260101_000000"
    details = run_dir / "results" / "mini-swe-agent" / "harbor_deepswe" / "details"
    for index, status in enumerate(["ok", "failed"]):
        trial = details / f"task-{index}__abc"
        trial.mkdir(parents=True)
        payload = {
            "task_name": f"task-{index}",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:10:00+00:00",
            "agent_result": {"n_input_tokens": 1000, "n_output_tokens": 100},
            "step_results": [{"agent_result": {"n_input_tokens": 500, "n_output_tokens": 50}}],
            "exception_info": None if status == "ok" else {"exception_type": "RuntimeError"},
        }
        (trial / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    db_path = tmp_path / "library.sqlite3"
    with Library(db_path) as library:
        summary = import_source(
            library,
            run_dir,
            task_type="ais_bench/harbor_agent",
            alias="harbor_deepswe-v1.1",
            observed_concurrency=16,
            decode_tps=2300.0,
        )
        assert summary.source_type == "harbor_run"
        assert summary.datasets[0].case_count == 2
        assert summary.datasets[0].input_tokens_total == 3000
        assert summary.datasets[0].output_tokens_total == 300
        assert summary.datasets[0].error_cases == 1
        assert library.lookup_alias("harbor_deepswe-v1.1", "ais_bench/harbor_agent") == "harbor_deepswe"

        cases = library.load_cases("harbor_deepswe")
        assert {case["status"] for case in cases} == {"success", "failed"}
        assert cases[0]["duration_sec"] == 600.0
        # 150 tokens * 16 / 2300 ≈ 1.04s，剩余 598.96s 计入 overhead
        assert cases[0]["overhead_sec"] > 590


def test_dry_run_does_not_write(tmp_path):
    run_dir = _write_predictions(tmp_path)
    db_path = tmp_path / "library.sqlite3"

    with Library(db_path) as library:
        import_source(
            library,
            run_dir,
            task_type="ais_bench/llm_accuracy",
            dry_run=True,
        )
        assert library.list_datasets() == []
