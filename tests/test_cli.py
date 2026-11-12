import json

from autoeval_planner.cli import main


def _seed_run(tmp_path):
    run_dir = tmp_path / "gpqa_run" / "20260101_000000"
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
            "end_time": "2026-01-01T00:00:05+00:00",
        }
        for index in range(4)
    ]
    (predictions / "GPQA_diamond.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return run_dir


def test_dataset_import_list_and_plan(tmp_path, capsys):
    run_dir = _seed_run(tmp_path)
    db_path = tmp_path / "data" / "library.sqlite3"

    assert (
        main(
            [
                "dataset",
                "import",
                "--source",
                str(run_dir),
                "--alias",
                "gpqa_gen_0_shot_cot_chat_prompt",
                "--db",
                str(db_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["datasets"][0]["dataset_key"] == "GPQA_diamond"

    assert main(["dataset", "list", "--db", str(db_path)]) == 0
    listing = capsys.readouterr().out
    assert "GPQA_diamond" in listing
    assert "cases=4" in listing

    (tmp_path / "throughput.json").write_text(
        json.dumps({"decode_tps": 1000, "max_concurrency": 4}), encoding="utf-8"
    )
    (tmp_path / "gpqa.json").write_text(
        json.dumps(
            {
                "tool": "ais_bench",
                "subcommand": "llm_accuracy",
                "params": {"datasets": ["gpqa_gen_0_shot_cot_chat_prompt"], "batch_size": 4},
            }
        ),
        encoding="utf-8",
    )
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "target": {"base_url": "https://api.example.com/v1", "model": "m"},
                "throughput": {"mode": "file", "path": "throughput.json"},
                "tasks": [{"task_id": "gpqa", "config": "gpqa.json"}],
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"

    code = main(
        [
            "plan",
            "--tasks",
            str(tasks_path),
            "--db",
            str(db_path),
            "--output",
            str(plan_path),
            "--runs",
            "5",
        ]
    )
    assert code == 0

    report = json.loads(plan_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["plan"]["dataset_order"] == ["gpqa"]
    assert report["plan"]["case_concurrency"]["gpqa"] <= 4
    assert report["tasks"][0]["data_quality"] == "history"
    assert report["estimate"]["range"]["p10_sec"] <= report["estimate"]["range"]["p90_sec"]
    assert report["confidence"] in {"high", "medium", "low"}
    assert report["plan"]["suggested_adaptor_overrides"]["gpqa"] == {
        "batch_size": report["plan"]["case_concurrency"]["gpqa"]
    }


def test_dataset_add_is_reserved(tmp_path):
    assert main(["dataset", "add", "--db", str(tmp_path / "x.sqlite3")]) == 4


def test_plan_requires_existing_tasks_file(tmp_path):
    assert main(["plan", "--tasks", str(tmp_path / "missing.json")]) == 2


def test_probe_rejects_invalid_header():
    assert (
        main(
            [
                "probe",
                "--base-url",
                "http://127.0.0.1:1/v1",
                "--model",
                "m",
                "--header",
                "bad-header",
            ]
        )
        == 2
    )


def test_probe_rejects_invalid_body():
    assert (
        main(
            [
                "probe",
                "--base-url",
                "http://127.0.0.1:1/v1",
                "--model",
                "m",
                "--body",
                "[]",
            ]
        )
        == 2
    )
