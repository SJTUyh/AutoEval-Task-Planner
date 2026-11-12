import json

import pytest

from autoeval_planner.config import parse_taskset
from autoeval_planner.errors import ConfigError


def _base() -> dict:
    return {
        "target": {"base_url": "https://api.example.com/v1", "model": "m"},
        "throughput": {"mode": "inline", "inline": {"decode_tps": 1000, "max_concurrency": 8}},
        "tasks": [{"task_id": "gpqa", "config": "cfg.json"}],
    }


def test_parse_minimal_taskset():
    taskset = parse_taskset(_base(), base_dir=None, source="test")

    assert taskset.target.base_url == "https://api.example.com/v1"
    assert taskset.throughput.mode == "inline"
    assert taskset.resources.mode == "infinite"
    assert taskset.resources.headroom == 0.2
    assert taskset.simulation.runs == 100
    assert len(taskset.tasks) == 1
    assert taskset.tasks[0].config_path == "cfg.json"


def test_unknown_top_level_field_is_rejected():
    raw = _base()
    raw["extra"] = 1
    with pytest.raises(ConfigError, match="未知字段"):
        parse_taskset(raw, source="test")


def test_task_requires_exactly_one_config_source():
    raw = _base()
    raw["tasks"] = [{"task_id": "a", "config": "x.json", "config_object": {}}]
    with pytest.raises(ConfigError, match="只能提供 config 或 config_object"):
        parse_taskset(raw, source="test")


def test_estimate_p90_must_not_be_below_median():
    raw = _base()
    raw["tasks"][0]["estimate"] = {
        "case_count": 10,
        "input_tokens_median": 100,
        "input_tokens_p90": 50,
        "output_tokens_median": 100,
        "output_tokens_p90": 200,
    }
    with pytest.raises(ConfigError, match="p90"):
        parse_taskset(raw, source="test")


def test_unknown_estimate_field_is_rejected():
    raw = _base()
    raw["tasks"][0]["estimate"] = {
        "case_count": 10,
        "input_tokens_median": 100,
        "input_tokens_p90": 200,
        "output_tokens_median": 100,
        "output_tokens_p90": 200,
        "oops": 1,
    }
    with pytest.raises(ConfigError, match="未知字段"):
        parse_taskset(raw, source="test")


def test_api_key_is_not_part_of_taskset():
    raw = _base()
    raw["target"]["api_key"] = "sk-test"
    with pytest.raises(ConfigError, match="未知字段"):
        parse_taskset(raw, source="test")


def test_resources_server_mode_requires_server():
    raw = _base()
    raw["resources"] = {"mode": "server"}
    with pytest.raises(ConfigError, match="server"):
        parse_taskset(raw, source="test")


def test_resources_server_mode_parsed():
    raw = _base()
    raw["resources"] = {"mode": "server", "server": "srv", "headroom": 0.05}
    taskset = parse_taskset(raw, source="test")
    assert taskset.resources.mode == "server"
    assert taskset.resources.server == "srv"
    assert taskset.resources.headroom == 0.05


def test_resources_mode_static_is_rejected():
    raw = _base()
    raw["resources"] = {"mode": "static"}
    with pytest.raises(ConfigError, match="infinite"):
        parse_taskset(raw, source="test")


def test_load_taskset_from_file(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(_base()), encoding="utf-8")
    from autoeval_planner.config import load_taskset

    taskset = load_taskset(path)
    assert taskset.path == path
