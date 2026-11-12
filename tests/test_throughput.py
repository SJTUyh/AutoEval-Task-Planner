import pytest

from autoeval_planner.errors import ConfigError
from autoeval_planner.throughput import Level, ThroughputProfile


def test_manual_shorthand_expands_to_two_levels():
    profile = ThroughputProfile.from_dict(
        {"decode_tps": 1000, "max_concurrency": 8}, source="shorthand"
    )
    assert profile.max_concurrency == 8
    assert profile.peak_decode_tps == pytest.approx(1000)
    assert profile.rate_at(1) == pytest.approx(125)
    assert profile.rate_at(4) == pytest.approx(500)
    assert profile.rate_at(99) == pytest.approx(1000)


def test_level_interpolation_and_round_trip(tmp_path):
    profile = ThroughputProfile(
        source="manual",
        levels=[
            Level(concurrency=1, decode_tps=100, ttft_sec=1.0),
            Level(concurrency=4, decode_tps=400, ttft_sec=2.0),
        ],
    )
    assert profile.rate_at(2) == pytest.approx(200)
    assert profile.ttft_at(2) == pytest.approx(1.3333, rel=1e-3)

    path = tmp_path / "throughput.json"
    profile.save(path)
    loaded = ThroughputProfile.load(path)
    assert loaded.max_concurrency == 4
    assert loaded.rate_at(4) == pytest.approx(400)


def test_scaled_profile_keeps_concurrency():
    profile = ThroughputProfile(
        source="manual",
        levels=[Level(concurrency=1, decode_tps=100), Level(concurrency=8, decode_tps=800)],
    )
    scaled = profile.scaled(0.5)
    assert scaled.rate_at(8) == pytest.approx(400)
    assert scaled.max_concurrency == 8


def test_invalid_levels_are_rejected():
    with pytest.raises(ConfigError):
        ThroughputProfile(source="manual", levels=[])
    with pytest.raises(ConfigError):
        ThroughputProfile(source="manual", levels=[Level(concurrency=1, decode_tps=0)])
