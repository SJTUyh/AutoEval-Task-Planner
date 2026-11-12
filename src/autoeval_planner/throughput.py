"""推理服务的并发-吞吐曲线。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import ConfigError

SCHEMA_VERSION = 1
PROBE_SOURCE = "probe"
MANUAL_SOURCE = "manual"


@dataclass(frozen=True)
class Level:
    concurrency: int
    decode_tps: float
    prefill_tps: float | None = None
    ttft_sec: float | None = None
    error_rate: float = 0.0
    usage_estimated: bool = False


@dataclass
class ThroughputProfile:
    source: str
    levels: list[Level]
    base_url: str | None = None
    model: str | None = None
    effort: str | None = None
    measured_at: str | None = None
    max_concurrency: int = 1
    peak_decode_tps: float = 1.0
    peak_prefill_tps: float | None = None
    notes: str | None = None
    request: dict | None = None

    def __post_init__(self) -> None:
        if not self.levels:
            raise ConfigError("吞吐曲线至少需要一个并发档位")
        self.levels = sorted(self.levels, key=lambda level: level.concurrency)
        if self.levels[0].concurrency < 1:
            raise ConfigError("吞吐档位 concurrency 必须大于等于 1")
        if any(level.decode_tps <= 0 for level in self.levels):
            raise ConfigError("吞吐档位 decode_tps 必须大于 0")
        self.max_concurrency = self.levels[-1].concurrency
        self.peak_decode_tps = max(level.decode_tps for level in self.levels)
        prefill_values = [level.prefill_tps for level in self.levels if level.prefill_tps]
        self.peak_prefill_tps = max(prefill_values) if prefill_values else None

    # -- 查询 -------------------------------------------------------------
    def rate_at(self, concurrency: int) -> float:
        """给定全局在途请求数，返回可用的聚合 decode 吞吐（tok/s）。"""
        c = max(1, min(int(concurrency), self.max_concurrency))
        points = [(0, 0.0)] + [
            (level.concurrency, level.decode_tps) for level in self.levels
        ]
        return _interpolate(points, c)

    def ttft_at(self, concurrency: int) -> float | None:
        points = [
            (level.concurrency, level.ttft_sec)
            for level in self.levels
            if level.ttft_sec is not None
        ]
        if not points:
            return None
        c = max(1, min(int(concurrency), self.max_concurrency))
        return _interpolate(points, c)

    def prefill_tps(self) -> float | None:
        if self.peak_prefill_tps:
            return self.peak_prefill_tps
        return None

    def scaled(self, factor: float) -> "ThroughputProfile":
        return ThroughputProfile(
            source=self.source,
            levels=[
                replace(
                    level,
                    decode_tps=level.decode_tps * factor,
                    prefill_tps=(
                        level.prefill_tps * factor if level.prefill_tps is not None else None
                    ),
                )
                for level in self.levels
            ],
            base_url=self.base_url,
            model=self.model,
            effort=self.effort,
            measured_at=self.measured_at,
            notes=self.notes,
            request=self.request,
        )

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "base_url": self.base_url,
            "model": self.model,
            "effort": self.effort,
            "measured_at": self.measured_at,
            "levels": [
                {
                    "concurrency": level.concurrency,
                    "decode_tps": round(level.decode_tps, 3),
                    "prefill_tps": (
                        round(level.prefill_tps, 3) if level.prefill_tps is not None else None
                    ),
                    "ttft_sec": round(level.ttft_sec, 4) if level.ttft_sec is not None else None,
                    "error_rate": round(level.error_rate, 4),
                    "usage_estimated": level.usage_estimated,
                }
                for level in self.levels
            ],
            "request": self.request,
            "derived": {
                "max_concurrency": self.max_concurrency,
                "peak_decode_tps": round(self.peak_decode_tps, 3),
                "peak_prefill_tps": (
                    round(self.peak_prefill_tps, 3) if self.peak_prefill_tps else None
                ),
            },
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "ThroughputProfile":
        source = Path(path).expanduser()
        if not source.is_file():
            raise ConfigError(f"吞吐曲线文件不存在: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"吞吐曲线文件 {source} 不是合法 JSON: {exc}") from exc
        return cls.from_dict(payload, source=str(source))

    @classmethod
    def from_dict(cls, raw: Any, source: str = "<throughput>") -> "ThroughputProfile":
        if not isinstance(raw, dict):
            raise ConfigError(f"{source} 顶层必须是 JSON 对象")

        levels_raw = raw.get("levels")
        if levels_raw is None:
            return cls._from_shorthand(raw, source)

        if not isinstance(levels_raw, list) or not levels_raw:
            raise ConfigError(f"{source}.levels 必须是非空数组")
        levels: list[Level] = []
        for index, item in enumerate(levels_raw):
            where = f"{source}.levels[{index}]"
            if not isinstance(item, dict):
                raise ConfigError(f"{where} 必须是 JSON 对象")
            concurrency = _require_number(item, "concurrency", where)
            decode_tps = _require_number(item, "decode_tps", where)
            if concurrency != int(concurrency) or concurrency < 1:
                raise ConfigError(f"{where}.concurrency 必须是正整数")
            levels.append(
                Level(
                    concurrency=int(concurrency),
                    decode_tps=decode_tps,
                    prefill_tps=_optional_number(item, "prefill_tps", where),
                    ttft_sec=_optional_number(item, "ttft_sec", where),
                    error_rate=_optional_number(item, "error_rate", where) or 0.0,
                    usage_estimated=bool(item.get("usage_estimated")),
                )
            )

        return cls(
            source=str(raw.get("source") or MANUAL_SOURCE),
            levels=levels,
            base_url=_optional_str(raw.get("base_url")),
            model=_optional_str(raw.get("model")),
            effort=_optional_str(raw.get("effort")),
            measured_at=_optional_str(raw.get("measured_at")),
            notes=_optional_str(raw.get("notes")),
            request=raw.get("request") if isinstance(raw.get("request"), dict) else None,
        )

    @classmethod
    def _from_shorthand(cls, raw: dict, source: str) -> "ThroughputProfile":
        decode_tps = _require_number(raw, "decode_tps", source)
        max_concurrency_raw = _require_number(raw, "max_concurrency", source)
        if max_concurrency_raw != int(max_concurrency_raw) or max_concurrency_raw < 1:
            raise ConfigError(f"{source}.max_concurrency 必须是正整数")
        max_concurrency = int(max_concurrency_raw)
        prefill_tps = _optional_number(raw, "prefill_tps", source)
        ttft_sec = _optional_number(raw, "ttft_sec", source)

        # 简写形式假设并发线性扩展：c=1 时吞吐为峰值 / max_concurrency。
        single_tps = decode_tps / max_concurrency
        levels = [
            Level(
                concurrency=1,
                decode_tps=single_tps,
                prefill_tps=prefill_tps,
                ttft_sec=ttft_sec,
                error_rate=0.0,
            )
        ]
        if max_concurrency > 1:
            levels.append(
                Level(
                    concurrency=max_concurrency,
                    decode_tps=decode_tps,
                    prefill_tps=prefill_tps,
                    ttft_sec=ttft_sec,
                    error_rate=0.0,
                )
            )
        return cls(
            source=str(raw.get("source") or MANUAL_SOURCE),
            levels=levels,
            base_url=_optional_str(raw.get("base_url")),
            model=_optional_str(raw.get("model")),
            effort=_optional_str(raw.get("effort")),
            measured_at=_optional_str(raw.get("measured_at")),
            notes=_optional_str(raw.get("notes")),
            request=raw.get("request") if isinstance(raw.get("request"), dict) else None,
        )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _interpolate(points: list[tuple[int, float]], value: float) -> float:
    if value <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            ratio = (value - x0) / (x1 - x0)
            return float(y0 + (y1 - y0) * ratio)
    return float(points[-1][1])


def _require_number(obj: dict, key: str, where: str) -> float:
    if key not in obj:
        raise ConfigError(f"{where} 缺少必填字段: {key}")
    value = obj[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是数字")
    return float(value)


def _optional_number(obj: dict, key: str, where: str) -> float | None:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{where}.{key} 必须是数字")
    return float(value)


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None
