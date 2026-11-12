"""数据集画像：历史 case 分布、bootstrap 抽样与缺失数据估算。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..config import Estimate

QUALITY_HISTORY = "history"
QUALITY_ESTIMATE = "estimate"
QUALITY_HEURISTIC = "heuristic"


@dataclass(frozen=True)
class CaseSpec:
    """一个 case 的服务侧工作量。"""

    input_tokens: int
    output_tokens: int
    overhead_sec: float = 0.0


@dataclass
class DatasetProfile:
    dataset_key: str
    task_type: str
    cases: list[CaseSpec]
    quality: str = QUALITY_HISTORY
    source_runs: list[str] = field(default_factory=list)

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def totals(self) -> tuple[int, int]:
        return (
            sum(case.input_tokens for case in self.cases),
            sum(case.output_tokens for case in self.cases),
        )

    def summary(self) -> dict:
        input_total, output_total = self.totals()
        return {
            "dataset_key": self.dataset_key,
            "task_type": self.task_type,
            "case_count": self.case_count,
            "input_tokens_total": input_total,
            "output_tokens_total": output_total,
            "input_tokens_median": _median([c.input_tokens for c in self.cases]),
            "output_tokens_median": _median([c.output_tokens for c in self.cases]),
            "overhead_sec_median": _median([c.overhead_sec for c in self.cases]),
            "data_quality": self.quality,
            "source_runs": list(self.source_runs),
        }

    def sample(self, count: int, rng: random.Random, overhead_cv: float = 0.0) -> list[CaseSpec]:
        """有放回 bootstrap；overhead_cv > 0 时对 overhead 施加对数正态抖动。"""
        if not self.cases or count <= 0:
            return []
        picks = rng.choices(self.cases, k=count)
        if overhead_cv <= 0:
            return list(picks)
        sampled: list[CaseSpec] = []
        for case in picks:
            overhead = case.overhead_sec
            if overhead > 0:
                overhead *= math.exp(rng.gauss(0.0, overhead_cv))
            sampled.append(
                CaseSpec(
                    input_tokens=case.input_tokens,
                    output_tokens=case.output_tokens,
                    overhead_sec=overhead,
                )
            )
        return sampled


def profile_from_rows(
    dataset_key: str,
    task_type: str,
    rows: list[dict],
    source_runs: list[str],
) -> DatasetProfile:
    """按 case_id 聚合历史观测：多个 run / 重试只贡献同一个 case 的分布，
    避免把同一道题重复算成多个 case。"""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        case_id = str(row.get("case_id"))
        if case_id not in groups:
            groups[case_id] = []
            order.append(case_id)
        groups[case_id].append(row)

    cases: list[CaseSpec] = []
    for case_id in order:
        observations = groups[case_id]
        successes = [
            item for item in observations if (item.get("status") or "success") == "success"
        ]
        chosen = successes or observations
        cases.append(
            CaseSpec(
                input_tokens=max(
                    int(round(_median([int(item.get("input_tokens") or 0) for item in chosen]))),
                    0,
                ),
                output_tokens=max(
                    int(round(_median([int(item.get("output_tokens") or 0) for item in chosen]))),
                    0,
                ),
                overhead_sec=max(
                    _median([float(item.get("overhead_sec") or 0.0) for item in chosen]),
                    0.0,
                ),
            )
        )
    return DatasetProfile(
        dataset_key=dataset_key,
        task_type=task_type,
        cases=cases,
        quality=QUALITY_HISTORY,
        source_runs=source_runs,
    )


def profile_from_estimate(
    dataset_key: str,
    task_type: str,
    estimate: Estimate,
    seed: int,
) -> DatasetProfile:
    rng = random.Random(seed)
    input_median = float(estimate.input_tokens_median)
    output_median = float(estimate.output_tokens_median)
    input_sigma = _sigma(input_median, float(estimate.input_tokens_p90))
    output_sigma = _sigma(output_median, float(estimate.output_tokens_p90))

    cases = [
        CaseSpec(
            input_tokens=_lognormal(rng, input_median, input_sigma),
            output_tokens=_lognormal(rng, output_median, output_sigma),
            overhead_sec=float(estimate.overhead_sec),
        )
        for _ in range(estimate.case_count)
    ]
    return DatasetProfile(
        dataset_key=dataset_key,
        task_type=task_type,
        cases=cases,
        quality=QUALITY_ESTIMATE,
        source_runs=[],
    )


def heuristic_profile(dataset_key: str, task_type: str, seed: int) -> DatasetProfile:
    if task_type.endswith("harbor_agent"):
        estimate = Estimate(
            case_count=500,
            input_tokens_median=40000,
            input_tokens_p90=120000,
            output_tokens_median=6000,
            output_tokens_p90=25000,
            overhead_sec=300.0,
        )
    else:
        estimate = Estimate(
            case_count=1000,
            input_tokens_median=250,
            input_tokens_p90=1000,
            output_tokens_median=7000,
            output_tokens_p90=25000,
            overhead_sec=0.0,
        )
    profile = profile_from_estimate(dataset_key, task_type, estimate, seed)
    profile.quality = QUALITY_HEURISTIC
    return profile


def reference_profile(
    dataset_key: str,
    task_type: str,
    references: list[DatasetProfile],
    seed: int,
) -> DatasetProfile | None:
    """用同 task_type 其它历史数据集的中位数构造参考分布。"""
    same_type = [p for p in references if p.task_type == task_type and p.cases]
    if not same_type:
        return None
    input_median = _median([_median([c.input_tokens for c in p.cases]) for p in same_type])
    output_median = _median([_median([c.output_tokens for c in p.cases]) for p in same_type])
    overhead_median = _median([_median([c.overhead_sec for c in p.cases]) for p in same_type])
    estimate = Estimate(
        case_count=1000,
        input_tokens_median=max(input_median, 1),
        input_tokens_p90=max(input_median * 4, input_median + 1),
        output_tokens_median=max(output_median, 1),
        output_tokens_p90=max(output_median * 4, output_median + 1),
        overhead_sec=max(overhead_median, 0.0),
    )
    profile = profile_from_estimate(dataset_key, task_type, estimate, seed)
    profile.quality = QUALITY_HEURISTIC
    return profile


def _sigma(median: float, p90: float) -> float:
    if median <= 0 or p90 <= median:
        return 0.15
    return (math.log(p90) - math.log(median)) / 1.2816


def _lognormal(rng: random.Random, median: float, sigma: float) -> int:
    value = median * math.exp(rng.gauss(0.0, sigma))
    return max(1, int(round(value)))


def _median(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
