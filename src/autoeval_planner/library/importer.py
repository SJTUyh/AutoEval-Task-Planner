"""把已有测评产物导入数据集库。

支持两种来源：

1. AISBench run 目录（`predictions/<model>/<dataset>.jsonl`）；
2. Adaptor 的统一结果 `unified_result.json`（harbor_agent）。

通用自定义入库入口保留为 `Library.add_cases`。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..errors import LibraryError
from .db import CaseRecord, Library

TIMESTAMP_DIR = re.compile(r"^\d{8}_\d{6}$")

AIS_BENCH_RUN = "ais_bench_run"
AIS_BENCH_PREDICTIONS = "ais_bench_predictions"
HARBOR_RUN = "harbor_run"
UNIFIED_RESULT = "unified_result"


@dataclass(frozen=True)
class SourceInfo:
    kind: str
    path: Path
    run_id: str


@dataclass
class ImportedDataset:
    dataset_key: str
    case_count: int
    input_tokens_total: int
    output_tokens_total: int
    alias: str | None = None
    error_cases: int = 0

    def as_dict(self) -> dict:
        return {
            "dataset_key": self.dataset_key,
            "case_count": self.case_count,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "alias": self.alias,
            "error_cases": self.error_cases,
        }


@dataclass
class ImportSummary:
    run_id: str
    source_type: str
    source_path: str
    dry_run: bool
    datasets: list[ImportedDataset] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "dry_run": self.dry_run,
            "datasets": [item.as_dict() for item in self.datasets],
            "warnings": list(self.warnings),
        }


def detect_source(source: str | Path) -> SourceInfo:
    path = Path(source).expanduser()
    if not path.exists():
        raise LibraryError(f"导入来源不存在: {path}")

    if path.is_file():
        if path.suffix == ".jsonl":
            return SourceInfo(AIS_BENCH_PREDICTIONS, path, _run_id_from_predictions(path))
        if path.suffix == ".json":
            payload = _read_json(path)
            if isinstance(payload.get("cases"), list):
                return SourceInfo(UNIFIED_RESULT, path, _run_id_from_dir(path.parent))
            raise LibraryError(f"{path} 不是 unified_result.json（缺少 cases 数组）")
        raise LibraryError(f"不支持的导入文件类型: {path.suffix}（仅支持 .jsonl / .json）")

    if (path / "predictions").is_dir():
        return SourceInfo(AIS_BENCH_RUN, path, _run_id_from_dir(path))
    if any(path.glob("results/*/*/details")):
        return SourceInfo(HARBOR_RUN, path, _run_id_from_dir(path))

    candidates = sorted(
        (item for item in path.iterdir() if item.is_dir() and TIMESTAMP_DIR.match(item.name)),
        key=lambda item: item.name,
    )
    for candidate in reversed(candidates):
        if (candidate / "predictions").is_dir():
            return SourceInfo(AIS_BENCH_RUN, candidate, _run_id_from_dir(candidate))
        if any(candidate.glob("results/*/*/details")):
            return SourceInfo(HARBOR_RUN, candidate, _run_id_from_dir(candidate))

    if (path / "unified_result.json").is_file():
        return SourceInfo(UNIFIED_RESULT, path / "unified_result.json", _run_id_from_dir(path))

    raise LibraryError(
        f"{path} 既不是 AISBench run 目录（缺少 predictions/），也不是 unified_result.json 所在目录"
    )


def import_source(
    lib: Library,
    source: str | Path,
    *,
    task_type: str,
    alias: str | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    dataset_key: str | None = None,
    observed_concurrency: int | None = None,
    decode_tps: float | None = None,
    dry_run: bool = False,
) -> ImportSummary:
    info = detect_source(source)
    summary = ImportSummary(
        run_id=run_id or info.run_id,
        source_type=info.kind,
        source_path=str(info.path),
        dry_run=dry_run,
    )

    if info.kind == AIS_BENCH_RUN:
        datasets = _read_ais_bench_run(
            info.path, summary, observed_concurrency, decode_tps
        )
    elif info.kind == AIS_BENCH_PREDICTIONS:
        datasets = _read_predictions_file(
            info.path, summary, observed_concurrency, decode_tps
        )
    elif info.kind == HARBOR_RUN:
        datasets = _read_harbor_run(
            info.path, summary, observed_concurrency, decode_tps
        )
    else:
        datasets = _read_unified_result(
            info.path, summary, alias, observed_concurrency, decode_tps
        )

    if not datasets:
        raise LibraryError(f"{info.path} 中未解析到任何 case")
    if dataset_key:
        if len(datasets) > 1:
            raise LibraryError("指定 --dataset-key 时，导入来源只能包含一个数据集")
        datasets = {dataset_key: next(iter(datasets.values()))}
    if alias and len(datasets) > 1:
        raise LibraryError(
            f"{info.path} 包含多个数据集（{', '.join(sorted(datasets))}），"
            "请先拆分或逐个目录导入后再指定 --alias"
        )

    summary.run_id = summary.run_id.strip() or "run"
    imported_at = datetime.now().astimezone().isoformat(timespec="seconds")

    for dataset_key, cases in datasets.items():
        dataset_alias = alias or dataset_key
        input_total = sum(case.input_tokens for case in cases)
        output_total = sum(case.output_tokens for case in cases)
        error_cases = sum(1 for case in cases if case.status != "success")
        summary.datasets.append(
            ImportedDataset(
                dataset_key=dataset_key,
                case_count=len(cases),
                input_tokens_total=input_total,
                output_tokens_total=output_total,
                alias=dataset_alias,
                error_cases=error_cases,
            )
        )
        if dry_run:
            continue
        lib.upsert_run(
            run_id=summary.run_id,
            source_type=summary.source_type,
            source_uri=str(info.path),
            model_name=model_name,
            task_type=task_type,
            observed_concurrency=observed_concurrency,
            imported_at=imported_at,
            notes=None,
        )
        lib.upsert_dataset(
            dataset_key=dataset_key,
            task_type=task_type,
            display_name=dataset_key,
            created_at=imported_at,
        )
        lib.add_alias(dataset_alias, task_type, dataset_key)
        if dataset_alias != dataset_key:
            lib.add_alias(dataset_key, task_type, dataset_key)
        lib.add_cases(summary.run_id, dataset_key, cases)

    return summary


# -- AISBench -------------------------------------------------------------


def _read_ais_bench_run(
    run_dir: Path,
    summary: ImportSummary,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> dict[str, list[CaseRecord]]:
    files = sorted(run_dir.glob("predictions/*/*.jsonl"))
    if not files:
        files = sorted(run_dir.glob("predictions/*.jsonl"))
    if not files:
        raise LibraryError(f"{run_dir} 下未找到 predictions/<model>/*.jsonl")

    datasets: dict[str, list[CaseRecord]] = {}
    for path in files:
        parsed = _parse_predictions_file(
            path, summary, observed_concurrency, decode_tps
        )
        for dataset_key, cases in parsed.items():
            datasets.setdefault(dataset_key, []).extend(cases)
    return datasets


def _read_predictions_file(
    path: Path,
    summary: ImportSummary,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> dict[str, list[CaseRecord]]:
    return _parse_predictions_file(path, summary, observed_concurrency, decode_tps)


def _parse_predictions_file(
    path: Path,
    summary: ImportSummary,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> dict[str, list[CaseRecord]]:
    by_dataset: dict[str, list[CaseRecord]] = {}
    skipped = 0
    try:
        # 只按 \n 切分：case 内容里可能含 U+2028/U+2029，str.splitlines() 会误切。
        lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    except OSError as exc:
        raise LibraryError(f"读取 {path} 失败: {exc}") from exc

    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict):
            skipped += 1
            continue

        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        if input_tokens is None and output_tokens is None:
            skipped += 1
            continue

        dataset_key = str(row.get("data_abbr") or path.stem)
        duration = _duration_sec(row.get("start_time"), row.get("end_time"))
        overhead = _overhead_sec(
            duration, int(output_tokens or 0), observed_concurrency, decode_tps
        )
        case_id = str(row["id"]) if row.get("id") is not None else f"case_{index}"
        status = "success" if row.get("success", True) else "failed"

        by_dataset.setdefault(dataset_key, []).append(
            CaseRecord(
                case_id=case_id,
                attempt=1,
                input_tokens=int(input_tokens or 0),
                output_tokens=int(output_tokens or 0),
                duration_sec=duration,
                overhead_sec=overhead,
                started_at=_iso_or_none(row.get("start_time")),
                finished_at=_iso_or_none(row.get("end_time")),
                status=status,
            )
        )

    if skipped:
        summary.warnings.append(f"{path.name}: 跳过 {skipped} 行无效/缺少 token 的记录")
    return by_dataset


# -- Harbor ---------------------------------------------------------------


def _read_harbor_run(
    run_dir: Path,
    summary: ImportSummary,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> dict[str, list[CaseRecord]]:
    details_dirs = sorted(run_dir.glob("results/*/*/details"))
    if not details_dirs:
        raise LibraryError(f"{run_dir} 下未找到 results/<model>/<dataset>/details")

    datasets: dict[str, list[CaseRecord]] = {}
    for details in details_dirs:
        dataset_key = details.parent.name
        trial_dirs = sorted(item for item in details.iterdir() if item.is_dir())
        attempts: dict[str, int] = {}
        cases: list[CaseRecord] = []
        missing = 0
        for trial_dir in trial_dirs:
            result_path = trial_dir / "result.json"
            if not result_path.is_file():
                missing += 1
                continue
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError):
                missing += 1
                continue
            if not isinstance(payload, dict):
                missing += 1
                continue

            case_id = str(
                payload.get("task_name") or payload.get("task_id") or trial_dir.name
            )
            attempt = attempts.get(case_id, 0) + 1
            attempts[case_id] = attempt

            input_tokens, output_tokens = _harbor_tokens(payload)
            duration = _duration_sec(payload.get("started_at"), payload.get("finished_at"))
            cases.append(
                CaseRecord(
                    case_id=case_id,
                    attempt=attempt,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration_sec=duration,
                    overhead_sec=_overhead_sec(
                        duration, output_tokens, observed_concurrency, decode_tps
                    ),
                    started_at=_iso_or_none(payload.get("started_at")),
                    finished_at=_iso_or_none(payload.get("finished_at")),
                    status=_harbor_status(payload.get("exception_info")),
                )
            )

        if missing:
            summary.warnings.append(
                f"{details}: 跳过 {missing} 个缺少或损坏 result.json 的 trial 目录"
            )
        datasets[dataset_key] = cases
    return datasets


def _harbor_tokens(payload: dict) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    nodes = [payload.get("agent_result")]
    nodes.extend(
        step.get("agent_result")
        for step in (payload.get("step_results") or [])
        if isinstance(step, dict)
    )
    for node in nodes:
        if not isinstance(node, dict):
            continue
        input_tokens += int(node.get("n_input_tokens") or 0)
        output_tokens += int(node.get("n_output_tokens") or 0)
    return input_tokens, output_tokens


def _harbor_status(exception_info: object) -> str:
    if not exception_info:
        return "success"
    if isinstance(exception_info, dict):
        exception_type = str(exception_info.get("exception_type") or "")
        if "Cancelled" in exception_type:
            return "other"
    return "failed"


# -- Unified result -------------------------------------------------------


def _read_unified_result(
    path: Path,
    summary: ImportSummary,
    alias: str | None,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> dict[str, list[CaseRecord]]:
    payload = _read_json(path)
    dataset_key = str(
        alias
        or payload.get("dataset_abbr")
        or payload.get("dataset_key")
        or path.parent.name
    )
    cases: list[CaseRecord] = []
    for index, item in enumerate(payload.get("cases") or []):
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or item.get("trial_name") or f"case_{index}")
        duration = item.get("duration_sec")
        cases.append(
            CaseRecord(
                case_id=case_id,
                attempt=int(item.get("attempt") or 1),
                input_tokens=int(item.get("input_tokens") or 0),
                output_tokens=int(item.get("output_tokens") or 0),
                duration_sec=float(duration) if duration is not None else None,
                overhead_sec=_overhead_sec(
                    float(duration) if duration is not None else None,
                    int(item.get("output_tokens") or 0),
                    observed_concurrency,
                    decode_tps,
                ),
                started_at=_iso_or_none(item.get("started_at")),
                finished_at=_iso_or_none(item.get("finished_at")),
                status=str(item.get("status") or "other"),
            )
        )
    if not cases:
        summary.warnings.append(f"{path} 的 cases 为空")
    return {dataset_key: cases}


# -- 工具函数 -------------------------------------------------------------


def _run_id_from_dir(path: Path) -> str:
    name = path.name
    parent = path.parent.name
    if TIMESTAMP_DIR.match(name) and parent:
        return f"{parent}_{name}"
    return name or "run"


def _run_id_from_predictions(path: Path) -> str:
    for parent in path.parents:
        if TIMESTAMP_DIR.match(parent.name):
            return f"{parent.parent.name}_{parent.name}"
    return path.parent.name or "run"


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise LibraryError(f"{path} 不是合法 JSON: {exc}") from exc
    except OSError as exc:
        raise LibraryError(f"读取 {path} 失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise LibraryError(f"{path} 顶层必须是 JSON 对象")
    return payload


def _iso_or_none(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _duration_sec(start: object, end: object) -> float | None:
    start_dt = _parse_time(start)
    end_dt = _parse_time(end)
    if start_dt is None or end_dt is None:
        return None
    return max(0.0, (end_dt - start_dt).total_seconds())


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _overhead_sec(
    duration: float | None,
    output_tokens: int,
    observed_concurrency: int | None,
    decode_tps: float | None,
) -> float:
    """非 LLM 固定开销 = 实测耗时 - 该 run 并发下的纯 decode 时间。"""
    if duration is None or not observed_concurrency or not decode_tps:
        return 0.0
    if observed_concurrency <= 0 or decode_tps <= 0:
        return 0.0
    service_sec = output_tokens * observed_concurrency / decode_tps
    return max(0.0, duration - service_sec)
