"""SQLite 数据集库：schema、写入与查询。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from ..errors import LibraryError

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    model_name TEXT,
    task_type TEXT NOT NULL,
    observed_concurrency INTEGER,
    imported_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_key TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_aliases (
    alias TEXT NOT NULL,
    task_type TEXT NOT NULL,
    dataset_key TEXT NOT NULL,
    PRIMARY KEY (alias, task_type)
);

CREATE TABLE IF NOT EXISTS cases (
    run_id TEXT NOT NULL,
    dataset_key TEXT NOT NULL,
    case_id TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_sec REAL,
    overhead_sec REAL,
    started_at TEXT,
    finished_at TEXT,
    status TEXT,
    PRIMARY KEY (run_id, dataset_key, case_id, attempt)
);

CREATE INDEX IF NOT EXISTS idx_cases_dataset ON cases (dataset_key);

CREATE TABLE IF NOT EXISTS servers (
    name TEXT PRIMARY KEY,
    cpu_cores REAL NOT NULL,
    mem_mb INTEGER NOT NULL,
    disk_gb REAL,
    network_available INTEGER,
    network_pool TEXT,
    docker_networks INTEGER,
    captured_at TEXT NOT NULL,
    source TEXT NOT NULL,
    raw TEXT
);

CREATE TABLE IF NOT EXISTS tool_resource_defaults (
    tool TEXT NOT NULL,
    subcommand TEXT NOT NULL,
    cpu_base REAL NOT NULL DEFAULT 0,
    mem_mb_base INTEGER NOT NULL DEFAULT 0,
    cpu_per_case REAL NOT NULL DEFAULT 0,
    mem_mb_per_case INTEGER NOT NULL DEFAULT 0,
    disk_gb_per_case REAL NOT NULL DEFAULT 0,
    networks_per_case INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tool, subcommand)
);

CREATE TABLE IF NOT EXISTS dataset_resources (
    dataset_key TEXT NOT NULL,
    task_type TEXT NOT NULL,
    cpu_base REAL NOT NULL DEFAULT 0,
    mem_mb_base INTEGER NOT NULL DEFAULT 0,
    cpu_per_case REAL NOT NULL DEFAULT 0,
    mem_mb_per_case INTEGER NOT NULL DEFAULT 0,
    disk_gb_per_case REAL NOT NULL DEFAULT 0,
    networks_per_case INTEGER NOT NULL DEFAULT 0,
    max_cpu_per_case REAL,
    max_mem_mb_per_case INTEGER,
    task_count INTEGER,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    detail TEXT,
    PRIMARY KEY (dataset_key, task_type)
);
"""


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    attempt: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    duration_sec: float | None = None
    overhead_sec: float = 0.0
    started_at: str | None = None
    finished_at: str | None = None
    status: str = "success"


@dataclass(frozen=True)
class DatasetSummary:
    dataset_key: str
    task_type: str
    case_count: int
    unique_case_count: int
    input_tokens_total: int
    output_tokens_total: int
    run_count: int
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ServerSnapshot:
    name: str
    cpu_cores: float
    mem_mb: int
    disk_gb: float | None
    network_available: int | None
    network_pool: str | None
    docker_networks: int | None
    captured_at: str
    source: str
    raw: str | None = None


@dataclass(frozen=True)
class ResourceProfile:
    cpu_base: float
    mem_mb_base: int
    cpu_per_case: float
    mem_mb_per_case: int
    networks_per_case: int
    source: str
    disk_gb_per_case: float = 0.0
    max_cpu_per_case: float | None = None
    max_mem_mb_per_case: int | None = None
    task_count: int | None = None
    detail: str | None = None


class Library:
    """数据集库的薄封装；调用方负责 close()。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
        except (OSError, sqlite3.Error) as exc:
            raise LibraryError(f"无法打开数据集库 {self.path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.executescript(SCHEMA)
            self._ensure_column(
                "tool_resource_defaults", "disk_gb_per_case", "REAL NOT NULL DEFAULT 0"
            )
            self._ensure_column(
                "dataset_resources", "disk_gb_per_case", "REAL NOT NULL DEFAULT 0"
            )
        except sqlite3.Error as exc:
            raise LibraryError(f"初始化数据集库失败: {exc}") from exc

    # -- 生命周期 ---------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Library":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def commit(self) -> None:
        self._conn.commit()

    # -- 写入 -------------------------------------------------------------
    def upsert_run(
        self,
        *,
        run_id: str,
        source_type: str,
        source_uri: str,
        model_name: str | None,
        task_type: str,
        observed_concurrency: int | None,
        imported_at: str,
        notes: str | None = None,
    ) -> None:
        self._execute(
            "INSERT OR REPLACE INTO runs "
            "(run_id, source_type, source_uri, model_name, task_type, "
            " observed_concurrency, imported_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                source_type,
                source_uri,
                model_name,
                task_type,
                observed_concurrency,
                imported_at,
                notes,
            ),
        )

    def upsert_dataset(
        self,
        *,
        dataset_key: str,
        task_type: str,
        display_name: str | None,
        created_at: str,
    ) -> None:
        self._execute(
            "INSERT INTO datasets (dataset_key, task_type, display_name, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(dataset_key) DO UPDATE SET "
            "task_type=excluded.task_type, "
            "display_name=COALESCE(excluded.display_name, datasets.display_name)",
            (dataset_key, task_type, display_name, created_at),
        )

    def add_alias(self, alias: str, task_type: str, dataset_key: str) -> None:
        self._execute(
            "INSERT OR REPLACE INTO dataset_aliases (alias, task_type, dataset_key) "
            "VALUES (?, ?, ?)",
            (alias, task_type, dataset_key),
        )

    def add_cases(
        self,
        run_id: str,
        dataset_key: str,
        cases: Iterable[CaseRecord],
    ) -> int:
        """通用入库 API（供后续接入其它测评工具使用），按主键幂等覆盖。"""
        rows = [
            (
                run_id,
                dataset_key,
                case.case_id,
                case.attempt,
                case.input_tokens,
                case.output_tokens,
                case.duration_sec,
                case.overhead_sec,
                case.started_at,
                case.finished_at,
                case.status,
            )
            for case in cases
        ]
        if not rows:
            return 0
        try:
            self._conn.executemany(
                "INSERT OR REPLACE INTO cases "
                "(run_id, dataset_key, case_id, attempt, input_tokens, output_tokens, "
                " duration_sec, overhead_sec, started_at, finished_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise LibraryError(f"写入 case 失败: {exc}") from exc
        return len(rows)

    # -- 查询 -------------------------------------------------------------
    def lookup_alias(self, alias: str, task_type: str) -> str | None:
        row = self._query_one(
            "SELECT dataset_key FROM dataset_aliases WHERE alias = ? AND task_type = ?",
            (alias, task_type),
        )
        return str(row["dataset_key"]) if row is not None else None

    def list_datasets(self) -> list[DatasetSummary]:
        rows = self._query(
            "SELECT d.dataset_key AS dataset_key, "
            "       d.task_type AS task_type, "
            "       COUNT(c.case_id) AS case_count, "
            "       COUNT(DISTINCT c.case_id) AS unique_case_count, "
            "       COALESCE(SUM(c.input_tokens), 0) AS input_tokens_total, "
            "       COALESCE(SUM(c.output_tokens), 0) AS output_tokens_total, "
            "       COUNT(DISTINCT c.run_id) AS run_count "
            "FROM datasets d LEFT JOIN cases c ON c.dataset_key = d.dataset_key "
            "GROUP BY d.dataset_key, d.task_type "
            "ORDER BY d.dataset_key"
        )
        alias_rows = self._query("SELECT alias, task_type, dataset_key FROM dataset_aliases")
        aliases: dict[str, list[str]] = {}
        for row in alias_rows:
            aliases.setdefault(str(row["dataset_key"]), []).append(str(row["alias"]))

        return [
            DatasetSummary(
                dataset_key=str(row["dataset_key"]),
                task_type=str(row["task_type"]),
                case_count=int(row["case_count"]),
                unique_case_count=int(row["unique_case_count"]),
                input_tokens_total=int(row["input_tokens_total"]),
                output_tokens_total=int(row["output_tokens_total"]),
                run_count=int(row["run_count"]),
                aliases=tuple(sorted(aliases.get(str(row["dataset_key"]), []))),
            )
            for row in rows
        ]

    def load_cases(self, dataset_key: str) -> list[dict]:
        # 数字型 case_id 按数值排序，保证与 AISBench 的 FIFO 顺序一致；
        # 非数字 id（如 harbor 任务名）按字符串排序。
        rows = self._query(
            "SELECT case_id, attempt, input_tokens, output_tokens, duration_sec, "
            "       overhead_sec, started_at, finished_at, status "
            "FROM cases WHERE dataset_key = ? "
            "ORDER BY CASE WHEN case_id GLOB '[0-9]*' THEN 0 ELSE 1 END, "
            "         CAST(case_id AS INTEGER), case_id, attempt",
            (dataset_key,),
        )
        return [dict(row) for row in rows]

    def dataset_runs(self, dataset_key: str) -> list[str]:
        rows = self._query(
            "SELECT DISTINCT run_id FROM cases WHERE dataset_key = ? ORDER BY run_id",
            (dataset_key,),
        )
        return [str(row["run_id"]) for row in rows]

    def count_cases(self, dataset_key: str) -> int:
        row = self._query_one(
            "SELECT COUNT(*) AS n FROM cases WHERE dataset_key = ?", (dataset_key,)
        )
        return int(row["n"]) if row is not None else 0

    # -- 服务器资源 -------------------------------------------------------
    def upsert_server(self, snapshot: ServerSnapshot) -> None:
        self._execute(
            "INSERT OR REPLACE INTO servers "
            "(name, cpu_cores, mem_mb, disk_gb, network_available, network_pool, "
            " docker_networks, captured_at, source, raw) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.name,
                snapshot.cpu_cores,
                snapshot.mem_mb,
                snapshot.disk_gb,
                snapshot.network_available,
                snapshot.network_pool,
                snapshot.docker_networks,
                snapshot.captured_at,
                snapshot.source,
                snapshot.raw,
            ),
        )

    def get_server(self, name: str) -> ServerSnapshot | None:
        row = self._query_one("SELECT * FROM servers WHERE name = ?", (name,))
        if row is None:
            return None
        return ServerSnapshot(
            name=str(row["name"]),
            cpu_cores=float(row["cpu_cores"]),
            mem_mb=int(row["mem_mb"]),
            disk_gb=float(row["disk_gb"]) if row["disk_gb"] is not None else None,
            network_available=(
                int(row["network_available"]) if row["network_available"] is not None else None
            ),
            network_pool=row["network_pool"],
            docker_networks=(
                int(row["docker_networks"]) if row["docker_networks"] is not None else None
            ),
            captured_at=str(row["captured_at"]),
            source=str(row["source"]),
            raw=row["raw"],
        )

    def list_servers(self) -> list[ServerSnapshot]:
        servers: list[ServerSnapshot] = []
        for row in self._query("SELECT name FROM servers ORDER BY name"):
            snapshot = self.get_server(str(row["name"]))
            if snapshot is not None:
                servers.append(snapshot)
        return servers

    # -- 资源画像 ---------------------------------------------------------
    def upsert_tool_resources(
        self, tool: str, subcommand: str, profile: ResourceProfile
    ) -> None:
        self._execute(
            "INSERT OR REPLACE INTO tool_resource_defaults "
            "(tool, subcommand, cpu_base, mem_mb_base, cpu_per_case, mem_mb_per_case, "
            " disk_gb_per_case, networks_per_case, source, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                tool,
                subcommand,
                profile.cpu_base,
                profile.mem_mb_base,
                profile.cpu_per_case,
                profile.mem_mb_per_case,
                profile.disk_gb_per_case,
                profile.networks_per_case,
                profile.source,
                _now(),
            ),
        )

    def get_tool_resources(self, tool: str, subcommand: str) -> ResourceProfile | None:
        row = self._query_one(
            "SELECT * FROM tool_resource_defaults WHERE tool = ? AND subcommand = ?",
            (tool, subcommand),
        )
        return _tool_profile_from_row(row) if row is not None else None

    def upsert_dataset_resources(
        self, dataset_key: str, task_type: str, profile: ResourceProfile
    ) -> None:
        self._execute(
            "INSERT OR REPLACE INTO dataset_resources "
            "(dataset_key, task_type, cpu_base, mem_mb_base, cpu_per_case, "
            " mem_mb_per_case, disk_gb_per_case, networks_per_case, max_cpu_per_case, "
            " max_mem_mb_per_case, task_count, source, updated_at, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_key,
                task_type,
                profile.cpu_base,
                profile.mem_mb_base,
                profile.cpu_per_case,
                profile.mem_mb_per_case,
                profile.disk_gb_per_case,
                profile.networks_per_case,
                profile.max_cpu_per_case,
                profile.max_mem_mb_per_case,
                profile.task_count,
                profile.source,
                _now(),
                profile.detail,
            ),
        )

    def get_dataset_resources(
        self, dataset_key: str, task_type: str
    ) -> ResourceProfile | None:
        row = self._query_one(
            "SELECT * FROM dataset_resources WHERE dataset_key = ? AND task_type = ?",
            (dataset_key, task_type),
        )
        return _profile_from_row(row) if row is not None else None

    def list_dataset_resources(self) -> list[tuple[str, str, ResourceProfile]]:
        rows = self._query(
            "SELECT * FROM dataset_resources ORDER BY dataset_key, task_type"
        )
        return [
            (str(row["dataset_key"]), str(row["task_type"]), _profile_from_row(row))
            for row in rows
        ]

    # -- 内部 -------------------------------------------------------------
    def _execute(self, sql: str, params: Sequence[object] = ()) -> None:
        try:
            self._conn.execute(sql, params)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise LibraryError(f"数据库写入失败: {exc}") from exc

    def _query(self, sql: str, params: Sequence[object] = ()) -> list[sqlite3.Row]:
        try:
            return list(self._conn.execute(sql, params))
        except sqlite3.Error as exc:
            raise LibraryError(f"数据库查询失败: {exc}") from exc

    def _query_one(self, sql: str, params: Sequence[object] = ()) -> sqlite3.Row | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        columns = {
            str(row["name"]) for row in self._conn.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
            self._conn.commit()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _tool_profile_from_row(row: sqlite3.Row) -> ResourceProfile:
    return ResourceProfile(
        cpu_base=float(row["cpu_base"]),
        mem_mb_base=int(row["mem_mb_base"]),
        cpu_per_case=float(row["cpu_per_case"]),
        mem_mb_per_case=int(row["mem_mb_per_case"]),
        networks_per_case=int(row["networks_per_case"]),
        source=str(row["source"]),
        disk_gb_per_case=float(row["disk_gb_per_case"] or 0),
    )


def _profile_from_row(row: sqlite3.Row) -> ResourceProfile:
    return ResourceProfile(
        cpu_base=float(row["cpu_base"]),
        mem_mb_base=int(row["mem_mb_base"]),
        cpu_per_case=float(row["cpu_per_case"]),
        mem_mb_per_case=int(row["mem_mb_per_case"]),
        networks_per_case=int(row["networks_per_case"]),
        source=str(row["source"]),
        disk_gb_per_case=float(row["disk_gb_per_case"] or 0),
        max_cpu_per_case=(
            float(row["max_cpu_per_case"]) if row["max_cpu_per_case"] is not None else None
        ),
        max_mem_mb_per_case=(
            int(row["max_mem_mb_per_case"])
            if row["max_mem_mb_per_case"] is not None
            else None
        ),
        task_count=int(row["task_count"]) if row["task_count"] is not None else None,
        detail=row["detail"],
    )
