"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import load_taskset
from .errors import ConfigError, PlannerError
from .library.db import Library
from .library.importer import import_source
from .optimize import optimize
from .probe import run_probe
from .resource import (
    collect_via_ssh,
    load_github_tasks,
    load_tasks_dir,
    snapshot_from_file,
    snapshot_from_values,
    upsert_harbor_resources,
)
from .report import build_report, format_summary
from .resources import build_provider
from .taskset import build_task_models, plan_tasks
from .throughput import ThroughputProfile

DEFAULT_DB = Path("data") / "dataset_library.sqlite3"
DEFAULT_PLAN_OUTPUT = "plan.json"
DEFAULT_THROUGHPUT_OUTPUT = "throughput.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoeval-planner",
        description="评估整套测评任务的最优总时长并给出任务编排建议",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="生成测评任务时长规划")
    plan.add_argument("--tasks", required=True, help="taskset JSON 路径")
    plan.add_argument("--db", default=None, help=f"数据集库路径（默认 {DEFAULT_DB}）")
    plan.add_argument("--output", default=DEFAULT_PLAN_OUTPUT, help="规划结果输出路径")
    plan.add_argument("--runs", type=int, default=None, help="覆盖 taskset 中的蒙特卡洛次数")
    plan.add_argument("--seed", type=int, default=None, help="覆盖 taskset 中的随机种子")
    plan.add_argument("--strict", action="store_true", help="缺少历史数据时直接报错")
    plan.add_argument("--json", action="store_true", help="向 stdout 打印完整 JSON")

    probe = subparsers.add_parser("probe", help="探测推理服务的并发-吞吐曲线")
    probe.add_argument("--base-url", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--api-key", default=None)
    probe.add_argument("--api-key-env", default=None, help="从该环境变量读取 API key")
    probe.add_argument("--concurrency", default="1,2,4,8", help="并发档位，逗号分隔")
    probe.add_argument("--requests-per-level", type=int, default=2)
    probe.add_argument("--prompt-tokens", type=int, default=512)
    probe.add_argument("--max-tokens", type=int, default=512)
    probe.add_argument("--effort", default=None, help="reasoning_effort，如 max")
    probe.add_argument("--timeout", type=float, default=300.0)
    probe.add_argument(
        "--endpoint",
        default=None,
        help="完整请求地址（覆盖 base-url 拼接）；Azure 等需要带 query 的场景使用",
    )
    probe.add_argument("--api-key-header", default="Authorization", help="API key 所在请求头")
    probe.add_argument("--api-key-scheme", default="Bearer ", help="API key 前缀，Azure 用空字符串")
    probe.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="追加请求头，可重复",
    )
    probe.add_argument("--body", default=None, help="追加/覆盖请求体的 JSON 对象")
    probe.add_argument("--no-stream", action="store_true", help="使用非流式请求")
    probe.add_argument("--insecure", action="store_true", help="跳过 TLS 证书校验（自签名服务）")
    probe.add_argument("--retries", type=int, default=2, help="429/5xx 重试次数")
    probe.add_argument("--output", default=DEFAULT_THROUGHPUT_OUTPUT)

    dataset = subparsers.add_parser("dataset", help="数据集库管理")
    dataset_sub = dataset.add_subparsers(dest="action", required=True)

    dataset_import = dataset_sub.add_parser("import", help="导入已有测评产物")
    dataset_import.add_argument("--source", required=True, help="AISBench run 目录或 unified_result.json")
    dataset_import.add_argument("--task-type", default="ais_bench/llm_accuracy")
    dataset_import.add_argument("--alias", default=None, help="数据集别名（对应配置中的 datasets 项）")
    dataset_import.add_argument("--dataset-key", default=None, help="强制使用的数据集键（多个 run 合并到同一数据集时使用）")
    dataset_import.add_argument("--model-name", default=None)
    dataset_import.add_argument("--run-id", default=None)
    dataset_import.add_argument("--observed-concurrency", type=int, default=None)
    dataset_import.add_argument("--decode-tps", type=float, default=None)
    dataset_import.add_argument("--db", default=None)
    dataset_import.add_argument("--dry-run", action="store_true")

    dataset_list = dataset_sub.add_parser("list", help="列出数据集库内容")
    dataset_list.add_argument("--db", default=None)

    dataset_add = dataset_sub.add_parser("add", help="预留：通用自定义入库")
    dataset_add.add_argument("--db", default=None)

    resource = subparsers.add_parser("resource", help="服务器与数据集资源管理")
    resource_sub = resource.add_subparsers(dest="action", required=True)

    res_probe = resource_sub.add_parser("probe", help="通过 SSH 采集服务器资源快照")
    res_probe.add_argument("--name", required=True)
    res_probe.add_argument("--ssh-host", required=True)
    res_probe.add_argument("--ssh-user", required=True)
    res_probe.add_argument("--ssh-port", type=int, default=22)
    res_probe.add_argument("--ssh-password", default=None)
    res_probe.add_argument("--ssh-key", default=None)
    res_probe.add_argument("--timeout", type=float, default=20.0)
    res_probe.add_argument("--db", default=None)
    res_probe.add_argument("--output", default=None, help="可选：同时导出 JSON 快照")

    res_set = resource_sub.add_parser("set", help="手工写入服务器资源快照")
    res_set.add_argument("--name", required=True)
    res_set.add_argument("--cpu", type=float, required=True)
    res_set.add_argument("--mem-mb", type=int, required=True)
    res_set.add_argument("--disk-gb", type=float, default=None)
    res_set.add_argument("--network-available", type=int, default=None)
    res_set.add_argument("--network-pool", default=None)
    res_set.add_argument("--docker-networks", type=int, default=None)
    res_set.add_argument("--db", default=None)

    res_import = resource_sub.add_parser("import", help="从 JSON 导入服务器资源快照")
    res_import.add_argument("--file", required=True)
    res_import.add_argument("--db", default=None)

    res_list = resource_sub.add_parser("list", help="列出服务器与数据集资源")
    res_list.add_argument("--db", default=None)

    res_show = resource_sub.add_parser("show", help="查看数据集资源画像")
    res_show.add_argument("--dataset-key", required=True)
    res_show.add_argument("--task-type", default=None)
    res_show.add_argument("--db", default=None)

    res_harbor = resource_sub.add_parser(
        "import-harbor", help="从 Harbor task.toml 导入单 trial 资源"
    )
    res_harbor.add_argument("--dataset-key", required=True)
    res_harbor.add_argument("--task-type", default="ais_bench/harbor_agent")
    group = res_harbor.add_mutually_exclusive_group(required=True)
    group.add_argument("--tasks-dir", default=None)
    group.add_argument("--github", default=None, help="owner/repo@ref")
    res_harbor.add_argument("--db", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "probe":
            return _cmd_probe(args)
        if args.command == "resource":
            return _cmd_resource(args)
        return _cmd_dataset(args)
    except PlannerError as exc:
        print(f"[error] {exc.message}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("[error] 已中断", file=sys.stderr)
        return 130


# -- plan ------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> int:
    taskset = load_taskset(args.tasks)
    simulation = taskset.simulation
    if args.runs is not None:
        if args.runs < 1:
            raise ConfigError("--runs 必须大于等于 1")
        simulation = replace(simulation, runs=args.runs)
    if args.seed is not None:
        simulation = replace(simulation, seed=args.seed)
    taskset = replace(taskset, simulation=simulation)

    throughput = _load_throughput(taskset)
    db_path = Path(args.db) if args.db else DEFAULT_DB

    with Library(db_path) as library:
        planned = plan_tasks(taskset, library)
        models, warnings = build_task_models(
            planned, library, seed=simulation.seed, strict=args.strict
        )
        provider = build_provider(taskset.resources, library, models)

    result = optimize(
        models,
        throughput,
        provider,
        runs=simulation.runs,
        seed=simulation.seed,
        throughput_cv=simulation.throughput_cv,
    )
    report = build_report(
        taskset=taskset,
        throughput=throughput,
        tasks=models,
        result=result,
        warnings=warnings,
        provider=provider,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_summary(report))
        print(f"  结果文件: {output.resolve()}")
    return 0


def _load_throughput(taskset) -> ThroughputProfile:
    spec = taskset.throughput
    if spec.mode == "inline":
        assert spec.inline is not None
        return ThroughputProfile.from_dict(spec.inline, source="<inline>")
    path = Path(str(spec.path)).expanduser()
    if not path.is_absolute():
        path = taskset.base_dir / path
    return ThroughputProfile.load(path)


# -- probe -----------------------------------------------------------------


def _cmd_probe(args: argparse.Namespace) -> int:
    api_key = args.api_key or (
        os.environ.get(args.api_key_env) if args.api_key_env else None
    )
    if not api_key:
        print(
            "[warn] 未提供 API key：将不携带鉴权头（自建服务可忽略）",
            file=sys.stderr,
        )

    levels = _parse_levels(args.concurrency)
    profile = run_probe(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        concurrency_levels=levels,
        requests_per_level=args.requests_per_level,
        prompt_tokens=args.prompt_tokens,
        max_tokens=args.max_tokens,
        effort=args.effort,
        timeout=args.timeout,
        endpoint=args.endpoint,
        api_key_header=args.api_key_header,
        api_key_scheme=args.api_key_scheme,
        headers=_parse_headers(args.header),
        stream=not args.no_stream,
        body_overrides=_parse_body(args.body),
        insecure=args.insecure,
        retries=args.retries,
    )
    output = profile.save(args.output)
    print("探测结果")
    print(f"  {'并发':>4}  {'decode tok/s':>13}  {'ttft(s)':>8}  {'错误率':>6}")
    for level in profile.levels:
        ttft = f"{level.ttft_sec:.2f}" if level.ttft_sec is not None else "-"
        print(
            f"  {level.concurrency:>4}  {level.decode_tps:>13.1f}  {ttft:>8}  "
            f"{level.error_rate:>6.1%}"
        )
    print(f"  峰值吞吐: {profile.peak_decode_tps:.1f} tok/s")
    print(f"  最大可用并发: {profile.max_concurrency}")
    if any(level.usage_estimated for level in profile.levels):
        print("  [warn] 部分档位缺少 usage，输出 token 为估算值", file=sys.stderr)
    print(f"  结果文件: {output.resolve()}")
    return 0


def _parse_headers(items: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ConfigError(f"--header 需要 NAME:VALUE 形式，实际为 {item!r}")
        name, value = item.split(":", 1)
        name = name.strip()
        if not name:
            raise ConfigError(f"--header 的头部名称不能为空: {item!r}")
        headers[name] = value.strip()
    return headers


def _parse_body(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"--body 不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("--body 必须是 JSON 对象")
    return payload


def _parse_levels(raw: str) -> list[int]:
    try:
        levels = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ConfigError(f"--concurrency 必须是逗号分隔的整数: {raw!r}") from exc
    if not levels:
        raise ConfigError("--concurrency 不能为空")
    return levels


# -- resource --------------------------------------------------------------


def _cmd_resource(args: argparse.Namespace) -> int:
    db_path = Path(args.db) if args.db else DEFAULT_DB
    with Library(db_path) as library:
        if args.action == "probe":
            snapshot = collect_via_ssh(
                name=args.name,
                host=args.ssh_host,
                user=args.ssh_user,
                port=args.ssh_port,
                password=args.ssh_password,
                key_path=args.ssh_key,
                timeout=args.timeout,
            )
            library.upsert_server(snapshot)
            if args.output:
                output = Path(args.output).expanduser()
                output.write_text(
                    json.dumps(_snapshot_dict(snapshot), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            _print_server(snapshot)
            return 0

        if args.action == "set":
            snapshot = snapshot_from_values(
                name=args.name,
                cpu_cores=args.cpu,
                mem_mb=args.mem_mb,
                disk_gb=args.disk_gb,
                network_available=args.network_available,
                network_pool=args.network_pool,
                docker_networks=args.docker_networks,
            )
            library.upsert_server(snapshot)
            _print_server(snapshot)
            return 0

        if args.action == "import":
            snapshot = snapshot_from_file(args.file)
            library.upsert_server(snapshot)
            _print_server(snapshot)
            return 0

        if args.action == "list":
            servers = library.list_servers()
            print(f"服务器快照 ({len(servers)})")
            for snapshot in servers:
                print(
                    f"  {snapshot.name}: {snapshot.cpu_cores:g} 核 / {snapshot.mem_mb} MB / "
                    f"磁盘 {snapshot.disk_gb or '-'} GB / 网段 {snapshot.network_available or '-'} / "
                    f"{snapshot.captured_at} ({snapshot.source})"
                )
            resources = library.list_dataset_resources()
            print(f"数据集资源画像 ({len(resources)})")
            for dataset_key, task_type, profile in resources:
                print(
                    f"  {dataset_key} [{task_type}]: "
                    f"base {profile.cpu_base:g}C/{profile.mem_mb_base}MB + "
                    f"每 case {profile.cpu_per_case:g}C/{profile.mem_mb_per_case}MB/"
                    f"{profile.disk_gb_per_case:g}GB/{profile.networks_per_case}net "
                    f"({profile.source})"
                )
            return 0

        if args.action == "show":
            if args.task_type:
                entries = [
                    (args.dataset_key, args.task_type,
                     library.get_dataset_resources(args.dataset_key, args.task_type))
                ]
            else:
                entries = [
                    (key, task_type, profile)
                    for key, task_type, profile in library.list_dataset_resources()
                    if key == args.dataset_key
                ]
            entries = [entry for entry in entries if entry[2] is not None]
            if not entries:
                print(f"未找到 {args.dataset_key} 的资源画像", file=sys.stderr)
                return 4
            for key, task_type, profile in entries:
                print(
                    json.dumps(
                        {
                            "dataset_key": key,
                            "task_type": task_type,
                            "cpu_base": profile.cpu_base,
                            "mem_mb_base": profile.mem_mb_base,
                            "cpu_per_case": profile.cpu_per_case,
                            "mem_mb_per_case": profile.mem_mb_per_case,
                            "disk_gb_per_case": profile.disk_gb_per_case,
                            "networks_per_case": profile.networks_per_case,
                            "max_cpu_per_case": profile.max_cpu_per_case,
                            "max_mem_mb_per_case": profile.max_mem_mb_per_case,
                            "task_count": profile.task_count,
                            "source": profile.source,
                            "detail": json.loads(profile.detail) if profile.detail else None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return 0

        # import-harbor
        source = f"github:{args.github}" if args.github else f"tasks-dir:{args.tasks_dir}"
        documents = (
            load_github_tasks(args.github) if args.github else load_tasks_dir(args.tasks_dir)
        )
        profile = upsert_harbor_resources(
            library,
            dataset_key=args.dataset_key,
            task_type=args.task_type,
            documents=documents,
            source=source,
        )
        print(f"已导入 {len(documents)} 个 task（{source}）")
        print(
            json.dumps(
                {
                    "dataset_key": args.dataset_key,
                    "task_type": args.task_type,
                    "task_count": profile.task_count,
                    "cpu_per_case": profile.cpu_per_case,
                    "mem_mb_per_case": profile.mem_mb_per_case,
                    "disk_gb_per_case": profile.disk_gb_per_case,
                    "networks_per_case": profile.networks_per_case,
                    "max_cpu_per_case": profile.max_cpu_per_case,
                    "max_mem_mb_per_case": profile.max_mem_mb_per_case,
                    "detail": json.loads(profile.detail) if profile.detail else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0


def _snapshot_dict(snapshot) -> dict:
    return {
        "name": snapshot.name,
        "cpu_cores": snapshot.cpu_cores,
        "mem_mb": snapshot.mem_mb,
        "disk_gb": snapshot.disk_gb,
        "network_available": snapshot.network_available,
        "network_pool": snapshot.network_pool,
        "docker_networks": snapshot.docker_networks,
        "captured_at": snapshot.captured_at,
        "source": snapshot.source,
    }


def _print_server(snapshot) -> None:
    print(f"服务器资源: {snapshot.name}")
    print(f"  CPU: {snapshot.cpu_cores:g} 核")
    print(f"  内存: {snapshot.mem_mb} MB")
    print(f"  磁盘: {snapshot.disk_gb if snapshot.disk_gb is not None else '-'} GB")
    print(f"  可用网段: {snapshot.network_available if snapshot.network_available is not None else '-'}")
    if snapshot.network_pool:
        print(f"  网段池: {snapshot.network_pool}")
    print(f"  采集时间: {snapshot.captured_at} ({snapshot.source})")


# -- dataset ---------------------------------------------------------------


def _cmd_dataset(args: argparse.Namespace) -> int:
    if args.action == "add":
        print(
            "[error] 通用自定义入库尚未实现：请使用 dataset import，"
            "或在代码中调用 Library.add_cases()",
            file=sys.stderr,
        )
        return 4

    db_path = Path(args.db) if args.db else DEFAULT_DB

    with Library(db_path) as library:
        if args.action == "list":
            datasets = library.list_datasets()
            if not datasets:
                print(f"数据集库为空: {library.path.resolve()}")
                return 0
            print(f"数据集库: {library.path.resolve()}")
            for item in datasets:
                aliases = ", ".join(item.aliases) if item.aliases else "-"
                observations = (
                    f"  observations={item.case_count}"
                    if item.case_count != item.unique_case_count
                    else ""
                )
                print(
                    f"  {item.dataset_key}  task_type={item.task_type}  "
                    f"cases={item.unique_case_count}{observations}  runs={item.run_count}  "
                    f"in={item.input_tokens_total}  out={item.output_tokens_total}  "
                    f"aliases={aliases}"
                )
            return 0

        summary = import_source(
            library,
            args.source,
            task_type=args.task_type,
            alias=args.alias,
            model_name=args.model_name,
            run_id=args.run_id,
            dataset_key=args.dataset_key,
            observed_concurrency=args.observed_concurrency,
            decode_tps=args.decode_tps,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    return 0
