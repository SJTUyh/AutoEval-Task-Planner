"""服务器资源采集与数据集资源画像导入。"""

from __future__ import annotations

import concurrent.futures
import json
import re
import statistics
import tomllib
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .errors import ConfigError, LibraryError
from .library.db import Library, ResourceProfile, ServerSnapshot

NETWORK_SNIPPET = (
    'python3 -c "import json,subprocess,ipaddress as I,os;'
    "F='/etc/docker/daemon.json';"
    "p=(json.load(open(F)).get('default-address-pools') if os.path.exists(F) else None) "
    "or [{'base':'172.17.0.0/12','size':16}];"
    "ids=subprocess.check_output(['docker','network','ls','-q']).decode().split();"
    "u=[I.ip_network(c['Subnet'],strict=False) for n in "
    "(json.loads(subprocess.check_output(['docker','network','inspect']+ids).decode()) if ids else []) "
    "for c in ((n.get('IPAM') or {}).get('Config') or []) if c.get('Subnet')];"
    "ns=lambda bl:sum(1 for b in bl if any(z.overlaps(b) for z in u));"
    "R=[(x['base'],int(x['size']),"
    "[b for b in I.ip_network(x['base'],strict=False).subnets(new_prefix=int(x['size'])) "
    "if b.network_address>=I.ip_address(x['base'].split('/')[0])]) for x in p];"
    "D=[(b,s,len(bl),ns(bl)) for b,s,bl in R];"
    "A=sum(c-k for _,_,c,k in D);"
    "print('pool: '+'  +  '.join('%s size=%d (cap=%d used=%d free=%d)'%(b,s,c,k,c-k) for b,s,c,k in D));"
    "print('existing_networks=%d'%len(ids));"
    "print('available=%d'%A)\""
)

PROBE_COMMAND = (
    "echo __CPU__; nproc; "
    "echo __MEM__; free -m; "
    "echo __DISK__; (df -BG --output=avail / 2>/dev/null || df -BG /); "
    "echo __NET__; " + NETWORK_SNIPPET
)


def collect_via_ssh(
    *,
    name: str,
    host: str,
    user: str,
    port: int = 22,
    password: str | None = None,
    key_path: str | None = None,
    timeout: float = 20.0,
) -> ServerSnapshot:
    try:
        import paramiko  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - 依赖缺失路径
        raise ConfigError(
            "SSH 采集需要 paramiko，请先安装：pip install 'autoeval-task-planner[ssh]'"
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            host,
            port=port,
            username=user,
            password=password,
            key_filename=key_path,
            timeout=timeout,
        )
        _, stdout, stderr = client.exec_command(PROBE_COMMAND, timeout=timeout + 30)
        text = stdout.read().decode("utf-8", "replace")
        error = stderr.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - 统一转换为可读错误
        raise ConfigError(f"SSH 采集失败: {exc}") from exc
    finally:
        client.close()

    if "available=" not in text:
        raise ConfigError(f"SSH 采集输出不完整: {error or text[:200]}")
    snapshot = parse_probe_output(text, name=name, source="ssh")
    return snapshot


def parse_probe_output(text: str, *, name: str, source: str = "ssh") -> ServerSnapshot:
    sections = _split_sections(text)
    cpu_text = sections.get("CPU", "").strip()
    try:
        cpu_cores = float(cpu_text.splitlines()[0].strip())
    except (IndexError, ValueError) as exc:
        raise ConfigError(f"无法解析 CPU 核数: {cpu_text[:80]!r}") from exc

    mem_mb = _parse_mem_mb(sections.get("MEM", ""))
    disk_gb = _parse_disk_gb(sections.get("DISK", ""))
    net = _parse_network(sections.get("NET", ""))

    return ServerSnapshot(
        name=name,
        cpu_cores=cpu_cores,
        mem_mb=mem_mb,
        disk_gb=disk_gb,
        network_available=net["available"],
        network_pool=net["pool"],
        docker_networks=net["existing"],
        captured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source=source,
        raw=json.dumps({"output": text}, ensure_ascii=False),
    )


def snapshot_from_values(
    *,
    name: str,
    cpu_cores: float,
    mem_mb: int,
    disk_gb: float | None = None,
    network_available: int | None = None,
    network_pool: str | None = None,
    docker_networks: int | None = None,
    source: str = "manual",
) -> ServerSnapshot:
    if cpu_cores <= 0 or mem_mb <= 0:
        raise ConfigError("CPU 核数与内存必须大于 0")
    return ServerSnapshot(
        name=name,
        cpu_cores=cpu_cores,
        mem_mb=mem_mb,
        disk_gb=disk_gb,
        network_available=network_available,
        network_pool=network_pool,
        docker_networks=docker_networks,
        captured_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        source=source,
    )


def snapshot_from_file(path: str | Path) -> ServerSnapshot:
    source = Path(path).expanduser()
    if not source.is_file():
        raise ConfigError(f"资源快照文件不存在: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"资源快照不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("资源快照顶层必须是 JSON 对象")
    for key in ("name", "cpu_cores", "mem_mb"):
        if key not in payload:
            raise ConfigError(f"资源快照缺少字段: {key}")
    return snapshot_from_values(
        name=str(payload["name"]),
        cpu_cores=float(payload["cpu_cores"]),
        mem_mb=int(payload["mem_mb"]),
        disk_gb=float(payload["disk_gb"]) if payload.get("disk_gb") is not None else None,
        network_available=(
            int(payload["network_available"])
            if payload.get("network_available") is not None
            else None
        ),
        network_pool=payload.get("network_pool"),
        docker_networks=(
            int(payload["docker_networks"])
            if payload.get("docker_networks") is not None
            else None
        ),
        source=str(payload.get("source") or "file"),
    )


# -- Harbor task.toml -----------------------------------------------------


def resource_profile_from_tasks(
    documents: Iterable[tuple[str, dict]],
    source: str,
) -> tuple[ResourceProfile, dict]:
    cpu_values: list[float] = []
    mem_values: list[int] = []
    storage_values: list[int] = []
    verifier_cpu: list[float] = []
    verifier_mem: list[int] = []
    network_modes: set[str] = set()
    task_count = 0

    for _name, doc in documents:
        environment = doc.get("environment") or {}
        cpus = environment.get("cpus")
        memory_mb = environment.get("memory_mb")
        if not isinstance(cpus, (int, float)) or not isinstance(memory_mb, (int, float)):
            continue
        task_count += 1
        cpu_values.append(float(cpus))
        mem_values.append(int(memory_mb))
        storage = environment.get("storage_mb")
        if isinstance(storage, (int, float)):
            storage_values.append(int(storage))
        verifier = (doc.get("verifier") or {}).get("environment") or {}
        if isinstance(verifier.get("cpus"), (int, float)):
            verifier_cpu.append(float(verifier["cpus"]))
        if isinstance(verifier.get("memory_mb"), (int, float)):
            verifier_mem.append(int(verifier["memory_mb"]))
        agent = doc.get("agent") or {}
        if isinstance(agent.get("network_mode"), str):
            network_modes.add(agent["network_mode"])

    if not cpu_values:
        raise ConfigError("task.toml 中没有可用的 environment.cpus / memory_mb")

    profile = ResourceProfile(
        cpu_base=0.5,
        mem_mb_base=256,
        cpu_per_case=statistics.median(cpu_values),
        mem_mb_per_case=int(statistics.median(mem_values)),
        disk_gb_per_case=(
            statistics.median(storage_values) / 1024 if storage_values else 0.0
        ),
        networks_per_case=1,
        source=source,
        max_cpu_per_case=max(cpu_values),
        max_mem_mb_per_case=max(mem_values),
        task_count=task_count,
        detail=json.dumps(
            {
                "verifier_cpu_median": statistics.median(verifier_cpu) if verifier_cpu else None,
                "verifier_mem_mb_median": (
                    int(statistics.median(verifier_mem)) if verifier_mem else None
                ),
                "network_modes": sorted(network_modes),
                "cpu_min": min(cpu_values),
                "cpu_max": max(cpu_values),
                "mem_mb_min": min(mem_values),
                "mem_mb_max": max(mem_values),
            },
            ensure_ascii=False,
        ),
    )
    return profile, json.loads(profile.detail or "{}")


def load_tasks_dir(path: str | Path) -> list[tuple[str, dict]]:
    root = Path(path).expanduser()
    if not root.is_dir():
        raise ConfigError(f"tasks 目录不存在: {root}")
    documents: list[tuple[str, dict]] = []
    for candidate in sorted(root.glob("*/task.toml")):
        try:
            documents.append(
                (candidate.parent.name, tomllib.loads(candidate.read_text(encoding="utf-8")))
            )
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"解析 {candidate} 失败: {exc}") from exc
    if not documents:
        raise ConfigError(f"{root} 下没有 */task.toml")
    return documents


def load_github_tasks(repo_ref: str, timeout: float = 30.0) -> list[tuple[str, dict]]:
    if "@" not in repo_ref:
        raise ConfigError("--github 需要 owner/repo@ref 形式，如 datacurve-ai/deep-swe@main")
    repo, ref = repo_ref.split("@", 1)
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{ref}?recursive=1"
    tree = _http_json(tree_url, timeout)
    paths = [
        str(item["path"])
        for item in tree.get("tree", [])
        if str(item.get("path", "")).startswith("tasks/")
        and str(item.get("path", "")).endswith("/task.toml")
    ]
    if not paths:
        raise ConfigError(f"{repo_ref} 中未找到 tasks/*/task.toml")
    raw_base = f"https://raw.githubusercontent.com/{repo}/{ref}/"

    def fetch(path: str) -> tuple[str, dict]:
        payload = _http_bytes(raw_base + path, timeout).decode("utf-8", "replace")
        return Path(path).parent.name, tomllib.loads(payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        return list(executor.map(fetch, paths))


def _http_json(url: str, timeout: float) -> dict:
    payload = _http_bytes(url, timeout).decode("utf-8", "replace")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise LibraryError(f"GET {url} 返回的不是 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LibraryError(f"GET {url} 返回的 JSON 顶层不是对象")
    return data


def _http_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "autoeval-planner/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LibraryError(f"GET {url} 失败: {exc}") from exc


# -- 解析辅助 -------------------------------------------------------------


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"__(CPU|MEM|DISK|NET)__", line.strip())
        if match:
            if current:
                sections[current] = "\n".join(buffer)
            current = match.group(1)
            buffer = []
            continue
        if current:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer)
    return sections


def _parse_mem_mb(text: str) -> int:
    for line in text.splitlines():
        if line.strip().startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    return int(parts[6])
                except ValueError:
                    break
    raise ConfigError(f"无法解析内存可用量: {text[:120]!r}")


def _parse_disk_gb(text: str) -> float | None:
    for line in text.splitlines():
        match = re.match(r"^\s*(\d+(?:\.\d+)?)G\s*$", line)
        if match:
            return float(match.group(1))
    return None


def _parse_network(text: str) -> dict:
    available: int | None = None
    existing: int | None = None
    pool: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if "available=" in stripped:
            available = _int_after(stripped, "available=")
        elif re.search(r"可用网络数[:：]", stripped):
            available = _int_after(stripped, "可用网络数")
        elif "existing_networks=" in stripped:
            existing = _int_after(stripped, "existing_networks=")
        elif "现存网络总数" in stripped:
            existing = _int_after(stripped, "现存网络总数")
        if pool is None and ("pool:" in stripped or "池配置:" in stripped):
            pool = stripped.split(":", 1)[-1].strip()
    if available is None:
        raise ConfigError(f"无法解析可用网段数: {text[:200]!r}")
    return {"available": available, "existing": existing, "pool": pool}


def _int_after(text: str, marker: str) -> int | None:
    match = re.search(re.escape(marker) + r"\s*[:：]?\s*(\d+)", text)
    return int(match.group(1)) if match else None


def upsert_harbor_resources(
    library: Library,
    *,
    dataset_key: str,
    task_type: str,
    documents: list[tuple[str, dict]],
    source: str,
) -> ResourceProfile:
    profile, _detail = resource_profile_from_tasks(documents, source)
    library.upsert_dataset(
        dataset_key=dataset_key,
        task_type=task_type,
        display_name=dataset_key,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    library.upsert_dataset_resources(dataset_key, task_type, profile)
    return profile
