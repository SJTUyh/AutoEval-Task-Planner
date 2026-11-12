"""对 OpenAI 兼容接口做并发-吞吐探测。

默认按 OpenAI Chat Completions 协议发送（与 AISBench 的 `vllm_api_general_chat` 一致），
同时允许覆盖 endpoint、鉴权头、额外请求头、请求体字段与流式开关，以适配：

- 自建服务：vLLM / SGLang / TGI / MindIE / Triton（OpenAI 兼容模式，通常无鉴权或 Bearer）；
- 云厂商：DeepSeek、OpenRouter、Kimi、通义等 OpenAI 兼容接口；
- Azure OpenAI：`--endpoint` 带 deployment + api-version，`--api-key-header api-key --api-key-scheme ""`；
- 仅支持非流式或拒绝 `stream_options` 的后端：`--no-stream` 或自动回退。

只使用标准库。
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from .errors import ProbeError
from .throughput import Level, ThroughputProfile, now_iso

DEFAULT_PREFILL_TPS = 8000.0
DEFAULT_ENDPOINT_SUFFIX = "/chat/completions"
MAX_CONCURRENCY = 64
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
USER_AGENT = "autoeval-planner/0.1"


@dataclass
class ProbeResult:
    ok: bool
    status: int | None = None
    ttft_sec: float | None = None
    duration_sec: float = 0.0
    output_tokens: int = 0
    output_tokens_estimated: bool = False
    error: str | None = None


@dataclass
class RequestOptions:
    url: str
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_scheme: str = "Bearer "
    headers: dict[str, str] = field(default_factory=dict)
    stream: bool = True
    body_overrides: dict = field(default_factory=dict)
    insecure: bool = False
    retries: int = 2


def run_probe(
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    concurrency_levels: list[int],
    requests_per_level: int = 2,
    prompt_tokens: int = 512,
    max_tokens: int = 512,
    effort: str | None = None,
    timeout: float = 300.0,
    endpoint: str | None = None,
    api_key_header: str = "Authorization",
    api_key_scheme: str = "Bearer ",
    headers: dict[str, str] | None = None,
    stream: bool = True,
    body_overrides: dict | None = None,
    insecure: bool = False,
    retries: int = 2,
) -> ThroughputProfile:
    levels = sorted({int(level) for level in concurrency_levels})
    if not levels:
        raise ProbeError("并发档位不能为空")
    if levels[0] < 1 or levels[-1] > MAX_CONCURRENCY:
        raise ProbeError(f"并发档位必须在 1..{MAX_CONCURRENCY} 之间")
    if requests_per_level < 1:
        raise ProbeError("--requests-per-level 必须大于等于 1")
    if prompt_tokens < 1 or max_tokens < 1:
        raise ProbeError("--prompt-tokens / --max-tokens 必须大于等于 1")
    if retries < 0:
        raise ProbeError("--retries 不能为负数")

    url = endpoint.strip() if endpoint else _chat_url(base_url)
    if not url:
        raise ProbeError("--endpoint 不能为空")
    prompt = _build_prompt(prompt_tokens)
    options = RequestOptions(
        url=url,
        api_key=api_key,
        api_key_header=api_key_header,
        api_key_scheme=api_key_scheme,
        headers=dict(headers or {}),
        stream=stream,
        body_overrides=dict(body_overrides or {}),
        insecure=insecure,
        retries=retries,
    )

    levels_out: list[Level] = []
    errors: list[str] = []
    stream_used = stream

    for concurrency in levels:
        total = max(concurrency, requests_per_level)
        results = _run_level(
            options=options,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            effort=effort,
            timeout=timeout,
            concurrency=concurrency,
            total=total,
        )
        # 后端不支持流式时（整档 4xx/405/415），自动回退到非流式重试该档位。
        if stream_used and all(not item.ok for item in results) and any(
            item.status in {400, 404, 405, 415, 422} for item in results
        ):
            fallback = _run_level(
                options=replace(options, stream=False),
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                effort=effort,
                timeout=timeout,
                concurrency=concurrency,
                total=total,
            )
            if any(item.ok for item in fallback):
                results = fallback
                stream_used = False
                options = replace(options, stream=False)

        ok_results = [item for item in results if item.ok]
        if not ok_results:
            first = next((item.error for item in results if item.error), "未知错误")
            errors.append(f"concurrency={concurrency}: {first}")
            continue

        # 只统计纯 decode 吞吐：从每个请求的耗时中扣除 TTFT，再按并发聚合，
        # 避免仿真里再单独计一次首 token 延迟造成重复计算。
        rates = []
        for item in ok_results:
            decode_time = item.duration_sec - (item.ttft_sec or 0.0)
            if decode_time <= 0.05:
                decode_time = item.duration_sec
            rates.append(item.output_tokens / max(decode_time, 0.05))
        decode_tps = (sum(rates) / len(rates)) * min(concurrency, len(rates))
        ttfts = [item.ttft_sec for item in ok_results if item.ttft_sec is not None]
        levels_out.append(
            Level(
                concurrency=concurrency,
                decode_tps=max(decode_tps, 1e-3),
                ttft_sec=sum(ttfts) / len(ttfts) if ttfts else None,
                error_rate=1 - len(ok_results) / len(results),
                usage_estimated=any(item.output_tokens_estimated for item in ok_results),
            )
        )

    if not levels_out:
        raise ProbeError("所有并发档位均探测失败: " + "; ".join(errors))

    peak = max(level.decode_tps for level in levels_out)
    healthy = [
        level
        for level in levels_out
        if level.error_rate <= 0.05 and level.decode_tps >= peak * 0.95
    ]
    max_concurrency = (
        max(level.concurrency for level in healthy) if healthy else levels_out[-1].concurrency
    )

    # 推理模型的 TTFT 大部分是思考时间，不能反推 prefill；只在 TTFT 很短时采用。
    single = next((level for level in levels_out if level.concurrency == 1), None)
    if single is not None and single.ttft_sec and single.ttft_sec <= 1.0:
        prefill_tps: float | None = prompt_tokens / single.ttft_sec
    else:
        prefill_tps = None

    for level in levels_out:
        if level.error_rate > 0:
            errors.append(f"concurrency={level.concurrency}: 错误率 {level.error_rate:.0%}")
        if level.usage_estimated:
            errors.append(f"concurrency={level.concurrency}: 缺少 usage，输出 token 为估算值")

    notes = "; ".join(errors) if errors else None
    profile = ThroughputProfile(
        source="probe",
        levels=levels_out,
        base_url=base_url,
        model=model,
        effort=effort,
        measured_at=now_iso(),
        notes=notes,
        request={
            "endpoint": url,
            "stream": stream_used,
            "api_key_header": api_key_header if api_key else None,
            "headers": sorted((headers or {})),
            "body_overrides": sorted((body_overrides or {})),
        },
    )
    profile.peak_prefill_tps = prefill_tps
    return profile


def _run_level(
    *,
    options: RequestOptions,
    model: str,
    prompt: str,
    max_tokens: int,
    effort: str | None,
    timeout: float,
    concurrency: int,
    total: int,
) -> list[ProbeResult]:
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _one_request,
                options=options,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                effort=effort,
                timeout=timeout,
            )
            for _ in range(total)
        ]
        return [future.result() for future in futures]


def _one_request(
    *,
    options: RequestOptions,
    model: str,
    prompt: str,
    max_tokens: int,
    effort: str | None,
    timeout: float,
) -> ProbeResult:
    body = _build_body(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        effort=effort,
        stream=options.stream,
        overrides=options.body_overrides,
    )
    headers = _build_headers(options)

    last: ProbeResult | None = None
    for attempt in range(options.retries + 1):
        result = _attempt(options, body, headers, timeout)
        if result.ok or result.status not in RETRYABLE_STATUS or attempt >= options.retries:
            return result
        last = result
        time.sleep(min(2**attempt, 8))
    return last or ProbeResult(ok=False, error="请求失败")


def _attempt(
    options: RequestOptions,
    body: dict,
    headers: dict[str, str],
    timeout: float,
) -> ProbeResult:
    request = urllib.request.Request(
        options.url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.monotonic()
    try:
        context = None
        if options.insecure:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            if options.stream:
                result = _parse_stream(response, start)
            else:
                result = _parse_json(response, start)
        return result
    except urllib.error.HTTPError as exc:
        return ProbeResult(
            ok=False,
            status=exc.code,
            error=f"HTTP {exc.code}: {_short_body(exc)}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return ProbeResult(ok=False, error=f"请求失败: {exc}")


def _parse_stream(response, start: float) -> ProbeResult:
    ttft: float | None = None
    output_tokens = 0
    text_chars = 0
    delta_chunks = 0
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens"):
            output_tokens = int(usage["completion_tokens"])
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = (
            delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("reasoning")
            or ""
        )
        if piece:
            if ttft is None:
                ttft = time.monotonic() - start
            delta_chunks += 1
        text_chars += len(str(delta.get("content") or ""))

    duration = time.monotonic() - start
    estimated = output_tokens <= 0
    if estimated:
        # 无 usage 时：优先用非空 delta 数（OpenAI 流式下约等于 token 数），退化为字符数/4。
        output_tokens = max(delta_chunks, text_chars // 4, 1)
    return ProbeResult(
        ok=True,
        status=200,
        ttft_sec=ttft,
        duration_sec=duration,
        output_tokens=output_tokens,
        output_tokens_estimated=estimated,
    )


def _parse_json(response, start: float) -> ProbeResult:
    raw = response.read().decode("utf-8", "replace")
    duration = time.monotonic() - start
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ProbeResult(
            ok=False,
            status=200,
            error=f"响应不是合法 JSON: {raw[:120]}",
        )
    if not isinstance(payload, dict):
        return ProbeResult(ok=False, status=200, error="响应顶层不是 JSON 对象")

    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    output_tokens = int(usage.get("completion_tokens") or 0)
    estimated = output_tokens <= 0
    if estimated:
        choices = payload.get("choices") or []
        message = (choices[0].get("message") if choices else {}) or {}
        text = str(message.get("content") or message.get("reasoning_content") or "")
        output_tokens = max(len(text) // 4, 1)

    if payload.get("error") and not payload.get("choices"):
        return ProbeResult(
            ok=False,
            status=200,
            error=f"服务返回错误: {json.dumps(payload['error'], ensure_ascii=False)[:160]}",
        )
    return ProbeResult(
        ok=True,
        status=200,
        ttft_sec=None,
        duration_sec=duration,
        output_tokens=output_tokens,
        output_tokens_estimated=estimated,
    )


def _build_body(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    effort: str | None,
    stream: bool,
    overrides: dict,
) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if effort:
        body["reasoning_effort"] = effort
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

    body.update(overrides)
    # 允许用 null 删除默认字段（例如后端不接受 stream_options / temperature）。
    for key in [key for key, value in body.items() if value is None]:
        body.pop(key)
    return body


def _build_headers(options: RequestOptions) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if options.stream else "application/json",
        "User-Agent": USER_AGENT,
    }
    if options.api_key:
        headers[options.api_key_header] = (options.api_key_scheme or "") + options.api_key
    headers.update(options.headers)
    return headers


def _short_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 错误信息尽力而为
        return exc.reason or ""
    return body[:200]


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith(DEFAULT_ENDPOINT_SUFFIX):
        return base
    return base + DEFAULT_ENDPOINT_SUFFIX


def _build_prompt(prompt_tokens: int) -> str:
    prefix = (
        "Read the following material carefully, then write a long, detailed explanation "
        "in English. Keep writing until you reach the length limit.\n\n"
    )
    filler = (
        "The evaluation system processes requests concurrently and records per-case token "
        "usage, latency and throughput. "
    )
    target_chars = max(64, prompt_tokens * 4)
    body = (filler * (target_chars // len(filler) + 1))[:target_chars]
    return prefix + body
