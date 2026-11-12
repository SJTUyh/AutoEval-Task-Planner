# AutoEval-Task-Planner

评估一整套测评任务在指定推理服务上的**最优总时长**，并给出可直接落地的编排建议：

- 总测评时长（p10 / p50 / p90 波动区间）；
- 数据集任务执行顺序、数据集级并发数；
- 每个数据集内部的 case 并发数；
- 可直接回填到 AutoEval-Benchmark-Adaptor 的参数建议（`batch_size` / `n_concurrent_trials` / `max_num_workers`）。

## 1. 工作原理

```
Adaptor 任务配置 ─┐
历史 case token 库 ├─> case 粒度离散事件仿真 ─> 枚举任务顺序/两级并发 ─> 蒙特卡洛 ─> plan.json
服务吞吐曲线 ──────┘
服务器资源（预留）
```

- **历史库**：SQLite 记录每个数据集每个 case 的输入/输出 token 与耗时，不区分被测模型；规划时按 `case_id` 去重；
- **吞吐曲线**：对目标 API 实测「并发 → 纯 decode 吞吐 / TTFT」，或手工传入；
- **仿真**：按 case 的 token 量与 overhead 推进离散事件，服务吞吐按当前解码并发从曲线插值并公平分享；
- **寻优**：枚举数据集顺序、数据集并发、case 并发，先确定性粗筛，再对前 5 个方案做蒙特卡洛得到区间。

## 2. 安装

```bash
cd AutoEval-Task-Planner
pip install -e .
```

只依赖 Python 标准库（>=3.10）。不安装也可以：

```bash
$env:PYTHONPATH='src'; python -m autoeval_planner --help
```

## 3. 快速开始

### 3.1 实测服务吞吐

```bash
$env:DEEPSEEK_API_KEY='sk-...'
autoeval-planner probe --base-url https://api.deepseek.com/v1 --model deepseek-flash \
  --api-key-env DEEPSEEK_API_KEY --concurrency 1,2,4,8,16,32 \
  --requests-per-level 1 --prompt-tokens 256 --max-tokens 2048 --effort max \
  --output examples/throughput_deepseek_flash.json
```

- 每个并发档位至少发 `max(concurrency, --requests-per-level)` 个请求；
- 用流式响应测 TTFT，用末帧 `usage.completion_tokens` 统计真实输出 token；
- `decode_tps` 只统计**纯 decode 阶段**（已扣除 TTFT），避免与仿真里的首 token 延迟重复计算；
- `max_concurrency` 取「错误率 ≤ 5% 且吞吐 ≥ 峰值 95%」的最大档位。

API key 只在 `probe` 需要，两种传法（`plan` 只读吞吐文件，不需要 key）：

```bash
# 方式一：环境变量（推荐，不会留在命令历史里）
export DEEPSEEK_API_KEY='sk-...'          # PowerShell: $env:DEEPSEEK_API_KEY='sk-...'
autoeval-planner probe --api-key-env DEEPSEEK_API_KEY ...

# 方式二：命令行直接传入
autoeval-planner probe --api-key 'sk-...' ...
```

key 只用于请求头，不会写入 `throughput.json`、`plan.json` 或日志。

#### 兼容不同服务

默认按 OpenAI Chat Completions 协议发请求，以下参数用于适配自建服务与不同厂商：

```bash
# 自建 vLLM / SGLang（通常无鉴权）
autoeval-planner probe --base-url http://10.0.0.8:8000/v1 --model qwen3 --concurrency 1,2,4,8

# Bearer 鉴权（DeepSeek / OpenRouter / Kimi 等）
autoeval-planner probe --base-url https://api.deepseek.com/v1 --model deepseek-flash \
  --api-key-env DEEPSEEK_API_KEY

# Azure OpenAI：完整 endpoint（含 deployment 与 api-version）+ api-key 头
autoeval-planner probe \
  --endpoint "https://<res>.openai.azure.com/openai/deployments/<dep>/chat/completions?api-version=2024-10-21" \
  --model gpt-4o --api-key-env AZURE_OPENAI_KEY \
  --api-key-header api-key --api-key-scheme=

# 只支持非流式的后端
autoeval-planner probe ... --no-stream

# 后端拒绝 stream_options 时（用 null 删除该字段）
autoeval-planner probe ... --body '{"stream_options": null}'

# 自签名证书 / 额外请求头 / 额外请求体字段
autoeval-planner probe ... --insecure --header "X-Org: eval" --body '{"top_p": 0.95}'
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--endpoint` | 空 | 完整请求地址，优先于 `base_url + /chat/completions` |
| `--api-key-header` | `Authorization` | `api-key` 等非标准头 |
| `--api-key-scheme` | `Bearer ` | Azure 用空字符串 |
| `--header` | 空 | 追加请求头，可重复 |
| `--body` | 空 | 合并进请求体的 JSON；值为 `null` 表示删除默认字段 |
| `--no-stream` | 流式 | 非流式模式下没有 TTFT，decode 吞吐按总耗时计算 |
| `--insecure` | 关闭 | 跳过 TLS 校验 |
| `--retries` | 2 | 仅对 408/429/5xx 重试 |
| `--concurrency` | `1,2,4,8` | 并发档位，可省略；自建服务建议按显存/服务上限追加 16、32 |
| `--requests-per-level` | 2 | 每档至少发的请求数（实际取 `max(concurrency, 该值)`） |
| `--prompt-tokens` / `--max-tokens` | 512 / 512 | 探测用输入与最大输出长度 |
| `--timeout` | 300 | 单请求超时秒数 |

流式请求被整档拒绝（400/404/405/415/422）时会自动回退为非流式重试该档位。原生 Anthropic `/v1/messages`、Gemini `generateContent`、TGI 原生 `/generate` 不是 OpenAI 协议，需使用各家的 OpenAI 兼容入口或网关。

### 3.2 导入历史测评数据

```bash
# GPQA（AISBench run）
autoeval-planner dataset import \
  --source /home/yanhe/runs/gpqa_full_256k/20261111_071448 \
  --task-type ais_bench/llm_accuracy \
  --alias gpqa_gen_0_shot_cot_chat_prompt \
  --observed-concurrency 8 --decode-tps 1106

# MMMU-Pro（AISBench run）
autoeval-planner dataset import \
  --source /home/yanhe/runs/mmmu_pro_full_256k/20261111_093144 \
  --task-type ais_bench/llm_accuracy \
  --alias mmmu_pro_options10_gen \
  --observed-concurrency 8 --decode-tps 1244

# DeepSWE（Harbor run，两组 run 合并到同一数据集）
for run in /home/yanhe/history/byf_outputs/default/20260907_191217 \
           /home/yanhe/history/byf_outputs/default/20261021_103051; do
  autoeval-planner dataset import --source "$run" \
    --task-type ais_bench/harbor_agent \
    --alias harbor_deepswe-v1.1 --dataset-key harbor_deepswe-v1.1 \
    --observed-concurrency 16 --decode-tps 2300
done

autoeval-planner dataset list
```

- 来源支持 AISBench run 目录（`predictions/*/*.jsonl`）、Harbor run 目录（`results/<model>/<dataset>/details/*/result.json`）和 Adaptor 的 `unified_result.json`；
- `--alias` 对应 Adaptor 配置里 `params.datasets` 的别名，用于规划时定位数据集；
- `--observed-concurrency` + `--decode-tps` 用于把「实测耗时 − 纯 decode 时间」拆成每 case 的非 LLM 开销（agent 场景有意义，accuracy 场景一般接近 0）；
- 多个 run 用 `--dataset-key` 归并到同一数据集键；同一 run 重复导入幂等。

当前库内容：

```text
GPQA_diamond         task_type=ais_bench/llm_accuracy  cases=198   runs=1  in=53,823        out=2,785,651
mmmu_pro             task_type=ais_bench/llm_accuracy  cases=1730  runs=1  in=926,757       out=11,732,376
harbor_deepswe-v1.1  task_type=ais_bench/harbor_agent cases=113   runs=2  in=5,379,540,058 out=23,091,129  observations=226
```

通用自定义入库入口保留为 `Library.add_cases()`（见 `autoeval_planner.library.db`）；`dataset add` 子命令当前明确返回「未实现」。

### 3.3 生成规划

```bash
autoeval-planner plan --tasks examples/tasks_deepseek_v41_flash.json \
  --db data/dataset_library.sqlite3 --output plan.json
```

输出示例（节选）：

```text
规划结果
  目标服务: https://api.deepseek.com/v1 (deepseek-flash)
  吞吐来源: probe (decode 4257.561 tok/s @ c=32)
  最优总时长: 3h38m26s (p10 3h21m40s, p90 4h11m40s)
  数据集顺序: deepswe -> mmmu_pro -> gpqa
  数据集并发: 2
  case 并发: gpqa=32, mmmu_pro=32, deepswe=27
  服务利用率: 53.4%
  置信度: high
  各数据集时长:
    - gpqa: 单独 18m51.4s，计划内 22m27.9s（第 1h18m57s 开始）
    - mmmu_pro: 单独 1h05m10s，计划内 1h18m57s（第 0.0s 开始）
    - deepswe: 单独 3h24m38s，计划内 3h38m04s（第 0.0s 开始）
```

「单独」指该数据集以计划中的 case 并发独占服务与资源时的耗时；「计划内」是该数据集在当前编排中的实际起止时长。该结果使用 `resources.mode=server`（179.214.2.3，headroom=0.2）；资源无限模式下的结果为 2h19m44s。峰值 CPU 56.1/72 核（约 78%），保留约 22% 物理裕量。

### 3.4 手工指定吞吐

不跑 `probe` 时，可在 taskset 里直接给定吞吐（简写假设 `c=1` 到 `c=max` 线性扩展）：

```json
"throughput": {"mode": "inline", "inline": {"decode_tps": 4000, "max_concurrency": 32}}
```

同一组任务、不同吞吐假设的实测对比（`resources.mode=infinite`，用于观察吞吐敏感性）：

| 吞吐来源 | 最优总时长 | p10 – p90 | 数据集顺序 | 数据集并发 | case 并发（gpqa / mmmu / deepswe） | 利用率 | 置信度 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probe 4257 tok/s @ c=32 | 2h19m44s | 2h02m04s – 2h40m18s | deepswe→gpqa→mmmu | 2 | 32 / 32 / 113 | 74.6% | high |
| 手工 4000 tok/s @ c=32 | 2h23m52s | 2h05m20s – 2h45m37s | gpqa→deepswe→mmmu | 2 | 32 / 32 / 113 | 76.8% | medium |
| 手工 1100 tok/s @ c=8 | 6h40m45s | 5h55m24s – 7h58m48s | mmmu→deepswe→gpqa | 2 | 8 / 8 / 64 | 99.0% | medium |

手工吞吐未实测，`confidence` 会降一级；`4000 @ c=32` 与实测 `4257 @ c=32` 相差约 3%，可作为一致性参考。

## 4. 输入格式（taskset）

```json
{
  "target": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-flash"},
  "throughput": {"mode": "file", "path": "throughput_deepseek_flash.json"},
  "resources": {"mode": "server", "server": "179.214.2.3", "headroom": 0.2},
  "tasks": [
    {"task_id": "gpqa", "config": "../../AutoEval-Standard/.../GPQA_Diamond/standard_config.json"},
    {"task_id": "mmmu_pro", "config": "../../AutoEval-Standard/.../MMMU-Pro/standard_config.json",
     "estimate": {"case_count": 1730, "input_tokens_median": 417, "input_tokens_p90": 964,
                  "output_tokens_median": 1102, "output_tokens_p90": 21856, "overhead_sec": 1}},
    {"task_id": "deepswe", "config": "../../AutoEval-Standard/.../DeepSWE/standard_config.json",
     "estimate": {"case_count": 113, "input_tokens_median": 22869388,
                  "input_tokens_p90": 43972280, "output_tokens_median": 105860,
                  "output_tokens_p90": 144340, "overhead_sec": 1597}}
  ],
  "simulation": {"runs": 100, "seed": 20261111, "throughput_cv": 0.10}
}
```

| 字段 | 说明 |
| --- | --- |
| `target` | 被测服务地址与模型名；规划本身不调用服务，不需要 API key（`probe` 的 key 通过 `--api-key` / `--api-key-env` 传入） |
| `throughput.mode=file` | 读取吞吐曲线 JSON；相对路径基于 taskset 所在目录 |
| `throughput.mode=inline` | 直接内嵌曲线或简写 `{"decode_tps": 1100, "max_concurrency": 8}` |
| `resources.mode` | `infinite`（默认，只受服务吞吐约束）或 `server`（受服务器 CPU/内存/磁盘/网段约束） |
| `resources.server` | `mode=server` 时必填，对应 `resource set/probe` 写入的快照名 |
| `resources.headroom` | 预留比例，默认 0.2；CPU/内存/磁盘按 `1-headroom` 可用 |
| `tasks[].config` | 任意符合 Adaptor 定义的配置（可以是 AutoEval-Standard 的**部分配置**） |
| `tasks[].estimate` | 仅在该数据集**无历史数据**时生效的兜底估算；有历史时会被忽略。示例中的数值取自真实历史分位数 |
| `tasks[].case_count` | 本次要跑的 case 数（数据集真实规模）。与历史唯一 case 数不同时，按历史分布重采样；默认用历史唯一 case 数 |
| `simulation.runs` | 蒙特卡洛次数，默认 100 |
| `simulation.throughput_cv` | 吞吐抖动（对数正态），默认 0.10 |

任务识别规则：

- `llm_accuracy`：遍历 `params.datasets`，每个别名生成一个子任务；case 上限取 `num_prompts` 或 `dataset_test_range`，case 并发上限取 `params.batch_size`；
- `harbor_agent`：数据集取 `params.dataset_abbr`（缺失时用 `path` 目录名）；case 数上限取 `n_tasks`，总数再乘 `n_attempts`，case 并发上限取 `params.n_concurrent_trials`；
- 别名通过库内 `dataset_aliases` 解析；导入时注册的别名与 `data_abbr` 都会记录。

## 5. 数据模型

`data/dataset_library.sqlite3`（纯标准库 `sqlite3`）：

| 表 | 关键字段 |
| --- | --- |
| `runs` | `run_id`、来源类型/路径、模型名、`observed_concurrency`、导入时间 |
| `datasets` | `dataset_key`（如 `GPQA_diamond`）、`task_type` |
| `dataset_aliases` | `(alias, task_type) → dataset_key` |
| `cases` | `(run_id, dataset_key, case_id, attempt)`、输入/输出 token、耗时、`overhead_sec`、状态 |

统计时跨所有 run 按 `dataset_key` 聚合；规划时再**按 `case_id` 去重**（多个 run / 重试只作为同一 case 的多条观测，取成功观测的中位数），避免把同一道题重复计成多个 case。数字型 `case_id` 按数值排序，保持与 AISBench 的 FIFO 顺序一致。

`dataset list` 同时展示唯一 case 数与观测条数，例如：

```text
harbor_deepswe-v1.1  task_type=ais_bench/harbor_agent  cases=113  observations=226  runs=2  ...
```

## 6. 吞吐曲线格式

```json
{
  "schema_version": 1,
  "source": "probe",
  "base_url": "https://api.deepseek.com/v1",
  "model": "deepseek-flash",
  "effort": "max",
  "measured_at": "2026-11-11T16:40:06+08:00",
  "levels": [
    {"concurrency": 1, "decode_tps": 143.9, "ttft_sec": 6.2, "error_rate": 0.0, "usage_estimated": false},
    {"concurrency": 8, "decode_tps": 1134.2, "ttft_sec": 8.4, "error_rate": 0.0, "usage_estimated": false},
    {"concurrency": 32, "decode_tps": 4257.6, "ttft_sec": 9.9, "error_rate": 0.0, "usage_estimated": false}
  ],
  "request": {
    "endpoint": "https://api.deepseek.com/v1/chat/completions",
    "stream": true,
    "api_key_header": "Authorization",
    "headers": [],
    "body_overrides": []
  },
  "derived": {"max_concurrency": 32, "peak_decode_tps": 4257.6, "peak_prefill_tps": null}
}
```

- 档位之间线性插值，超过最大档位按最大档位处理；
- 手工简写 `{"decode_tps": 1100, "max_concurrency": 8}` 会展开成 `c=1`（137.5 tok/s）与 `c=8`（1100 tok/s）两个档位；
- 仿真中的服务吞吐按「当前解码请求数」取曲线，并公平分配给所有解码请求；
- `usage_estimated=true` 表示该档位没有拿到 `usage`，输出 token 由 delta 数/字符数估算；
- `request` 记录实际使用的 endpoint、流式模式与自定义项（不含 key），便于复现。

## 7. 输出（plan.json）

| 字段 | 说明 |
| --- | --- |
| `throughput_source` | 吞吐来源（`probe` / `manual`）、曲线与派生值 |
| `resource_assumption` | 服务器快照、headroom、可用 CPU/内存/磁盘/网段 |
| `resources` | 峰值占用、瓶颈资源、每任务资源画像与资源允许的最大并发 |
| `estimate.total_duration_sec` | 最优方案的总时长（p50） |
| `estimate.range` | `p10` / `p50` / `p90` 波动区间 |
| `estimate.utilization` | 服务利用率（已解码 token / 峰值吞吐 × 时长） |
| `plan.dataset_order` | 数据集执行顺序 |
| `plan.dataset_concurrency` | 数据集级并发数（同时运行的数据集数量） |
| `plan.case_concurrency` | 每个数据集内部的 case 并发数 |
| `plan.timeline` | 各任务在计划中的起止时间与 `duration_sec` |
| `tasks[].standalone_duration_sec` | 该数据集单独执行（计划并发、独占资源）的耗时 |
| `tasks[].scheduled_start_sec` / `scheduled_end_sec` / `scheduled_duration_sec` | 该数据集在当前编排中的起止与时长 |
| `plan.suggested_adaptor_overrides` | 回填建议：accuracy 用 `batch_size`，harbor 用 `n_concurrent_trials` |
| `plan.execution_groups` | 按配置文件分组，`max_num_workers` 为该配置下建议并行度 |
| `alternatives` | 次优方案（最多 2 个） |
| `tasks[].data_quality` | `history` / `estimate` / `heuristic` |
| `tasks[].source_runs` | 该数据集用到的历史 run |
| `confidence` | 综合数据质量与吞吐来源：`high` / `medium` / `low` |

## 8. 模型假设与局限

1. **纯 decode 吞吐**：曲线来自实测或传入的纯 decode 吞吐；accuracy 任务的输入 token 通过 TTFT（`实测 TTFT + 输入 token / prefill_tps`，默认 8000 tok/s）体现，不单独消耗 decode 吞吐。harbor_agent 的输入 token 是跨多轮累计值且大量命中缓存，因此**不计入 TTFT**，只保留在统计信息里。
2. **overhead**：非 LLM 固定开销（容器启动、工具执行）占用该任务的 case 并发名额，但不消耗服务吞吐。导入时可反推（`--observed-concurrency` + `--decode-tps`），估算时可直接给出。
3. **case 顺序**：数据集内部按 FIFO（与 AISBench 行为一致）；「最优」只体现在任务顺序和两级并发上。
4. **公平分享**：服务饱和后所有在途请求均分吞吐；不模拟排队延迟、限流与 429 退避。
5. **并发上限**：accuracy 的 case 并发不超过吞吐曲线的 `max_concurrency`；harbor 的 trial 并发允许更高（默认上限 256），因为大量时间花在工具执行上；两者都不会超过该任务实际的 case 数。
6. **服务器资源**：`resources.mode=infinite` 时只有推理服务吞吐是瓶颈；`server` 模式下 CPU/内存/磁盘/网段都是硬约束，见第 9 节。
7. **数据质量**：缺失历史数据时可用输入估算，或退化为同类参考分布/启发式默认；后两者会写入 `warnings` 并降低 `confidence`，`--strict` 下直接报错。
8. **探测协议**：默认 OpenAI Chat Completions + SSE；非 OpenAI 原生协议需走兼容入口。后端不回传 `usage` 时 token 由 delta 数/字符数估算，该档位会标记 `usage_estimated=true`。

## 9. 服务器资源

### 9.1 三层资源模型

| 层 | 表 | 说明 |
| --- | --- | --- |
| 服务器快照 | `servers` | CPU、内存、磁盘、可用 Docker 网段、采集时间与来源 |
| 工具默认 | `tool_resource_defaults` | 按 `(tool, subcommand)` 定义默认画像，便于后续扩展新工具 |
| 数据集画像 | `dataset_resources` | 按数据集覆盖工具默认（如 DeepSWE 从 task.toml 导入） |

内置默认：

| tool/subcommand | 任务基础 | 每在途 case | 网段 |
| --- | --- | --- | --- |
| `ais_bench/llm_accuracy` | 1 核 / 1024 MB | 0.02 核 / 16 MB | 0 |
| `ais_bench/harbor_agent` | 0.5 核 / 256 MB | 2 核 / 8192 MB | 1 |

### 9.2 采集与导入

```bash
# SSH 采集 CPU/内存/磁盘/可用网段
autoeval-planner resource probe --name 179.214.2.3 \
  --ssh-host 179.214.2.3 --ssh-user yanhe --ssh-password '...'

# 无 SSH 时手工写入 / 从 JSON 导入
autoeval-planner resource set --name 179.214.2.3 --cpu 72 --mem-mb 744391 \
  --disk-gb 3279 --network-available 1024
autoeval-planner resource import --file snapshot.json

# Harbor 资源：本地 task 目录或 GitHub
autoeval-planner resource import-harbor --dataset-key harbor_deepswe-v1.1 \
  --github datacurve-ai/deep-swe@main
autoeval-planner resource import-harbor --dataset-key harbor_deepswe-v1.1 \
  --tasks-dir /data/deep-swe/tasks

autoeval-planner resource list
autoeval-planner resource show --dataset-key harbor_deepswe-v1.1
```

当前 `179.214.2.3` 快照（SSH 实测）：**72 核 / 744,391 MB / 3,279 GB / 可用网段 1024**。
DeepSWE 资源来自 GitHub `datacurve-ai/deep-swe@main` 的 113 个 `task.toml`：每个 trial **2 核 / 8192 MB / 20480 MB 存储 / 1 个网段**；verifier 环境同为 2C/8G，独立顺序执行，不叠加。

### 9.3 对规划的影响

- 资源是硬约束：CPU/内存/磁盘/网段任一不足都不能启动新 case；服务吞吐是另一条独立约束；
- 默认 headroom=0.2：可用 CPU 为 57.6 核；扣除各任务基础资源后，DeepSWE trial 并发上限为 **27**（内存、磁盘、网段都不紧）；
- 同一套任务：`infinite` 为 2h19m44s，`server` 为 **3h38m26s**；峰值 CPU 56.1/72 核（约 78%），**保留约 22% 物理裕量**，不会被拉满；
- 输出中可见 `resource_assumption`、`resources.peak`、`resources.binding_resource`、`resources.max_feasible_case_concurrency` 与每任务资源画像。

### 9.4 口径与局限

- `headroom` 默认 0.2，CPU/内存/磁盘按 80% 计；网段直接用脚本给出的可用数；需要更多/更少裕量时调整该值（0–0.9）；
- verifier 资源记录但不叠加（Harbor `environment_mode=separate`，验证在 agent 之后）；
- 磁盘按 `storage_mb` 计入准入，overlay 共享镜像层可能使实际占用更低，属于保守估计；
- 服务器快照不自动刷新，超过 7 天会在 `warnings` 中提示。

## 10. 与 GPQA 实测的回归校验

历史 run：198 case、2,785,651 输出 token、wall 2,518 s、`batch_size=8`（有效吞吐 ≈ 1,106 tok/s）。

用实测曲线在 `case_concurrency=8` 下仿真 GPQA，得到约 2,892 s（+14.8%），与实测吻合；剩余差异主要来自长尾 case 与 TTFT 的近似。

## 11. 常见问题

| 现象 | 处理 |
| --- | --- |
| `未找到 predictions/...` | `--source` 指向 AISBench 时间戳目录或其父目录 |
| 规划时数据集被判为 `estimate` | 说明 `dataset_aliases` 中没有该别名，先 `dataset import --alias ...` |
| case 并发等于 case 数 | 说明服务并发上限高于 case 数，并发已被截断，属正常结果 |
| Harbor 的 `n_trials` 小于数据集 case 数 | `n_trials` 只统计拿到 verifier reward 的 trial；异常 trial 不计入，规划以唯一 task 数为准 |
| `没有可行的并发组合` | 检查 `batch_size` / `n_concurrent_trials` 与吞吐曲线的 `max_concurrency` |
| `缺少 API key` | `probe` 需要 `--api-key` 或 `--api-key-env` |
| `未找到服务器资源快照` | 先执行 `resource probe` 或 `resource set`，名字与 `resources.server` 保持一致 |
| 告警「资源受限：cpu 达到容量上限」 | 说明规划被 CPU 卡住；可降低 `headroom`、减少并发或增加机器 |
| `单个 case 需要 ... 超过服务器可用容量` | 调整 `resource import-harbor` 画像或服务器快照，单 trial 资源大于整机时无法规划 |
| 时长明显偏大 | 确认历史数据是否覆盖该数据集，或检查估算的 `output_tokens_median`、`overhead_sec` |

## 12. 测试

```bash
python -m pytest -q
```

覆盖：配置校验、AISBench / Harbor / unified_result 导入与幂等、按 case_id 聚合、别名与 case 选择、case_count 覆盖、仿真并发/开销行为、寻优约束（声明并发与 case 数截断）、吞吐曲线解析、探测（流式/非流式/自动回退/自定义头与请求体/usage 缺失估算，基于本地假服务）、服务器资源（快照解析、headroom、资源画像优先级、CPU/内存/网段准入、资源并发截断）、CLI 端到端。
