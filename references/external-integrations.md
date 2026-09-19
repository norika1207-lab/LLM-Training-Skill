# 外部最佳實踐整合指南

本文件把外部 GitHub 專案中值得採用的部分，轉成 Mercury 的可選 adapter，而不是把任何單一框架當成唯一正解。

## 整合原則

Mercury 是 control plane：負責任務契約、資料證據、長跑狀態、checkpoint、評估 gate、交接與 promotion。外部工具是 data plane：負責真正的 trainer、tracking backend、model registry 或 Hub transport。

任何外部整合都必須遵守三個邊界：

1. 外部 tracker 失效時，local JSONL／檔案證據仍然可用。
2. registry 或 Hub 的 alias 不能取代 immutable artifact、SHA-256 與 eval report。
3. framework adapter 可以改變訓練實作，但不能改變 model-only、runtime、distribution、product gate 的分層。

## 建議的 adapter 矩陣

| 層 | 首選 | 備選 | Mercury 必須保存的本地證據 |
| --- | --- | --- | --- |
| SFT／LoRA／DPO／GRPO | Transformers + TRL | Axolotl、LLaMA-Factory、Unsloth | command、版本、config hash、checkpoint、eval |
| Distributed | Accelerate | DDP、FSDP、DeepSpeed、Ray Train | world size、device、launcher、host、resume 設定 |
| Tracking | local JSONL | MLflow、W&B、TensorBoard | run id、metrics、artifacts、offline export |
| Registry | release-manifest | MLflow Model Registry、HF Hub | artifact SHA、來源 run、gate、promotion decision |
| Evaluation | 自訂 deterministic evaluator | lm-evaluation-harness、BigCode eval | suite id、版本、raw result、failure samples |
| Quantization | bitsandbytes／TorchAO | GPTQ、AWQ、HQQ、GGUF | base hash、quant config、同環境回歸 |

## Config-driven training contract

每個 trainer adapter 應接受一份可序列化設定，並在啟動前把完整設定寫入 run：

```yaml
run_id: receipt-sft-001
method: sft
seed: 17
data:
  train: data/train.jsonl
  valid: data/valid.jsonl
  sealed: data/sealed.jsonl
model:
  base: Qwen/Qwen2.5-1.5B
  output_dir: checkpoints
training:
  epochs: 3
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 0.00002
  mixed_precision: bf16
evaluation:
  suite: eval/suite.json
  every_checkpoint: true
promotion:
  minimum_valid: 0.80
  max_regression: 0.01
tracking:
  provider: local
  uri: logs/metrics.jsonl
```

實際上可以用 JSON 取代 YAML；重點是禁止只靠 shell command 的隱含預設。所有外部參數都必須進 `config_hash`，並和 checkpoint 一起保存。

## Tracking 的降級策略

先寫 `events.jsonl`、`progress.json` 與 `metrics.jsonl`，再嘗試 MLflow 或 W&B。遠端服務不可用時：

1. 不停止已經安全運作的訓練。
2. 把 tracking 狀態標記為 `degraded`。
3. 完成後以 offline artifact 匯入遠端 tracker。
4. 在 report 中保留服務錯誤與缺失欄位。

不可因為 dashboard 有一條曲線，就刪除原始 metrics 或宣稱資料已被追蹤。

## Model registry 與 promotion

registry 的 `latest`、`staging`、`production` 只是索引，不是品質證據。Mercury 的 promotion 順序是：

```text
candidate artifact
  → eval report with decision=validated
  → release-manifest.json with SHA-256
  → local active pointer
  → optional registry alias / Hub upload
```

如果 registry upload 成功但 local gate 失敗，狀態仍是 `rejected`。如果 local promotion 成功但遠端 upload 失敗，狀態是 `promoted_local`，不能誤稱為已發布。

## Framework selection

- 小型本機 SFT／LoRA：優先使用 Transformers + TRL 或現有專案 trainer，先確保 resume 和 evaluator 可接入。
- 需要很多 CLI recipe：可採用 Axolotl、LLaMA-Factory 或 Unsloth，但把它們包在 Mercury run contract 內。
- 需要 DPO／GRPO／distillation：先建立 teacher、filter、reward 或 preference 的 raw evidence，再呼叫 TRL 等 backend。
- 需要多機：先在單機 smoke 驗證 checkpoint 與 evaluator，再選 Accelerate、FSDP、DeepSpeed 或 Ray Train。
- 需要量化：先完成 FP／LoRA checkpoint 的 sealed gate，再做量化；量化版本要在同一 evaluator 和相同資料分層下重測。

## 借用的評估治理規則

每個 branch／run 都必須有 named next action，並可標記為 `active`、`promising`、`plateaued`、`dead`、`superseded` 或 `promoted`。死路徑不刪除，因為它們是避免重犯的 evidence。

Evaluator 不得讀取 hidden answer 來生成提示、修正輸出或挑選資料；不得用與 holdout 重複的 fixture；若發現污染，該 checkpoint 必須標記 `contaminated`，不可 promotion。

## Skill 自我驗證

除了測模型，也要測 Skill：同一個請求分別在「有 Skill」與「無 Skill」環境執行，檢查是否真的產生 run manifest、preflight、checkpoint、eval、hash、gate 與下一步。應包含：

- success case
- known-failure case
- regression case
- missing-permission／missing-GPU case
- 不應觸發本 Skill 的 negative case

可以使用 `agent-skill-eval` 等 harness，但 raw state delta 與命令輸出仍要保留在本地測試資料夾。

## 來源

這份指南的設計參考了 [MLOps-agent-skills](https://github.com/timwukp/MLOps-agent-skills)、[AI-research-SKILLs](https://github.com/firecrawl/AI-research-SKILLs)、[Hugging Face skills](https://github.com/huggingface/skills)、[agent-skill-eval](https://github.com/tardigrde/agent-skill-eval)、[Revolve 的評估規則](https://github.com/agent0ai/revolve/blob/main/AGENTS.md) 與 [Flatbuild](https://github.com/flatseek/flatbuild)。它們提供的是可借用的模式，不代表 Mercury 已經替它們驗證所有 backend。
