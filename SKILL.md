---
name: mercury-model-training-ops
description: "Run reproducible local model-training work with staged experiments, durable checkpoints, honest evaluation gates, safe resume, and continuous worker scheduling. Use for SFT, LoRA, distillation, pruning, quantization, domain-specialist models, or long-running training that must keep progressing without losing work."
---

# Mercury 模型訓練作業系統

這個 Skill 把 Mercury「服務中控」Session 中反覆驗證過的訓練方法，和外部 MLOps／LLM research tooling 的成熟做法，固定成可重複執行的作業流程。目標不是把程序啟動，而是讓每一輪都有可追溯資料、可觀測進度、可恢復 checkpoint、獨立驗收、實驗比較、模型註冊與明確的升級／回退決策。

適用於本機或受控私有節點上的 SFT、LoRA、蒸餾、分類器／ranker、seq2seq、剪枝、量化、領域專家模型，以及需要跨小時或跨天持續執行的工作。除非使用者明確要求，不自動上雲、公開發佈、購買資源、刪除原始資料或改動正在服務的模型。

## 核心不變條件

1. 把四件事分開記帳：權重訓練、模型手術、推論編排、規則／快取包裝。wrapper 的 100% 不得寫成模型的 100%；synthetic、OCR-visible、cache-hit、runtime、真實 holdout 也不得混成一個分數。
2. 每一輪訓練都必須有 `run manifest`、資料 fingerprint、config hash、seed、硬體／runtime preflight、進度紀錄、checkpoint、eval report 和 artifact hash。缺任何一項，只能叫實驗，不得升級成產品成果。
3. 長跑先證明「能觀測、能存檔、能評估、能續跑」，再放大資料、序列長度、hidden size 或 epoch。不要讓第一個可見結果出現在幾小時之後。
4. loss 下降只是必要條件，不是成功條件。每個重要 checkpoint 都要跑 validation 或小型真實 eval；只有 sealed holdout／產品回歸通過才可 promote。
5. 失敗版本保留為證據，正式版本採 append-only timestamp 目錄。不得原地覆蓋上一個安全 artifact；先寫新版本、驗證、再更新指標或 symlink。
6. 只把「能被上游看見的內容」交給 student 學習。OCR／VLM 沒有提供的欄位不得硬餵給 OCR-only 模型；不可見、錯標或 role 污染資料要進 failure bank／needs-review，而不是靜默修成漂亮分數。
7. 外部 trainer、tracker、registry 都是 adapter，不是證據本身。遠端服務失效時必須仍保有 local JSONL、manifest、checkpoint、eval report 與 SHA-256；`latest`、`staging`、`production` alias 不能取代 immutable release manifest。
8. 研究分支必須有生命週期與下一步：`active`、`promising`、`plateaued`、`dead`、`superseded`、`promoted`。dead branch 不刪除，因為它是防止重犯的 evidence；被污染的 holdout 或 evaluator 不得 promotion。

## 每次訓練的標準閉環

### 1. 先寫任務契約與成功定義

先回答：要保留哪個能力、輸入／輸出 schema、模型大小與延遲上限、允許的本機節點、產品禁區、基準模型、sealed holdout、最低 gate，以及什麼情況必須回退。把「研究 prototype」「可重跑 artifact」「可部署 runtime」「可公開產品」分成四種狀態。

建立 run 目錄時可使用 `scripts/scaffold_run.py`；它會產生 `manifest.json`、`STATUS.md`、`queue.jsonl`、`checkpoints/`、`eval/`、`artifacts/`、`logs/`、`failure-bank/`。已有 run 不要重新 scaffold 覆蓋，改用新 timestamp 目錄或 resume。

若要使用外部 trainer、tracking、registry、evaluation 或 Skill 自我測試，先讀 [references/external-integrations.md](references/external-integrations.md)。它定義 config-driven adapter、local-first tracking、promotion 順序、branch lifecycle 與 anti-overfit 規則。

### 2. 做資料與硬體 preflight

資料方面先統計 rows／pages／images、schema 欄位、空值、重複、train/valid/test 分割、來源、可見性、標註品質與資料 fingerprint。針對多任務資料做均衡抽樣；至少準備一個小 smoke set、一個一般 valid set、一個按欄位或任務均衡的 eval set，以及一個不可在訓練中反覆查看的 sealed holdout。把 hard cases 和 failure bank 分開保存。

硬體方面確認實際使用的是 CPU、MPS、CUDA、VLM endpoint 還是錯誤的 fallback。檢查 `torch.cuda.is_available()`／MPS、GPU memory、RAM、磁碟、可用 Python／torch／driver、SSH／遠端節點與當前 active service。看到 GPU 不等於 Python 會用 GPU；GPU utilization=0 時不得把 CPU 長跑報成 GPU 訓練。先寫 preflight JSON。

### 3. 先 smoke，再逐級放大

推薦梯度如下：

`64–256 rows → 1k–8k balanced subset → full dataset → target capacity`

每級都先確認：資料能讀、vocab／tokenizer 穩定、forward/backward 成功、第一個 batch 有輸出、checkpoint 可載入、valid 可跑、metadata 名稱正確。對長序列／autoregressive JSON，先縮短 input/output 和 batch；對欄位抽取，優先考慮 field-query、分類器或 candidate-ranker，不要為了「像 LLM」硬用長 JSON decoder。

煙霧訓練的目的不是宣稱準確率，而是驗證管線。完成一個可載入 checkpoint 後，才進入完整資料或更大模型。

### 4. 讓長跑可觀測、可中斷、可續跑

至少要有：

- `progress.json` 或 JSONL：run id、PID、host、device、epoch、batch、loss、learning rate、elapsed、last checkpoint、最佳 valid、錯誤與心跳時間。
- 每 N batch 寫 checkpoint；N 必須讓一次中斷最多損失可接受的工作量。長序列或慢硬體可從每 1／10／20 batch 開始，確認穩定後再放寬到每 50／100 batch。
- stdout／stderr 持續寫 log，不以終端畫面存在作為唯一狀態。
- checkpoint 使用 temp 檔寫完後原子 rename；sidecar 同步記錄資料 fingerprint、vocab hash、config hash、git／程式版本與硬體。
- 用獨立 monitor／watcher 讀狀態，不要同時啟動兩個相同 GPU 重任務。每張 GPU 一個 lock；VLM、fullgrid、SFT 等重工作進 queue，依資源串行或安全並行。

若數分鐘沒有 epoch，不要只等，也不要直接殺掉：先查 PID、CPU/GPU、RSS、磁碟寫入、最後 checkpoint 和 stdout。只有在「無進度且超過該配置的合理門檻」時才停止，並保留 partial artifact 與診斷原因。

可用 `scripts/track_run.py` 將 heartbeat、metric、checkpoint 與狀態 append 到 `events.jsonl`，並同步更新 `progress.json`。這個 local event log 是 tracking backend 的最低保證；MLflow、W&B 或 TensorBoard 只能作為額外輸出。

### 5. 用證據選擇繼續、縮小或換路線

- loss 正常下降、checkpoint 持續落地、valid 同步改善：讓目前 rung 完成，再決定是否加 epoch／容量。
- loss 下降但 valid 不動：先查資料 leakage、輸出格式、欄位不均衡、teacher label、tokenizer／vocab，不要盲目加 epoch。
- 輸出塌成 `<null>`、固定日期、固定數字或無 EOS：檢查資料分布與 evaluator；重做 balanced subset／移除主導欄位，必要時把 seq2seq 改成 field-query 或 candidate-ranker。
- 只有單一小樣本、synthetic 或可見欄位變好：不得 promote；補 hard／realistic／sealed gate。
- 新模型改善 hard cases 但污染全域：回退 runtime 接入，保留模型與報告；改用 gated／field-level ablation，通過完整回歸後再接回。
- MPS／GPU 比 CPU 慢、沒有 checkpoint 或 device fallback：先切到可觀測的 CPU／短序列配置，保留原 run 的狀態，不把硬體問題誤報成模型失敗。
- 需要新字元或新 vocab：不可直接 `resume-from` 舊模型；要合併原始資料與新資料重新建 vocab 後從 base 重訓，或明確實作 vocab expansion 並另立 run。

### 6. 每個 checkpoint 都要分層驗收

最少跑四層，並在 report 中分欄：

1. `model-only`：固定輸入、固定 tokenizer、無 runtime 修補。
2. `runtime`：實際 tokenizer、decoder、schema、validator、ranker、policy。
3. `distribution`：真實／退化／跨版型／跨語言／跨設備輸入。
4. `product gate`：完整 App 或私有服務路徑，包括延遲、記憶體、封裝、offline、錯誤回報與 needs-review。

每一層記錄樣本數、分母、欄位／任務分數、exact、valid JSON、latency、cache 命中、失敗原因。基準回歸不過，禁止更新 active artifact。`validate_training_run.py` 可檢查 manifest、狀態、checkpoint、eval 和 artifact 是否具備，不替模型做虛假的品質判定。

### 7. Promote、回退與交接

通過 gate 後才建立 immutable release manifest：artifact 路徑、大小、SHA-256、資料／config／vocab hash、基準差異、已知限制、可重跑命令與下一步。用 `candidate → validated → promoted` 狀態流，不直接把最新檔案叫 final。

若回歸下降，保留 rejected artifact 和完整 diff，active 版本回到上一個 validated 版本。每輪結束更新 `STATUS.md` 和 `NEXT_ACTIONS.md`，讓下一個 session 能從檔案接手，而不是重新掃聊天記憶。

不要用複製檔案或改名冒充 promotion。使用 `scripts/promote_artifact.py` 時，必須提供有效 eval report、`decision=validated` 或 `promoted`，並明確傳入 `--confirm`；腳本會計算 SHA-256、建立 immutable release manifest，只有 `--decision promoted` 才更新 active pointer。

## 持續不中斷的工作隊列

當使用者要求持續訓練、跨夜執行或自動安排時，建立 durable queue，不把所有工作塞進一個不可觀測 shell：

`preflight → smoke → train rung → checkpoint eval → promote/reject → next rung`

每個 job 必須有 `queued/running/checkpointed/evaluating/promoted/rejected/blocked` 狀態、輸入／輸出路徑、資源需求、retry 次數、最後心跳和停止原因。watcher 的循環是：讀 queue → 取得資源 lock → 啟動一個可恢復 job → 週期性讀 progress → 完成後跑 eval → 依 gate promote 或 reject → 釋放 lock → 啟動下一個相容 job。

一個 job 被阻塞時，只標記它和原因（缺權限、缺模型、缺 GPU、gated download、資料錯誤），繼續跑安全且獨立的 jobs；不可為了維持「不中斷」而繞過 gate、猜測缺失資料或搶占正在服務的 GPU。若需要 Codex 在對話外持續監控，使用產品提供的 heartbeat／automation 機制，讓通知只在完成、失敗、狀態改變或需要使用者處置時發出；不要在狀態沒變時刷屏。

repo 內的 `scripts/continuous_worker.py` 是一個安全的本機最小 worker：它只消費 JSONL 中尚未執行的 `queued` job、建立 run-level lock、把 stdout/stderr 寫入 run log，並以 append-only event 記錄 `running`、`checkpointed` 或 `rejected`。它不自行 promotion，也不會因 queue 空就虛構工作；真正的 evaluator 與 promotion 仍由 gate 控制。

## 交接時的固定回報格式

回報先講狀態，再講證據：

`Run ID / 狀態 / device / dataset rows / current checkpoint / train loss / valid metrics / product gate / active artifact / next action / blocker`

禁止只回「還在跑」「已完成」「準確率很好」。若沒有 artifact、checkpoint、eval report 或 hash，就明確寫「尚未形成可驗收成果」。

需要理解本 Skill 的來源方法與具體案例時，讀 [references/extracted-mercury-method.md](references/extracted-mercury-method.md)；需要建立長跑 queue、狀態 schema、資源鎖與恢復策略時，讀 [references/continuous-runbook.md](references/continuous-runbook.md)。

需要接入 TRL、Axolotl、LLaMA-Factory、Unsloth、Accelerate、FSDP、DeepSpeed、MLflow、W&B、Model Registry、quantization 或 Skill 自我測試時，讀 [references/external-integrations.md](references/external-integrations.md)。

四個 Claude session「LLM 研究學者」「Bragi老師」「ISEEU OCR V1 老師 Retire」「ISEEU OCR V2.5 學生Rertire」的逐段訓練方法、證據、失敗案例、教師／學生交接、長跑排程與 watchdog 萃取，集中在 [references/claude-session-training-methods.md](references/claude-session-training-methods.md)。遇到模型訓練任務時，先讀該 reference，再依本 Skill 的 gate、checkpoint、queue 與交接格式執行。

目前最新時間軸以 [references/bragi-asclepius-latest.md](references/bragi-asclepius-latest.md) 為準：它明確區分 Bragi／Asclepius 的最新方法基準與 ISEEU 的實際訓練目標，並記錄最新 v6 checkpoint、實際路徑、續訓 queue 與尚未完成的 product gate。遇到 ISEEU 或長跑訓練任務時，先讀該 reference，再讀歷史 session reference，最後依本 Skill 的 gate、checkpoint、queue 與交接格式執行。Bragi 的模型與權重不可被誤當成 ISEEU 的模型成果。
