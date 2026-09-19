# Bragi Asclepius 最新主線：訓練不中斷與證據優先

這份 reference 是本 Skill 的「最新工作基準」。它不是 Bragi 模型的使用說明，也不是把 Bragi 權重拿來訓練 ISEEU；它只萃取 Bragi／Asclepius 主線中已被實際操作、查檔、跑 log 或讀 checkpoint 證明過的工作方法。

## 來源與優先級

目前時間軸以 Bragi Asclepius 最新主線為準。ISEEU OCR V1、V2、V2.5 的退休 session 仍然保留，作為歷史判斷、失敗案例與資料路徑的來源，但不能覆蓋較新的狀態。

來源 session 與實際檔案必須分開看：

- 最新 Bragi／Asclepius 主線的可追溯 session：`/Users/norikaoda/.claude/projects/-Users-norikaoda/325f2601-d60a-42db-89f8-f864da98d78f.jsonl`
- Bragi 方法與早期救援文件：`/Users/norikaoda/Dropbox/My project/Asclepius/Asclepius_conversation.md`、`/Users/norikaoda/.claude/projects/-Users-norikaoda/memory/bragi_rescue_handoff.md`
- ISEEU 訓練程式：`/Users/norikaoda/Dropbox/iseeu-llm-git/EXTRA-tools-iseeu/learned_ocr/train_iseeu_line_ctc.py`
- ISEEU v6 資料與輸出：`/Users/norikaoda/Dropbox/iseeu-newdrive-mirror/card/ocr-trainset-v5/`

若 session 文字、log、checkpoint metadata、產品評測互相矛盾，採以下優先級：

`可載入 artifact／checkpoint metadata > raw log > 可重跑命令輸出 > session 回報 > 記憶或推測`

session 說「啟動」只代表啟動；session 說「完成」也要用 artifact、exit code、eval report 和 hash 再核對。沒有 product gate 的 checkpoint，不得稱為產品完成。

## 最新 ISEEU 狀態：只記錄已證實的部分

Bragi 主線最後明確啟動了 ISEEU OCR v6：

- 方法：CTC line recognizer，從零訓練，不繼承 Apple 偽標註 checkpoint。
- device：MPS。
- train：96,409 rows。
- valid：9,951 rows。
- vocab：2,613 chars。
- parameters：1,407,350。
- 訓練設定：16 epochs、batch size 64、learning rate 0.001、seed `20260914`。
- 資料來源包含程式生成的精確文字與 bbox；常用漢字曝光資料降低了字表缺口，但這仍然是資料／model-only 證據。

實際讀取 `artifacts/recog-v6.pt` 的 checkpoint metadata 後，已落地到 epoch 12：

- valid exact：`67.9228%`
- valid CER：`0.116795`
- numeric subset exact：`57.2407%`
- numeric subset CER：`0.099012`
- epoch 10 曾達 valid CER `0.121823`，epoch 11 短暫回升至 `0.123950`，epoch 12 再改善；因此保留 best 與 last，不能只看最後一輪。

`train_v6.log` 後來出現 epoch 13 的 batch log，但沒有完整 epoch-13 eval record；所以 epoch 13 只能標成 partial，不能拿來與 epoch 12 比較。`v6-cont-20260919` 的 durable run 目前是 `queued`，其 `STATUS.md` 要求先做 preflight、manifest fingerprint 與 smoke；這不是已經續訓。

目前仍缺：

- 真實名片／App 輸入的 product-level end-to-end 評測。
- 與 Apple 使用同一把尺、同一分母的完整比較。
- detector 與 recognizer 串接後的 sealed holdout gate。
- v6 續訓的 preflight、smoke、checkpoint 與 product eval。

因此正確報告是「v6 已形成可載入的 model-only candidate」，不是「ISEEU 已贏 Apple」或「已可出貨」。

## Bragi 主線帶出的可重用工作法

### 1. 先說真實目標，再說能承諾的事

使用者要求「一路做到贏過基準」時，不能保證尚未驗證的結果。可以承諾的是持續推進、每一輪留下證據、失敗後改變一個可辨識的假設，並在沒有達標前不把候選稱為完成。

### 2. 先查正本與真實路徑

Dropbox 搬移曾讓舊命令指向不存在的 `/Users/norikaoda/iseeu-llm-git` 路徑。正確做法是：先 `find` 定位實際腳本，再讀檔確認 import 與資料路徑，最後才啟動。路徑修正後要重新跑 smoke；不能因為 Python process 出現就假設它使用正確腳本。

### 3. 每個漂亮結果都主動找反證

Bragi 主線反覆使用這個循環：

`提出解釋 → 找對立樣本 → 用獨立 evaluator 重算 → 若推翻就立刻收回 → 把新規則落盤`

實例包括：

- Apple accurate 與 fast／ja-JP 的差異先被誤讀，重新窮舉後改成同一把尺的 `82.386%` 字元覆蓋基準。
- 合成資料 100% classifier accuracy 被識別為紅旗，換到真實 OCR 文字後只有 `64.2%`；問題是固定模板，不是模型已學會欄位本質。
- synthetic valid 變好，但真實名片輸出幾乎固定成單一字元「ア」，於是停止 v1，不把漂亮的 synthetic loss 當成功。
- 5000 張合成圖的膚色代理篩選先被當成乾淨度證據，抽樣發現它抓不到版面重疊；判準改回「GT 對應文字是否仍可讀」，不再用背景美觀代替 OCR 品質。

### 4. 把瓶頸拆開再訓練

ISEEU 的完整鏈路至少拆為：名片／文字 detector、line recognizer、欄位選擇／pack、runtime policy。recognizer 不能解決 detector 沒框出的字，欄位規則也不能被算成 OCR 模型能力。每一層都要有獨立 metric，再跑端到端 product gate。

### 5. 合成資料的正確角色

合成資料可以解開人工標註死結，尤其在 renderer 當下保存精確文字與實際 bbox；但它先證明的是管線和 representation，不是產品泛化。必須同時檢查：

- 文字分布是否貼近真實資料，而不是固定模板。
- 字型、模糊、透視、光照、JPEG、遮擋是否覆蓋實際輸入。
- GT bbox 是否真的對應到成品圖，而不是只存在於理想 layout。
- synthetic valid 與 real crop／real page／App input 是否分開報告。

### 6. 長跑要「自動接棒」，不是只開背景 process

把工作拆為 durable jobs：

`preflight → smoke → render／label → train rung → checkpoint eval → real eval → promote/reject → next rung`

每個 job 都要有 `run_id`、host、device、lock、PID、heartbeat、輸入 fingerprint、輸出位置、retry 次數、狀態與停止原因。watcher 不能只看 PID；要看 log 是否前進、checkpoint 是否更新、artifact 是否可載入、eval 是否完成。

若一個 job 因缺路徑、資料、權限或遠端節點阻塞，標成 `blocked`，繼續不相依的安全工作。retry 必須記錄改了什麼；不要無限重跑同一個壞命令。

### 7. Session 交接靠落盤，不靠記憶

每次交接至少更新：

- `STATUS.md`：現在狀態、最後驗證時間、device、checkpoint、指標。
- `NEXT_ACTIONS.md`：下一個可直接執行的步驟、前置條件與 blocker。
- `manifest.json`：資料 fingerprint、config hash、vocab hash、seed、gate。
- `failure-bank/`：失敗輸出與判定原因。

訊息是通知，檔案才是交接正本。下一個 session 先讀檔、驗 artifact，再讀聊天；不能只複述上一個 session 的「我已完成」。

## 每輪的最小回報格式

```text
Run ID / phase / status
Host / device / PID / lock
Dataset rows / fingerprint / leakage status
Current checkpoint / SHA-256 / epoch / step
Train loss / valid metrics / real metrics / sealed metrics
Product gate: pass | fail | pending
Active artifact / next action / blocker
Evidence class: verified | read-from-file | reported | hypothesis
```

最後一行不可省略。它迫使報告者把「親自驗證」「讀檔所得」「他人回報」「尚未驗證的推測」分開，避免最新進度又被舊 session 或漂亮數字覆蓋。
