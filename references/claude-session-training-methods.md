# Claude 四個訓練 Session 的可重用方法

這份文件是從四個 Claude session 的原始 JSONL 逐段讀取後整理出的操作規則，不是從摘要反推。它把「如何把模型穩定訓練出來」拆成實驗設計、資料品質、老師／學生、長跑排程、驗收和交接六個面向。最新的 Bragi／Asclepius 狀態與 ISEEU v6 checkpoint 以同目錄的 `bragi-asclepius-latest.md` 為準；本文件中的退休 session 主要提供歷史脈絡與失敗證據。

## 來源與證據邊界

讀取的原始 session（原始 JSONL 僅存在萃取者的本機工作環境，未隨 repo 發佈）：

- `LLM 研究學者`：1,461 行。
- `Bragi老師`：3,054 行。
- `ISEEU OCR V1 老師 Retire`：11,279 行。
- `ISEEU OCR V2.5 學生Rertire`：9,981 行。

原始對話裡有些數字是 session 讀磁碟後親自跑出的，有些是交接轉述，有些是當時尚未驗證的假設。使用本文件時仍要保留三個標記：

- `親驗`：有命令、raw log、artifact 或磁碟檔可以重跑或核對。
- `讀檔`：讀過程式、報告或 JSON，但這一輪沒有重新執行。
- `轉述／假設`：來自另一個 session 的說法，不能直接當成模型成果。

### 最新主線的身份邊界

Bragi Asclepius 是目前最新的工作方法與訓練時間軸；ISEEU 是實際產品訓練目標。不要把 Bragi 的 1.5B coding model、量化、剪層或抗排斥藥成果寫成 ISEEU 的 checkpoint。ISEEU v6 的 model-only candidate、real-card evaluator 和 product gate 要各自記錄。

歷史數字只作案例，不能直接當新專案門檻。新專案必須用自己的 sealed holdout、資料 fingerprint、設定 hash 和同一套 evaluator 重建基準。

## 一套穩定訓練的總流程

### 0. 先定義「要訓什麼」與「不准混什麼」

先把產品任務拆成層次：模型權重、模型手術、推論 runtime、規則／工具、產品鏈。每一層單獨定義輸入、輸出、資料分母、指標和 gate。`runtime 92%`、`OCR-visible 100%`、`synthetic valid 83%`、`真實照片 27%` 不能放在同一欄，也不能用包裝層成果冒充裸模型能力。

任務契約至少要寫：

1. base model 與要保留的能力。
2. trainable／frozen 層，方法是 SFT、LoRA、distill、ranker、classifier 或 surgery。
3. train、valid、hard、realistic、sealed holdout 的來源和 fingerprint。
4. 裝置、記憶體、預計時間、checkpoint 頻率和可接受中斷損失。
5. promotion gate、最大回歸、停止條件、下一個 fallback。

### 1. 先校準尺，再做模型比較

ISEEU V2.5 的關鍵教訓是：只用 37 張 valid 挑 best checkpoint，在不同平台和不同 seed 會選出不同輪次；Mac 選第 24 輪、GX10 選第 16 輪，test 差距大於 seed 波動。修正順序是：

1. 先讀訓練腳本，確認實際 LR、batch、epoch、seed、optimizer、loss、frozen layers、資料路徑和選模邏輯。
2. 固定評估資料、固定 evaluator、固定比較規則。
3. 每個設定至少跑 3 個 seed；報 median、全距或 confidence interval，不報單次最佳。
4. 若要判定 A 比 B，先寫規則再看數字。例如兩組 seed 全距完全不重疊且方向一致才稱為有效差異；否則只能寫「在目前噪音下分不出來」。
5. 所有 threshold 必須在看結果前寫入規則檔，禁止事後選一個有利門檻。

Bragi Q3／Q5 的做法把這條規則推到量化：同機、同一 evaluator、同參數，各跑 5 到 8 次；只有 Q5 區間下緣高於 Q3 區間上緣才稱為穩定較優。區間重疊時，正確結論是「準度打平或可能略好，但 Q5 更快／更小」，不能寫成 Q5 更準。

### 2. 先讀資料，不要用模型掩蓋資料病

資料 preflight 不是數 row 而已，要做：schema、空值、重複、文件級 split、train/eval overlap、來源、欄位可見性、role contamination、標籤品質、難例比例和資料分布。

必要的資料分流：

- `clean train`：可以訓練的資料。
- `hard bank`：模糊、透視、反光、手繪、長尾欄位等困難樣本。
- `failure bank`：不可見、錯標、解析失敗、teacher hallucination、role 污染。
- `sealed holdout`：永不進訓練，也不拿來反覆調參。
- `needs-review`：不能靠規則自動決定的灰區。

ISEEU V1 發現舊 mild 資料與評測集重疊 5,063 行；後來把 `strict231` 和 `nonstrict269` 的 500 個文件當作 eval universe，用同一個 doc key 審計所有 train manifest，重新建立無重疊的 mild+hard 訓練集。ISEEU V2.5 又發現網路名片素材裡有大量設計範本與佔位文字，因此 447 張 teacher 標籤先經逐條規則清洗，再抽樣開圖覆核，最後保留 clean 265 張與 strict 244 張兩個可回溯版本。

規則不能代替抽驗。每一輪清洗都要抽驗「被剔除」和「被保留」兩邊，並保留剔除理由；否則清洗本身可能製造新的偏差。

### 3. 先 smoke，再把訊號逐級放大

訓練順序固定為：

`小資料 smoke → balanced subset → 低成本 rung → 完整資料 → 真實／sealed gate → deployment parity`

Smoke 必須在短時間內證明：資料能讀、vocab／tokenizer 穩定、forward／backward 成功、第一個 batch 有輸出、checkpoint 能載、evaluator 能跑、artifact metadata 正確。不要讓第一個可見結果在幾小時後才出現。

一個 smoke 只改一個主要變因。若同時改資料、模型、長度、device 和 evaluator，結果不能歸因。長序列、長 JSON、全量 VLM 標籤先縮短 input/output 或改 field-query、classifier、candidate-ranker；先讓 b1／b10 checkpoint 落地，再決定是否放大。

### 4. Checkpoint 不只是保存權重，而是訓練的決策節點

每個 checkpoint 要保存：權重、optimizer、scheduler、global step、epoch、config hash、資料 fingerprint、vocab hash、seed、device、程式版本、最近 loss、valid 指標、最佳指標和產生時間。checkpoint 用暫存檔寫完後 atomic rename，避免半檔冒充成品。

checkpoint 節奏依風險調整：

- 第一次 smoke：每 1 到 10 batch。
- 慢速或外接碟：每 10 到 20 batch，直到確認穩定。
- 穩定長跑：每 50 到 100 batch，並保留最近幾個與最佳幾個。
- 每出一個可載入 checkpoint，就先跑小型 eval；不要盲等整個 epoch。

ISEEU V1 的經驗是：訓練在 MPS／外接碟上慢時，先確認最後 batch、檔案 mtime、CPU／GPU、磁碟寫入和 checkpoint 是否健康，再決定切 CPU 或搬到 SSD。不是看到慢就殺，也不是看到 PID 還在就宣稱健康。

### 5. Resume 必須是可驗證的 resume

可續跑不是「PID 還在」或「重新下同一條命令」。resume 前必須核對：base model、資料 fingerprint、vocab／tokenizer hash、config hash、optimizer 相容性、runtime 和 device 假設。核對通過後，先載最新 valid checkpoint，用固定小 batch 做 deterministic load test，再接續完整訓練。

渲染、標註和訓練都要逐步落盤：

- 渲染每完成一個 document 就 append manifest，啟動時跳過已完成 document。
- 標註每張圖保存 raw response、解析結果、done reason、錯誤原因和重試次數。
- HTTP 400、空圖、timeout 或壞輸出隔離成 error record，不能讓一筆壞資料殺死全量工作。
- 訓練從最新可載 checkpoint 接續；完成後寫 `DONE` marker；已完成 job 不重跑。
- checkpoint、manifest、log、模型和報告都要有 hash。

ISEEU V1 曾因「最後才一次寫 manifest」在中斷後損失 38,436 張已渲染圖片，修正為逐筆 append 後，重跑 `resume.sh` 可以跳過已完成 doc。這是必須直接寫進所有長跑 pipeline 的基本機制。

### 6. Teacher／student 要先驗證 teacher 和標籤，再談蒸餾

Teacher 不是神諭。先驗證 teacher 在目標分布是否真的比 baseline 好，尤其不能用 teacher 的弱項當學生終點。

ISEEU V2.5 的 VLM teacher 流程：

1. 用 targeted per-field prompt，而不是一次要求整張名片自由生成；每個欄位要求 `value`、`visible`、`evidence`，不可見就回 `null`，降低幻覺。
2. Phase 1 先用 20 張測輸出格式，acceptance 設 100% JSON parse；19/20 時不放水，追查失敗。後來發現是 raw output 被截斷，原因是 `num_predict` 太小，於是保留 raw、調大上限並重測。
3. Phase 2 全量標籤時做斷點、重試、每欄 fill 統計和 HTTP error catch；447 張中 438 張 parse 成功、9 張隔離，不讓一筆壞圖使全程中斷。
4. Phase 2 不只看 fill rate，還抽 5 到 15 張開圖親驗。發現擺拍範本、佔位文字和假號碼污染素材，進 Phase 3 清洗。
5. Phase 4 才選 student 和蒸餾方案。若完整 VLM 無法達到大小約束，要把任務拆成成熟 OCR + 小欄位 head，而不是為了 50MB 故事硬砍不相容的 vision encoder。

Bragi 的 teacher／student 傳承也有一條非技術規則：學生可讀、可核對、可提問，但沒有授權就不能直接改機器、殺 process 或重跑同一實驗。這避免多 session 互撞資源，也避免學生把片段觀察誤當成已驗證知識。

### 7. 合成資料只能先證明管線，不能代替真實分布

ISEEU V1 的完整失敗／修正鏈是重要範本：

1. 1,127 行真實文字渲染成 4,484 張合成 line，第一顆 3.9MB recognizer 的 valid exact 從 0% 升到 15.1%，CER 從 100% 降到 50.9%。這證明管線能學，不代表產品能讀真實照片。
2. 擴成 26,124 行、11 種字型、26,058 張合成樣本後，synthetic valid exact 82.9%、CER 4.2%。
3. 一上 64 張真實 held-out，containment 只有 0.9%、CER 101.9%。隔離後發現 recognizer 只吐合成資料的高頻先驗字，合成到真實完全崩潰。
4. Apple Vision 在同批真實圖達 63.4%，證明任務可讀，問題是模型 domain gap，不是資料不可讀。
5. 因此加入真實 line crop + 文字配對，混入合成資料重訓。真實資料比例從 26% 提到 57% 後，真實 crop CER 50.4% 降到 27.5%，exact 14.7% 升到 37.5%。這些分數仍是對 teacher 標籤，不是真值準確率，報告必須明標。

結論：synthetic valid 是 smoke／representation gate；真實 crop、真實頁面、產品輸入才是 promotion gate。兩者不可互換。

### 8. 評估要隔離瓶頸，避免端到端一個低分掩蓋原因

至少拆成：

1. recognizer／student model-only：乾淨真實 crop 對文字。
2. detector：真實圖是否切到正確位置，檢查漏框與過切。
3. end-to-end：完整圖進，完整輸出出。
4. runtime／product：實際封裝、延遲、記憶體、offline、錯誤處理。

ISEEU V1 的 1.6% end-to-end 看似模型沒長，隔離後發現收據 detector 不適合雜誌自由版面；理想切行後 4.8%，再直接測 recognizer，epoch 3 的真實 crop exact 11%、CER 61%。這才知道 recognizer 正在成長，而 detector 是另一個瓶頸。訓練報告不得把兩個瓶頸混成一句「模型沒效」。

### 9. 特殊模型手術要用任務指標驗收

LLM 研究學者與 Asclepius 線提供的模型手術規則：

- 剪層先建立完整模型和每一層的 tolerance curve，再選手術點；不能從 28 層直接跳到 21 層，把中間可恢復區間誤判成斷崖。
- 剪層後的恢復可先只訓 RMSNorm scale，約 84K trainable parameters，不新增 LoRA／adapter；但是否有效要看任務 pass@1、逐題 flip 和 paired test，不可只看 KL。
- 27L 的 KL 持續下降，但 pass@1 在 step 200 後反而下滑，證明 KL 不能取代任務指標。
- 「抗排斥藥」結果要用 paired evaluation／McNemar 或其他事先指定的統計方法驗證；不能因一次漂亮分數就宣稱恢復成功。
- 保留 baseline、pruned、healed、random-control 和 rejected checkpoint；只留最佳 checkpoint 會丟失訓練曲線與選擇偏差。

### 10. 結果是負的，也要讓 pipeline 往前走

每一輪結束後只允許四種決策：

- `continue`：loss、valid、real gate 都按預期改善。
- `pivot`：資料、輸出形式、teacher 或 runtime 才是瓶頸，改一個主要假設開新 rung。
- `reject`：checkpoint 留存但不接 active；報明確 gate 失敗。
- `blocked`：缺資料、權限、硬體或 owner 決策；暫停該 job，繼續獨立安全工作。

ISEEU V2.5 的過夜流程在主要驗證收斂且 owner 未回覆時，沒有為了「持續」無限調 regex，而是寫 `A_PLAN_BLUEPRINT.md`，把已驗證數字、素材 hash、待改清單、工時和決策點收斂成下一棒可直接執行的文件。若剩下工作邊際價值很低，就進入 `noop` 待命，不製造假進展。

## 自動排程與不間斷執行

### Durable queue

把任務拆成可重啟 job，而不是一個長 shell：

`preflight → smoke → label/render → train rung → checkpoint eval → data-quality review → promote/reject → next rung`

每個 job 要有 `queued/running/checkpointed/evaluating/promoted/rejected/blocked` 狀態，並記錄 `run_id`、PID、host、device、輸入／輸出、lock、last heartbeat、last checkpoint、retry 次數、錯誤原因和下一步。

### Worker、watchdog、fallback

每台 GPU 只允許一個 heavy job；資料清理和報告可並行，但不能讓 VLM、訓練和大規模 rsync 搶同一個 RAM／磁碟。watchdog 不只看 PID，還要讀 last batch、checkpoint mtime、log 是否前進、device utilization、exit code、artifact 是否能載入。

ISEEU V1 的實作經驗是把高密度 VLM 標註搬到 RTX 2070 worker：冷啟動約 85 秒，但熱啟動每張約 4.5 到 5.8 秒；因此 GPU 應優先給 VLM 標註這種 GPU-bound 工作，小 recognizer 可能仍被 PNG 解碼和磁碟 I/O 限制。資源分配要看實測 bottleneck，不要看到 GPU 就把所有工作搬過去。

過夜或跨 session 工作要有兩層喚醒：

1. 主訊號：job 完成／失敗時通知。
2. fallback：固定間隔只檢查狀態，主訊號失效時接續。

健康且無變化時不要刷訊息；只有完成、失敗、卡住、資源衝突或需要 owner 決策才通知。等待不能被當成工作，若 session 不會自動醒，就必須用真實的外部 worker、heartbeat 或 automation，並在 60 到 65 秒後驗證心跳時間戳確實更新。

### 自動接棒的安全規則

- render 完成後才啟動 train；train 完成後才啟動 eval；eval 通過後才 promote。
- 上一階段失敗時，保留 log 和 partial artifact，先執行不相依的 job，不讓整條 queue 靜默停住。
- retry 必須說明變更了什麼：device、batch、input length、資料清洗、objective、依賴或 timeout；不能無限重跑同一個壞設定。
- 跨機器任務用 self-contained bundle、相對路徑、明確命令、輸出 schema、checksum 和回傳位置；不要依賴 live session 還活著。
- 自動化不得繞過資料 gate、權限、服務中的 GPU 或 owner 的產品方向決策。

## 交接格式

每輪 append 一份短狀態，不覆蓋舊紀錄：

```text
Run ID / phase / status
Device / host / PID / lock
Dataset rows / fingerprint / leakage result
Current checkpoint / hash / epoch / step
Train loss / valid / real / sealed metrics
Gate: pass or fail, exact reason
Artifact path / next action / blocker
```

交接文件要把「我親自跑的」「我讀檔知道的」「我從別人收到的」分欄。下一個 session 先讀交接和磁碟 artifact，再讀聊天；若 context 快滿，先把當前狀態、下一步、停止條件和 raw 路徑落盤，再交棒。傳承的目標不是讓下一個 session 相信前一個，而是讓它能用同一把尺獨立重驗。

## 絕對禁止的反模式

- 用 loss、KL、synthetic valid 或 single-run best checkpoint 宣稱模型成功。
- 用 `mtime`、PID 或一條「背景已啟動」訊息代替 checkpoint／artifact 實證。
- 以 Apple／teacher 產生的 noisy labels 當真值，卻不標註 teacher ceiling。
- 把 OCR 可見性不足、detector 漏框或範本污染塞回 student，讓模型學習錯誤答案。
- 新 vocab、不同資料 fingerprint 或不同 config 靜默 resume 舊 checkpoint。
- 多個 session 同時寫同一份狀態檔、搶同一張 GPU 或重跑同一個實驗。
- 為了看起來「持續工作」而在 owner 未決時做低價值調參，或把待命寫成有進展。
- 只保留最佳模型、刪掉 rejected／failed／random-control，讓未來無法知道結論如何形成。
