# 從「Mecury 服務中控」Session 萃取的訓練方法

這份參考不是把聊天摘要當成果，而是把 Session 中反覆出現、且有實際數字或檔案狀態支持的做法整理成可重用規則。數字是歷史案例，不是所有新專案的固定門檻。

## Session 反覆證明的模式

### 先做可觀測的小閉環

說明書／保証卡線先從 30 頁 smoke、再到 2,000 頁；先做 page-type classifier，再做 multi-label。第一版只餵 `text_preview` 得到 68.4%，查出關鍵詞可能在頁尾／中段後改用整頁文字。multi-label 的 micro F1=0.874、macro F1=0.849，但 exact label-set=0.441，於是沒有把單一 F1 寫成產品完成，而是轉成 curriculum 和下一輪 extractor。

同一條線的長 JSON seq2seq 在 1 epoch 得到 train loss=1.906、valid loss=0.560、3.97MB checkpoint，但 50 筆 eval 的 valid JSON=0、EOS=0。這個結果證明「有 loss／有 checkpoint」仍不代表輸出任務成立；之後改成短 field-query，並把長 JSON 生成降為非 hot path 實驗。

### 模型容量增加前，先測速度與訊號

收據 50MB student 以 34,552 train rows、500 realistic-hard holdout 和約 50.6MB fp32 參數配置起跑。CPU 上長序列 2-layer GRU 沒有及時 checkpoint，先停掉改短 input/output、較密 checkpoint，再試 29MB warm student。field-query 展開到 310,968 rows 後，仍發現 autoregressive decoder 太慢且輸出 collapse；先建 8,100 rows balanced subset，讓 b1 checkpoint 落地，再逐步改成 b50。

在 b50 eval 發現模型只吐 `<null>`，原因是 `unsupported_reason` 的資料分布主導了訓練。不是無限加 epoch，而是停掉舊 run、移除主導欄位，建立 8 個實際欄位各 1,000 筆的 no-null balanced set。b10 loss=4.10、b30 已有 checkpoint，但 balanced eval 到 b40／b70 仍為 0，於是判定 seq2seq 路線失敗，改成 candidate-ranker：先由 OCR 產候選，再由小模型選答案。

### 真正的訓練節奏是「checkpoint → eval → 決策」

名片 person v3 一度用 MPS、h192、3 epochs、每 20 batch checkpoint，但兩分鐘沒有 b20；檢查後切回 CPU。CPU 的 max input=768 仍沒有在合理時間落 checkpoint，再切成 input=512、每 10 batch checkpoint。此配置 b10 loss=4.19、b20=3.43、b30=3.04、b40=2.71、b50=2.45、b60=2.24、b70=2.08；epoch 1 checkpoint 一出就先跑 717 筆 holdout，而不是盲等三 epochs。

在較大 unified 版本，資料與 hidden 同時上調到 4,096 rows、hidden=384、embedding=160，總 231 batches。它從 b20 loss=3.298、b40=2.307、b60=1.769、b80=1.448、b100=1.231、b120=1.076、b140=0.960 持續下降；每個節點都先檢查 process、checkpoint、外接碟寫入，確認健康後才讓它跑完 epoch，再接 valid。這是「不要因為慢就亂切，也不要因為正在跑就盲等」的平衡。

### 資料錯誤優先修資料，不用模型掩蓋

teacher-distill 線先把 148 個 hard cards 分成 9 個「OCR／VLM 看得到但選錯」和 103 個「上游根本沒看見」。前者做 selector／runtime 修復，後者做 targeted image/VLM teacher；不能把缺失影像資訊硬灌進 OCR-only student。

ranker 重訓前，清查到 bbox 欄位混入 `"company"`、`"dept_title"` 等 role 字串。原始資料不覆蓋，另產 sanitized dataset：train 35,783 rows 修正 20,788 個壞 bbox，eval 2,477 rows 修正 992 個。這之後 person/company ranker 的合成 eval 可達 100%，但仍要跑產品 500 張回歸；如果接入 runtime 造成全域污染，就回退並保留 rejected artifact。

名片 v11 是以 152,513 rows 合併資料重訓指定欄位，保留已穩定的 contact 欄位；只有通過 v4 12,000、designer 6,900、grounded 500 三關才更新 pack。這種「只替換有證據改善的子模組」比整包重訓更穩。

### 輸出任務要選正確的學習形式

Session 中得到的穩定偏好是：

- 固定欄位抽取：field-query 或小型 classifier／ranker。
- 從候選中選正解：candidate-ranker，不讓 decoder 從字元空間盲猜。
- 長 JSON 生成：只有在輸出格式與 EOS 已驗證時才保留。
- 數字／稅額／日期：優先可見 OCR policy、候選排序與 verifier，並把 neural 與 policy 指標分開。
- 影像不可見欄位：先改善 OCR／VLM 上游，再做 teacher distillation。

### 這些結果不能混稱

Session 曾出現 88% 小樣本、79/100 穩定重測、500/500 synthetic、visible-only 100%、runtime exact、以及 real-photo 低分。它們各自回答不同問題。Skill 的硬規則是：每個結果都附上資料來源、分母、輸入可見性、是否 cache、是否經過 policy／verifier、執行環境與 artifact 路徑。
