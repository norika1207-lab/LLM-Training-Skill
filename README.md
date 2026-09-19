# LLM Training Operations Skill

這是一套用於長時間、可恢復、可驗證模型訓練的 Codex Skill。

核心方法包括：

- 固定評估規則、至少三個 seed 與 honest metrics
- 資料清洗、leakage 檢查、synthetic-to-real 驗證
- smoke test、分階段訓練、checkpoint、resume 與 artifact promotion
- teacher/student 交接、獨立驗證與完整 handoff
- durable queue、watchdog、資源鎖、retry、heartbeat 與持續排程

主要入口是 [`SKILL.md`](SKILL.md)。四個 Claude session 的逐段萃取在
[`references/claude-session-training-methods.md`](references/claude-session-training-methods.md)。
