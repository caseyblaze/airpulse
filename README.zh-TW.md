# airpulse

> **語言:** [English](README.md) | 繁體中文

> 一套建構於 **Dagster+（Serverless）** 之上的台灣空氣品質 **ML Pipeline**,每小時將
> 一個「無狀態的即時 API」轉化為能**自我累積、自我預測、自我監控**的系統。

`Dagster+ Serverless` · `PostgreSQL (Neon)` · `scikit-learn` · `pandas` · `pytest`

---

## 核心洞察

環境部（MOENV）的空氣品質是以**即時快照**形式發佈:每次呼叫只回傳「每個測站當下這一筆」
(約 84 個測站、單一時間戳),且**沒有歷史端點**——單次呼叫永遠無法構成時間序列。

airpulse 的每個設計決策都源自這項約束:

- 要做預測,pipeline 必須跨 run **累積自己的歷史** → 以冪等的 PostgreSQL 狀態層
  (`air_quality_history`)作為單一事實來源(source of truth)。
- 要誠實評估**未來**表現,模型需要**真正的時序切分**,而非隨機切分。
- 因為每小時的 run 都依賴前一次,**安全重跑與可靠性閘門**不是錦上添花——而是承重結構。

所以這不是「擷取 → 訓練」的玩具腳本,而是一套**圍繞著手上真實資料特性而設計的小型生產系統**。

## 架構概覽

單一每小時 job(`hourly_air_quality_pipeline`,cron `0 * * * *`,對齊 API 更新頻率)
materialize 一條由五個 asset 組成的 DAG:

```
raw_air_quality        (api, python)      ← GET MOENV aqx_p_432
        │
        ▼
cleaned_air_quality    (pandas)           ← 去除缺漏鍵、數值強制轉型、依時間排序
        │
        ▼
air_quality_history    (pandas, postgres) ← 冪等 upsert;回傳完整累積序列
        │
        ▼
model_predictions      (sklearn)          ← 防洩漏特徵 + 時序切分 → RF 預測 pm2.5
        │
        ▼
model_metrics          (postgres)         ← 整體 + 逐站 MAE/R²/相對誤差 + drift 旗標
```

清洗與驗證刻意**分離**:清洗只做結構化與型別轉換(數值缺漏轉為 `NaN`,絕不丟棄);至於
某個*值*是否合理、是否新鮮、是否漂移,則交給驗證層判斷。完整的 asset 職責與歷史表 schema
見 [`PIPELINE.zh-TW.md`](./PIPELINE.zh-TW.md)。

## 可靠性與容錯

因為每次 run 都建立在前一次之上,本 pipeline 的設計目標是「在公開 API 的不穩定中存活」與
「讓重跑安全」:

- **外部依賴重試**——ingestion asset 使用 Dagster `RetryPolicy`,以指數退避 + jitter
  (約 10s → 20s → 40s)重試,讓偶發 timeout 或 5xx 不會讓整個 run 失敗。
- **回應結構容錯**——MOENV v2 API 回傳的是「裸 JSON list」,並非文件所述的
  `{"records": [...]}` 包裝。解析器同時支援兩種結構,而非盲信 schema。
- **冪等 upsert → 安全重跑**——歷史以
  `INSERT ... ON CONFLICT (sitename, publishtime) DO NOTHING` 寫入。重跑某步或回填
  都不會插入重複列——這正是「從失敗點重跑」能**安全**而非危險的基礎。
- **冪等遷移**——加寬歷史表時用 `ADD COLUMN IF NOT EXISTS`,schema 變更不會遺失已累積的歷史。
- **Blocking 品質閘門 → 告警閉環**——`ERROR` 級 asset check 設為 `blocking=True`:壞資料
  或模型漂移會讓 run 失敗,再由 `run_failure_sensor` 轉成 **Slack** 告警。偵測 → 記錄 → 告警,
  錯誤不會無聲地往下游擴散。

## 資料治理

治理被當成一套**框架**而非事後補丁——設計上讓*同一條* pipeline 只需改**分類、而非架構**
即可延伸到受規管資料(完整內容見 [`DATA_GOVERNANCE.zh-TW.md`](./DATA_GOVERNANCE.zh-TW.md)):

- **分類**——每個 asset 標註 `data_classification` 標籤
  (`public` → `internal` → `confidential` → `restricted`)。現行 EPA 資料為 `public` 且
  無 PII;`restricted` 欄位則必須先通過單一受稽核的遮罩路徑才能儲存。
- **品質即 asset check**——六個 check 在每次 materialize 時執行,於 catalog 以通過／失敗
  呈現,並依嚴重度分級:`ERROR`(blocking,如 `raw_not_empty`、`pm25_non_negative`、
  `model_not_drifting`)vs `WARN`(僅標記,如 `aqi_in_range`、`data_is_fresh`)。
- **血緣、擁有權與 metadata**——assets 組成明確的 DAG;每個宣告
  `owners=["team:data-engineering"]`;`cleaned_air_quality` 發佈欄位層級 schema;每次 run
  皆輸出操作型 metadata(列數、測站數、來源 URL、模型指標)供 catalog 觀測。

## 嚴謹的 ML

建模層預測**各測站下一小時的 pm2.5**,而大部分工夫都花在「**不要自欺**」
(設計理由見 [`MODELING_PLAN.zh-TW.md`](./MODELING_PLAN.zh-TW.md)):

- **真正的時序切分**——測試集是最後 20% 的**時間點**;每個測站都同時出現在訓練與測試,
  差別只在測試是較晚的時間。此設計取代了早期的 `shuffle=False` 切分——後者在
  「先依測站、再依時間」排序的資料上,實際上是悄悄的**空間切分**(字母排序最後幾站),
  而非未來表現的評估。
- **防洩漏特徵(鐵律)**——外生訊號(其他污染物、氣象)**一律只用 lag 版本**,因為預測
  時點拿不到它們的當期值。只有**預測時已知**的特徵(時刻／星期／季節以環形 sin/cos 編碼;
  地理;county one-hot)才用同期值。另含自迴歸 lag 與由 lag 計算的滾動平均/標準差。
- **穩健補值**——只要求核心訊號(target + pm2.5 lags)非空;稀疏外生欄位保留 `NaN`,並在
  **切分後以訓練集中位數**補值。(此來自一個實際 bug:單一全缺的 EPA 欄位原本會把整個特徵
  矩陣清空。)
- **可比較的指標**——除 MAE/R² 外,以 `relative_mae = mae / mean(測試期實際 pm2.5)` 讓
  不同難易時段的 run 可互相比較,並提供**逐站**指標表。
- **Drift 作為治理閘門**——當相對誤差較**近 3 次 run 的中位數**惡化逾 15% 時,`drift_flag`
  觸發(用相對誤差,避免把「本來就難的時段」誤判為退化;用中位數,平滑單次雜訊)。它從一個
  默默寫入 DB 的欄位,提升為可呼叫 Slack 的 blocking `model_not_drifting` check。
- **誠實的冷啟動**——累積時間點不足 `MIN_TIMESTAMPS = 12` 時,模型回傳
  `status="insufficient_data"` 而非在雜訊上訓練;drift check 也回傳通過而非誤報。

## 這份專案展現了什麼

| 能力 | 具體證據 | 深入閱讀 |
| --- | --- | --- |
| **讀懂資料,而非讀懂文宣** | 看出快照 API 沒有歷史 → 圍繞「累積 + 預測」設計 | [PIPELINE §1.4](./PIPELINE.zh-TW.md) |
| **生產級可靠性** | RetryPolicy、回應結構容錯、冪等 upsert 與遷移、安全重跑 | [PIPELINE §1–2](./PIPELINE.zh-TW.md) |
| **監控與告警** | Blocking asset check → run 失敗 sensor → Slack 閉環 | [PIPELINE §3.5](./PIPELINE.zh-TW.md) |
| **資料治理** | 分類框架、品質閘門、血緣/擁有權/metadata | [DATA_GOVERNANCE](./DATA_GOVERNANCE.zh-TW.md) |
| **ML 評估嚴謹度** | 防洩漏特徵、真正時序切分、相對 MAE、逐站指標、drift 偵測 | [MODELING_PLAN](./MODELING_PLAN.zh-TW.md) |
| **工程判斷力** | 抓到並修正實際 bug(空間切分、特徵矩陣清空);記錄 v1/v2 取捨;對切分/冷啟動/drift 寫單元測試 | git 紀錄 · `tests/` |

## 技術棧

- **編排:** Dagster+ Serverless(PEX 部署),assets + asset checks + schedules + sensors
- **儲存:** Neon 上的 PostgreSQL(以冪等 upsert 作為單一事實來源)
- **ML:** scikit-learn `RandomForestRegressor`(跨站共用單一模型),pandas 特徵工程
- **告警:** 透過 run 失敗 sensor 的 Slack incoming webhook
- **工具:** uv / pip、pytest

## 快速開始

**安裝**(建議用 uv):

```bash
uv sync
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

<details>
<summary>pip 替代方案</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```
</details>

**本機啟動 Dagster**——接著開啟 http://localhost:3000:

```bash
dg dev
```

**執行測試:**

```bash
pytest
```

設定(Postgres 連線、`SLACK_WEBHOOK_URL`)透過環境變數;見 [`.env.example`](./.env.example)。
密鑰僅存於 Dagster+ 正式環境範圍,絕不進入 repo。

## 文件地圖

| 文件 | 內容 |
| --- | --- |
| [`PIPELINE.zh-TW.md`](./PIPELINE.zh-TW.md) | Pipeline 架構:ingestion、清洗、驗證層、ML 層 |
| [`DATA_GOVERNANCE.zh-TW.md`](./DATA_GOVERNANCE.zh-TW.md) | 分類、品質閘門、血緣、擁有權、roadmap |
| [`MODELING_PLAN.zh-TW.md`](./MODELING_PLAN.zh-TW.md) | ML 設計(已實作):切分、特徵、指標、drift、取捨與 v2 待辦 |

## Roadmap

- **建模 v2**——滾動訓練窗口／近期加權(讓預測能追上快速變化的空氣)、超參數調校、依特徵
  重要度修剪、多步預測(t+2、t+3),以及固定 baseline 的 drift 規格以抓季節性緩慢退化。
- **平台(Phase 2,GCP)**——歷史表保留/分區、DB 角色分離、密鑰輪替。
