# airpulse Pipeline 架構說明

台北／台灣空氣品質 ML Pipeline，建構於 **Dagster+（Serverless）** 之上，每小時自
環境部（MOENV）開放資料 API 擷取空氣品質觀測值，經清洗、累積、建模後，將模型指標
寫入 PostgreSQL（Neon）。本文件說明三個核心部分:

1. [每小時 Ingestion Job 結構](#1-每小時-ingestion-job-結構)
2. [資料清洗方式](#2-資料清洗方式)
3. [Schema Validation Layer 結構](#3-schema-validation-layer-結構)

---

## 1. 每小時 Ingestion Job 結構

### 1.1 Job 與排程

| 項目 | 內容 |
| --- | --- |
| Job 名稱 | `hourly_air_quality_pipeline` |
| 排程 | `ScheduleDefinition`，cron `0 * * * *`（每小時整點，對齊 EPA API 更新頻率） |
| 入口 | `src/airpulse/defs/pipeline_defs.py` |
| 資源 | `postgres`（`PostgresResource`，連線 Neon） |
| 執行環境 | Dagster+ Serverless（PEX 快速部署） |

> Job 與排程皆為**每小時**一次，與資料源每小時更新頻率一致。

### 1.2 Asset 血緣（DAG）

整條 pipeline 由 5 個 asset 組成，依賴關係如下:

```
raw_air_quality            (group: raw)   ← EPA API 擷取
        │
        ▼
cleaned_air_quality        (group: clean) ← 清洗 + 型別轉換
        │
        ▼
air_quality_history        (group: clean) ← upsert 累積至 PostgreSQL，回傳完整歷史
        │
        ▼
model_predictions          (group: ml)    ← 由 PostgreSQL 讀歷史，建 lag 特徵預測 pm2.5
        │
        ▼
model_metrics              (group: ml)    ← 寫入 MAE / R² / drift 指標
```

### 1.3 各 Asset 職責

| Asset | Kind 徽章 | 職責摘要 |
| --- | --- | --- |
| `raw_air_quality` | api, python | GET `aqx_p_432`（limit 1000），回傳原始 DataFrame |
| `cleaned_air_quality` | pandas | 去除缺漏鍵、數值欄位強制轉型、時間排序 |
| `air_quality_history` | pandas, postgres | 將本次快照 upsert 進歷史表，回傳累積時間序列 |
| `model_predictions` | sklearn | 由歷史表建 lag 特徵，隨機森林預測 pm2.5 |
| `model_metrics` | postgres | 持久化模型指標並標記 drift |

### 1.4 Ingestion 細節（`raw_air_quality`）

- 資料源:`https://data.moenv.gov.tw/api/v2/aqx_p_432`（環境部即時空氣品質）。
- **回應結構容錯**:MOENV v2 API 回傳的是「裸 JSON list」，並非 `{"records": [...]}`
  包裝；程式同時支援兩種結構:

  ```python
  payload = resp.json()
  records = payload.get("records", []) if isinstance(payload, dict) else payload
  ```

- **外部依賴重試（RetryPolicy）**:API 偶發 timeout／5xx 時，以指數退避自動重試，
  避免單次抖動讓整個 run 失敗:

  ```python
  retry_policy=dg.RetryPolicy(
      max_retries=3,
      delay=10,                    # 約 10s → 20s → 40s
      backoff=dg.Backoff.EXPONENTIAL,
      jitter=dg.Jitter.PLUS_MINUS,
  )
  ```

- 輸出 metadata:`row_count`、`source_url`。

> **資料源特性（重要）**:此 API 為「即時快照」，每次呼叫每個測站僅回傳「當下這一
> 小時」一筆（約 84 個測站、單一時間戳）。因此單次擷取無法構成時間序列，需靠
> `air_quality_history` 跨 run 累積——這也是 pipeline 採「累積歷史 + 時序預測」設計
> 的根本原因。

---

## 2. 資料清洗方式

清洗分為兩個 asset:**`cleaned_air_quality`（單次快照清洗）** 與
**`air_quality_history`（跨 run 累積與正規化）**。

### 2.1 單次快照清洗（`cleaned_air_quality`）

輸入為 `raw_air_quality` 的原始 DataFrame，依序執行:

1. **去除缺漏鍵**:`dropna(subset=["sitename", "publishtime"])`——缺少測站名稱或
   時間戳的列無法作為有效觀測，直接丟棄。
2. **數值欄位強制轉型**:對下列欄位套用 `pd.to_numeric(errors="coerce")`，將
   非數值（如 EPA 以 `"--"` 表示的無資料）轉為 `NaN`:

   ```
   NUMERIC_COLS = ["pm2.5", "pm10", "o3", "co", "so2", "no2", "aqi"]
   ```

3. **時間欄位解析**:`pd.to_datetime(df["publishtime"])`（原始格式如
   `2026/06/05 22:00:00`）。
4. **排序**:依 `publishtime` 排序，確保時序一致。
5. **輸出 metadata**:`row_count`、`distinct_sites`，以及欄位層級的
   `dagster/column_schema`（見 §3.3）。

> 設計取捨:此層僅做「結構化與型別正規化」，**不丟棄數值缺漏的列**（數值缺漏轉為
> `NaN` 保留），把「值是否合理」的判斷交給下游的 validation layer，使清洗與驗證
> 職責分離。

### 2.2 累積與正規化（`air_quality_history`）

此 asset 是 pipeline 的**持久化狀態層**與單一事實來源（source of truth）:

1. **欄位名正規化**:將 `pm2.5` 改名為 `pm25`，使每個欄位都是合法 SQL 識別字:

   ```python
   COL_MAP = {"pm2.5": "pm25", "pm10": "pm10", "o3": "o3",
              "co": "co", "so2": "so2", "no2": "no2", "aqi": "aqi"}
   ```

2. **NaN → SQL NULL**:`snap.replace({np.nan: None})`，避免寫入時型別問題。
3. **冪等 Upsert**:以 `(sitename, publishtime)` 為主鍵，
   `INSERT ... ON CONFLICT (sitename, publishtime) DO NOTHING`——
   **重跑不會插入重複資料**，這也是「從失敗點重跑／step 重試」能安全運作的基礎。
4. **回傳完整累積序列**:`SELECT ... FROM air_quality_history` 回傳整張歷史表，
   供下游建模。
5. **輸出 metadata**:`snapshot_rows`（本次新增）、`total_history_rows`（累積總量）、
   `distinct_timestamps`（已累積的不同時間點數）。

#### 歷史表 Schema（`air_quality_history`）

```sql
CREATE TABLE IF NOT EXISTS air_quality_history (
    sitename     TEXT NOT NULL,
    county       TEXT,
    publishtime  TIMESTAMP NOT NULL,
    pm25 FLOAT, pm10 FLOAT, o3 FLOAT, co FLOAT, so2 FLOAT, no2 FLOAT, aqi FLOAT,
    PRIMARY KEY (sitename, publishtime)
)
```

---

## 3. Schema Validation Layer 結構

驗證層以 Dagster **Asset Checks**（`src/airpulse/defs/checks.py`）實作，在每次
materialize 時自動執行，並在 Dagster catalog 以通過／失敗標記呈現。

### 3.1 設計原則

- **嚴重度分級**:
  - `ERROR`:代表「不可信任的壞資料／模型」，搭配 `blocking=True`，**失敗即讓
    該 run 失敗**，阻止錯誤往下游擴散，並觸發 Slack 告警 sensor。
  - `WARN`:代表「異常但不致命」，僅標記、不中斷 pipeline。
- **驗證與清洗分離**:清洗負責結構化，驗證負責「值是否合理／是否新鮮／模型是否
  漂移」。

### 3.2 驗證項目一覽

| Check | 目標 Asset | 嚴重度 | Blocking | 規則 |
| --- | --- | --- | --- | --- |
| `raw_not_empty` | `raw_air_quality` | ERROR | ✅ | 擷取結果列數 > 0 |
| `no_missing_keys` | `cleaned_air_quality` | ERROR | ✅ | `sitename`、`publishtime` 皆非空 |
| `pm25_non_negative` | `cleaned_air_quality` | ERROR | ✅ | `pm2.5` ≥ 0（負濃度不可能） |
| `aqi_in_range` | `cleaned_air_quality` | WARN | ✗ | `aqi` 落在 0–500（EPA AQI 範圍） |
| `data_is_fresh` | `cleaned_air_quality` | WARN | ✗ | 最新讀數距今 ≤ 3 小時 |
| `model_not_drifting` | `model_metrics` | ERROR | ✅ | 最新 MAE 未較前次訓練漂移逾門檻 |

關鍵常數:`FRESHNESS_HOURS = 3`、`AQI_MIN, AQI_MAX = 0, 500`、drift 門檻 15%
（`DRIFT_THRESHOLD`，定義於 `metrics.py`）。

### 3.3 欄位層級 Schema（Column Schema）

`cleaned_air_quality` 透過 `dagster/column_schema` metadata 發佈欄位定義，讓
Dagster catalog 顯示完整欄位文件:

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `sitename` | string | 測站名稱 |
| `county` | string | 行政區（縣市） |
| `publishtime` | datetime | 觀測時間戳 |
| `pm2.5` / `pm10` / `o3` / `co` / `so2` / `no2` / `aqi` | float | 污染物濃度／指標 |

### 3.4 容錯防呆

- 欄位可能缺席:數值類 check 以 `df.get("欄位")` 取值，欄位不存在時不報錯。
- 冷啟動:`model_not_drifting` 在尚無已訓練模型（`status != "ok"`）時，回傳
  `passed=True`（WARN 級資訊），不誤判為漂移。

### 3.5 驗證 → 告警閉環

```
資料／模型異常
   → ERROR 級 blocking asset check 失敗
   → 整個 run 標記失敗
   → run_failure_sensor（slack_on_run_failure）讀取 SLACK_WEBHOOK_URL
   → POST 到 Slack Incoming Webhook
```

`model_not_drifting` 將原本僅默默寫入 DB 的 `drift_flag` 提升為可告警的治理閘門，
完成「偵測 → 記錄 → 告警」的監控閉環。

---

## 相關文件

- 資料治理（分類、品質閘門、血緣、擁有權）:[`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md)
- 程式碼位置:`src/airpulse/defs/`（`ingestion` / `cleaning` / `history` /
  `modeling` / `metrics` / `checks` / `sensors` / `governance` / `postgres`）
