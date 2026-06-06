# 資料治理

> **語言:** [English](DATA_GOVERNANCE.md) | 繁體中文

**airpulse** 管線如何治理其擷取、轉換與對外提供的資料。目前的資料集（台灣環保署開放空氣品質資料）屬於**公開且不含個人識別資訊**；以下框架的設計目標是：只需變更分類設定而無需改動架構，即可讓同一條管線順暢延伸至受管制的資料（健康、金融）。

## 1. 資料分類

每個資產都會在 Dagster 目錄中標記一個 `data_classification` 標籤。等級由低到高的敏感度依序為：

| 等級 | 說明 | 處理方式 |
| --- | --- | --- |
| `public` | 開放資料，無任何限制 | 無需額外措施 — **所有現行 airpulse 資產** |
| `internal` | 僅限內部使用，非敏感性資料 | 存取控制的儲存空間 |
| `confidential` | 商業敏感資料 | 靜態加密、限制角色存取 |
| `restricted` | 個人識別資訊／受管制資料（PDPA、類 HIPAA、金融） | 遮罩處理＋稽核記錄＋最小權限原則 |

分類定義位於 `src/airpulse/defs/governance.py`。`restricted` 欄位在儲存前必須通過唯一的稽核遮罩路徑（`mask_pii`）——採用確定性假名化，以確保在原始資料不可還原的前提下仍可進行資料聯結。

## 2. 資料品質（資產檢查）

品質閘門於每次實體化時執行，並以通過／失敗的標記顯示於目錄中（`src/airpulse/defs/checks.py`）：

| 檢查項目 | 資產 | 嚴重程度 | 規則 |
| --- | --- | --- | --- |
| `raw_not_empty` | raw_air_quality | ERROR | 擷取作業需回傳資料列 |
| `no_missing_keys` | cleaned_air_quality | ERROR | sitename 與 publishtime 不得缺漏 |
| `pm25_non_negative` | cleaned_air_quality | ERROR | pm2.5 ≥ 0 |
| `aqi_in_range` | cleaned_air_quality | WARN | AQI 值需介於 0–500 之間 |
| `data_is_fresh` | cleaned_air_quality | WARN | 最新資料距今不超過 3 小時 |

ERROR 等級的失敗代表資料有誤，應阻止下游信任該資料；WARN 則在不中斷管線的情況下標示異常。

## 3. 元資料、血緣與所有權

- **血緣關係**：資產形成明確的有向無環圖（DAG）——  
  `raw_air_quality → cleaned_air_quality → air_quality_history → model_predictions → model_metrics`。
- **所有權**：每個資產均設有 `owners=["team:data-engineering"]`。
- **結構描述**：`cleaned_air_quality` 發布欄位層級的 `TableSchema`。
- **運維元資料**：每次執行皆會輸出資料列數、不重複站點數、來源網址及模型指標（MAE／R²／漂移量），以供目錄可觀測性使用。
- **種類標籤**：資產依使用技術標示徽章（api、pandas、sklearn、postgres）。

## 4. 保留政策與存取控制（路線圖）

本階段不在範疇之內；預計於 GCP 遷移（第二階段）時規劃：歷史資料表保留期限與分區策略、資料庫角色分離，以及密鑰輪換（目前密鑰僅存放於 Dagster+ 生產環境範圍，從未納入程式碼儲存庫）。
