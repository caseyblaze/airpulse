# airpulse ML 建模計畫(v1 藍圖)

> 狀態:**規劃中,尚未實作**。本文件為實作藍圖,供日後逐步落地。
> 預測目標:各測站「**下一小時**」的 pm2.5。

## 0. 設計目標與約束

- **目標**:對每個測站做一小時後的 pm2.5 預測,並能評估其**未來**表現。
- **約束**:資料源為即時快照(每站每小時一筆),須靠 `air_quality_history`
  跨 run 累積成時間序列。
- **v1 原則**:先求「**評估正確** + **特徵齊全**」;調參、特徵修剪、滾動訓練窗口
  皆留待 v2。

---

## 1. 資料切分:真正的時間切分

改掉現行 `train_test_split(shuffle=False)`——它在以 `["sitename","publishtime"]`
排序的資料上,實際切出的是「字母排序最後幾站」的**空間切分**,而非時間切分。

**新作法(全域時間切點):**

1. 取所有不同 `publishtime` 排序,選最後 20% 的時間點為切點 `T`。
2. `train = publishtime < T` 的列;`test = publishtime >= T` 的列。
3. **所有測站同時存在於 train 與 test**,差別在 test 是較晚的時間 → 評估的是
   真正的「未來」,且每站都能預測。

**冷啟動門檻**:時間切分需足夠時間點,設 `MIN_TIMESTAMPS = 12`(約 10 train /
2 test);不足時 `model_predictions` 回傳 `status="insufficient_data"`,不訓練。

---

## 2. 特徵工程

### 2.1 鐵律(防資料洩漏)

> 外生變數(其他污染物、氣象)**一律只用 lag 版本**;只有**時間、地理**這類
> 「預測時點本來就已知」的特徵可用同期值。歷史表仍儲存所有當期原始值(作為算
> lag 的原料),但**同期外生欄位不進特徵矩陣 X**。

### 2.2 建議特徵全集(v1,日後依 feature importance 修剪)

| 群組 | 特徵 | 同期/lag | 備註 |
| --- | --- | --- | --- |
| 自迴歸 | `pm25_lag1/2/3` | lag | 主訊號 |
| 滾動統計 | `pm25_roll3_mean`、`pm25_roll3_std` | lag(由 lag1..3 算) | 近期趨勢/波動 |
| 同期污染物 | `pm10_lag1`、`o3_lag1`、`co_lag1`、`so2_lag1`、`no2_lag1`、`nox_lag1`、`no_lag1`、`aqi_lag1` | lag | |
| 均值/8hr | `pm25_avg_lag1`、`pm10_avg_lag1`、`o3_8hr_lag1`、`co_8hr_lag1`、`so2_avg_lag1` | lag | 可選,易共線 |
| 氣象 | `wind_speed_lag1`、`wind_dir_sin_lag1`、`wind_dir_cos_lag1` | lag | 風向用 sin/cos 環形編碼 |
| 時間 | `hour_sin/cos`、`dow_sin/cos`、`month_sin/cos` | **同期** | 日週期/季節,預測時已知 |
| 地理 | `latitude`、`longitude` | **同期** | 讓模型學空間分布 |
| 站別 | `county`(**one-hot**,約 22 維) | **同期** | siteid(84 站)若日後加入則改用 target encoding |

**丟棄/小心**:`pollutant`、`status` 為描述「當期空氣狀態」的類別欄,等同洩漏;
要用須 lag,v1 先丟。

---

## 3. 模型

- **RandomForestRegressor**(baseline),`n_estimators=100`、`random_state=42`。
- **多站共用單一模型**(跨站學習)。
- **v1 全歷史訓練**(資料尚少,先看效果)。
  - v2 待辦:滾動訓練窗口(只用最近 K 天)或近期樣本加權,讓預測對快速變化的
    空氣更敏感。**注意**:這與下方 drift 窗口是兩件事——drift 窗口只影響告警,
    訓練資料新近性才影響預測能否反映當前狀態。

---

## 4. 評估指標

`model_predictions` 輸出:

| 指標 | 定義 |
| --- | --- |
| `mae` | `mean_absolute_error(y_test, y_pred)` |
| `r2` | `r2_score(y_test, y_pred)` |
| `relative_mae` | `mae / mean(test 期實際 pm2.5)`——跨不同難易時段可比 |
| `n_train` / `n_test` / `n_timestamps` | 樣本與時間點數,監控資料成長 |

**逐站表現**:在 test 集上 `groupby("sitename")` 各算 `mae` / `relative_mae` /
`n_test`,寫入新表 `model_metrics_by_site`。

---

## 5. Drift 偵測(改版)

```
relative_mae = mae / mean(test 期實際 pm2.5)
baseline     = median(最近 3 次的 relative_mae)   # 不足 3 次用現有的
drift_flag   = (relative_mae - baseline) / baseline > 0.15   # DRIFT_THRESHOLD
```

**設計理由**:
- **相對誤差**:時間切分後每次 test 是不同未來時段、難易不同,用相對誤差才能
  公平比較,避免把「時段本來就難」誤判為模型退化。
- **近 3 次中位數**:平滑單次雜訊(一次暴衝不會立刻變基準),同時 3 小時內就能
  反應,適合空品快速變化的特性。
- **不做固定 baseline**:固定 baseline 雖最能抓「緩慢退化」,但需精確定義「穩定」
  與重置政策,否則白做;留待 v2 連同穩定性規格一起評估。

**告警閉環**:`drift_flag` → `model_not_drifting`(ERROR + blocking)asset check
失敗 → run 失敗 → `slack_on_run_failure` sensor → Slack。

---

## 6. Schema 變更

### 6.1 `air_quality_history` 加寬

新增欄位(供日後建特徵;欄名正規化:`pm2.5→pm25`、`pm2.5_avg→pm25_avg`):

```sql
-- 既有: sitename, county, publishtime, pm25, pm10, o3, co, so2, no2, aqi
-- 新增:
siteid       TEXT,
latitude     FLOAT,
longitude    FLOAT,
pm25_avg     FLOAT,
pm10_avg     FLOAT,
o3_8hr       FLOAT,
co_8hr       FLOAT,
so2_avg      FLOAT,
no           FLOAT,
nox          FLOAT,
wind_speed   FLOAT,
wind_direc   FLOAT
-- PK 維持 (sitename, publishtime)
```

對應地,`ingestion` 與 `cleaning` 需保留這些欄位(目前被丟棄)。

### 6.2 `model_metrics` 新增欄位

```sql
ALTER TABLE model_metrics ADD COLUMN relative_mae FLOAT;
```

### 6.3 `model_metrics_by_site`(新表)

```sql
CREATE TABLE IF NOT EXISTS model_metrics_by_site (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL,
    sitename     TEXT NOT NULL,
    n_test       INT,
    mae          FLOAT,
    relative_mae FLOAT
)
```

---

## 7. 分階段實作順序

1. 加寬 `ingestion` / `cleaning` / `air_quality_history` 的欄位。
2. 特徵工程函式(lag/roll/時間/環形編碼/one-hot)+ 時間切分。
3. 評估指標(加 `relative_mae`)+ 逐站指標寫入 `model_metrics_by_site`。
4. drift 改為相對誤差 + 近 3 次中位數。
5. 單元測試(切分無洩漏、冷啟動、drift 邏輯)→ 部署。

---

## 8. v2+ 待辦

- 滾動訓練窗口 / 近期加權。
- 超參數調校、依 feature importance 修剪特徵。
- 固定 baseline + 穩定性規格(抓季節性緩慢退化)。
- 多步預測(t+2、t+3)。

---

## 相關文件

- Pipeline 架構:[`PIPELINE.md`](./PIPELINE.md)
- 資料治理:[`DATA_GOVERNANCE.md`](./DATA_GOVERNANCE.md)
