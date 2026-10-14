# 50 Startups 多元線性迴歸與特徵選取 (Multiple Linear Regression & RFE)

本專案使用著名的 **50 Startups Dataset**，透過 Python 建立完整的機器學習迴歸分析流程。實作重點涵蓋：資料探索與前處理（防範 Data Leakage）、One-Hot Encoding 與虛擬變數陷阱處理、遞迴特徵消除（Recursive Feature Elimination, RFE）、多元評估指標計算、視覺化圖表生成，以及模型的過擬合（Overfitting）與欠擬合（Underfitting）深入分析。

---

## 專案目錄結構 (Project Structure)

```text
50_startups_regression/
├── data/
│   └── 50_Startups.csv              # 自動下載的原始資料集
├── outputs/
│   ├── feature_selection.png        # 左右並排圖：RMSE 與 R-squared 隨特徵數變化趨勢
│   ├── train_test_rmse.png          # 訓練集 vs 測試集 RMSE 比較折線圖 (過擬合/欠擬合分析)
│   └── feature_selection_results.csv# 特徵選取與模型評估數值結果表格
├── web/                             # 互動式網頁儀表板 (Web Dashboard)
│   ├── index.html                   # 網頁結構 (語意化 HTML5、Tabs、預測器介面)
│   ├── style.css                    # 現代深色玻璃擬物化風格 (Vanilla CSS)
│   ├── app.js                       # 互動式邏輯、Chart.js 動態繪圖、即時計算
│   └── data.js                      # 預載模型權重與資料集 JSON
├── main.py                          # 主執行程式 (資料下載、前處理、特徵選取、評估、繪圖)
├── serve.py                         # 輕量級本地 Web 伺服器 (啟動於 http://localhost:8080)
├── requirements.txt                 # 專案依賴套件清單
└── README.md                        # 專案說明文件
```

---

## 資料集說明 (Dataset Overview)

* **資料來源**：`https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv`
* **樣本數**：50 筆創業公司營運資料
* **特徵欄位**：
  * `R&D Spend` (連續變數)：研發支出（美元）
  * `Administration` (連續變數)：行政管理支出（美元）
  * `Marketing Spend` (連續變數)：行銷推廣支出（美元）
  * `State` (類別變數)：公司設立州別（New York, California, Florida）
* **目標變數 (Target)**：
  * `Profit` (連續變數)：公司獲利（美元）

---

## 安裝與執行步驟 (Installation & Execution)

### 1. 環境需求
* Python 3.9+ (本專案於 Python 3.11 環境驗證通過)

### 2. 安裝依賴套件
```bash
pip install -r requirements.txt
```
或使用 `uv` 快速安裝：
```bash
uv pip install -r requirements.txt
```

### 3. 執行命令列分析 (CLI)
切換至專案資料夾後直接執行 `main.py`：
```bash
python main.py
```
程式將自動完成：
1. 自動下載 `50_Startups.csv` 至 `data/` 目錄。
2. 進行資料前處理、One-Hot Encoding、80/20 資料切分。
3. 執行 RFE 特徵選取（依序測試 1 至 5 個特徵）。
4. 輸出模型指標 DataFrame 並儲存至 `outputs/feature_selection_results.csv`。
5. 生成高解析視覺化折線圖儲存至 `outputs/` 目錄。
6. 印出過擬合與欠擬合之分析報告。

### 4. 啟動互動式網頁儀表板 (Web Dashboard)
執行內建的本地伺服器腳本：
```bash
python serve.py
```
或使用 Python 內建 HTTP 模組：
```bash
python -m http.server 8080 -d web
```
啟動後打開瀏覽器訪問：**`http://localhost:8080`**，即可享受完整、流暢的互動式機器學習儀表板！
也可以直接雙擊以瀏覽器打開：`web/index.html`。

---

## 資料前處理與特徵工程 (Preprocessing & Feature Engineering)

### 1. 防範資料外洩 (Data Leakage Prevention)
* 資料嚴格採用 80/20 切分（`test_size=0.2, random_state=42`）。
* **重要原則**：所有需要「學習/擬合資料統計特徵」的前處理步驟（包括類別編碼器 `OneHotEncoder`），一律**僅在 Training Set 上進行 `fit`**，隨後再 `transform` 訓練集與測試集，杜絕測試集資訊洩漏給訓練過程。

### 2. One-Hot Encoding 與虛擬變數陷阱 (Dummy Variable Trap)
* 類別欄位 `State` 包含 3 個類別：`California`、`Florida`、`New York`。
* 在多元線性迴歸模型中，若將 3 個州全數轉換為獨立虛擬變數，由於三個變數之和恆等於 1（即 $D_1 + D_2 + D_3 = 1$），會與迴歸模型的常數截距項（Intercept）形成完全共線性（Multicollinearity），稱為**虛擬變數陷阱（Dummy Variable Trap）**。
* 為了解決此問題，使用 `OneHotEncoder(drop='first')`，以 `California` 作為基準參考組（Baseline），產生 2 個虛擬變數：`State_Florida` 與 `State_New York`。
* 最終模型的候選特徵共有 **5 個**：
  1. `R&D Spend`
  2. `Administration`
  3. `Marketing Spend`
  4. `State_Florida`
  5. `State_New York`

---

## 迴歸評估指標解釋 (Evaluation Metrics)

| 指標名稱 | 數學定義 | 說明與特性 |
| :--- | :--- | :--- |
| **MSE (Mean Squared Error)** | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | **均方誤差**：預測誤差平方後的平均值。對離群值極為敏感，單位為目標變數單位的平方。 |
| **RMSE (Root Mean Squared Error)** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | **均方根誤差**：MSE 的平方根，使誤差單位回歸至與目標變數相同（美元），數值越小代表預測越精確。 |
| **MAE (Mean Absolute Error)** | $\frac{1}{n}\sum_{i=1}^{n}\|y_i - \hat{y}_i\|$ | **平均絕對誤差**：真實值與預測值差值絕對值的平均值，直接反映平均預測偏差幅度，對離群值的敏感度較 RMSE 低。 |
| **R-squared ($R^2$)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **判定係數 / 決定係數**：衡量模型解釋目標變數變異程度的比例。取值通常在 0 到 1 之間（越接近 1 模型配適度越佳）；若模型表現比單純預測平均值還差，在測試集上可能為負值。 |

---

## 實驗結果 (Experimental Results)

透過 RFE（Recursive Feature Elimination）依序挑選 1 到 5 個特徵，模型在測試集（Test Set, $N=10$）之實際執行數值如下：

| 特徵數量 (k) | 選取特徵 (Selected Features) | 訓練集 RMSE | 測試集 RMSE | 測試集 MSE | 測試集 MAE | 測試集 $R^2$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `State_New York` | 41,260.98 | 35,552.16 | 1,263,956,018.24 | 24,085.80 | **-0.5608** |
| **2** | `State_Florida`, `State_New York` | 40,631.23 | 35,902.08 | 1,288,959,486.69 | 24,725.66 | **-0.5917** |
| **3** | `R&D Spend`, `State_Florida`, `State_New York` | 9,517.84 | **7,946.37** | 63,144,831.84 | 6,296.78 | **0.9220** |
| **4** | `R&D Spend`, `Administration`, `State_Florida`, `State_New York` | 9,162.34 | 9,316.54 | 86,797,850.66 | 7,436.62 | **0.8928** |
| **5** | `R&D Spend`, `Administration`, `Marketing Spend`, `State_Florida`, `State_New York` | 8,927.49 | 9,055.96 | 82,010,363.04 | 6,961.48 | **0.8987** |

---

## 過擬合與欠擬合分析 (Overfitting vs. Underfitting Analysis)

### 1. 嚴重欠擬合 (Severe Underfitting)：$k = 1, 2$
* **特徵組合**：僅有州別虛擬變數（`State_New York`、`State_Florida`）。
* **表現特徵**：
  * 訓練集 RMSE 與測試集 RMSE 均高達 35,000 ~ 41,000 美元以上。
  * 測試集 $R^2$ 為負值（$-0.56$ 與 $-0.59$），表示模型的預測表現甚至不如直接使用平均值。
* **原因分析**：公司獲利的主要動力來自核心研發與營運支出，僅憑地理位置完全無法解釋獲利能力的變異。模型缺乏關鍵變數，高偏差（High Bias），屬於標準的**嚴重欠擬合**。

### 2. 最佳泛化效能點 (Optimal Trade-off)：$k = 3$
* **特徵組合**：`R&D Spend` + `State_Florida` + `State_New York`。
* **表現特徵**：
  * 測試集 RMSE 驟降至 **7,946.37 美元**（全實驗最低）。
  * 測試集 $R^2$ 攀升至 **0.9220**（全實驗最高）。
* **原因分析**：加入 `R&D Spend` 後，模型成功捕捉了決定利潤最核心的驅動因子，以精簡的特徵數達到了最強的泛化預測能力。

### 3. 微幅過擬合 (Slight Overfitting)：$k = 4, 5$
* **特徵組合**：加入 `Administration` 與 `Marketing Spend`。
* **表現特徵**：
  * **訓練集 RMSE 持續下降**：$9,517.84 \rightarrow 9,162.34 \rightarrow 8,927.49$。
  * **測試集 RMSE 反而上升**：由 $7,946.37$ 上升至 $9,316.54$ 與 $9,055.96$。
  * **測試集 $R^2$ 微幅下滑**：由 $0.9220$ 下降至 $0.8928$ 與 $0.8987$。
* **原因分析**：在線性迴歸中，隨著特徵數量增加，模型在訓練資料上的自由度更高，殘差平方和（RSS）必然不增，使得訓練集誤差單調遞減；但在測試集上，`Administration` 等變數與 `Profit` 相關度較低，反而引入了多重共線性與雜訊，導致泛化能力下降，此為典型的**微幅過擬合**。
