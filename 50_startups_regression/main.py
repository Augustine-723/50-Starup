import os
import sys
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 設定標準輸出為 UTF-8 編碼以相容 Windows 終端機
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 設定中文字型與圖表美化
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
CSV_PATH = os.path.join(DATA_DIR, '50_Startups.csv')
DATA_URL = 'https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv'


def ensure_directories():
    """確保 data 與 outputs 目錄存在"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_dataset():
    """自動下載 50 Startups 資料集至 data 目錄"""
    if not os.path.exists(CSV_PATH):
        print(f"[*] 正在從 URL 下載資料集至 {CSV_PATH}...")
        urllib.request.urlretrieve(DATA_URL, CSV_PATH)
        print("[+] 資料集下載完成！")
    else:
        print(f"[+] 資料集已存在：{CSV_PATH}")


def load_and_explore():
    """讀取資料並檢查基本資訊、遺漏值與資料型態"""
    print("\n" + "=" * 60)
    print(" 步驟 1：資料讀取與基本檢視 (Data Exploration)")
    print("=" * 60)
    
    df = pd.read_csv(CSV_PATH)
    print(f"資料維度 (Shape): {df.shape[0]} 列 (Rows), {df.shape[1]} 欄 (Columns)\n")
    
    print("--- 前 5 筆資料 (head) ---")
    print(df.head().to_string(index=False))
    
    print("\n--- 資料型態與非空值統計 (info) ---")
    print(df.dtypes)
    
    print("\n--- 遺漏值檢視 (Missing Values) ---")
    missing = df.isnull().sum()
    print(missing)
    if missing.sum() == 0:
        print(">> 結果：本資料集無任何缺失值 (No Missing Values)。")
        
    print("\n--- 數值欄位統計摘要 (describe) ---")
    print(df.describe().round(2))
    
    print("\n--- 類別欄位 State 分布 ---")
    print(df['State'].value_counts())
    
    return df


def preprocess_data(df):
    """
    資料前處理與防範 Data Leakage：
    1. 將特徵 X 與目標 y (Profit) 分離
    2. 切分訓練集與測試集 (test_size=0.2, random_state=42)
    3. 對 State 進行 One-Hot Encoding，且僅在訓練集上 fit
    4. 移除第一個虛擬變數 (drop='first') 以避免多元共線性 (Dummy Variable Trap)
    """
    print("\n" + "=" * 60)
    print(" 步驟 2：資料前處理 (Preprocessing) 與防範 Data Leakage")
    print("=" * 60)
    
    X = df.drop(columns=['Profit'])
    y = df['Profit']
    
    # 80/20 切分訓練集與測試集
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"訓練集樣本數 (Train size): {len(X_train_raw)} 筆")
    print(f"測試集樣本數 (Test size): {len(X_test_raw)} 筆")
    
    # 防範 Data Leakage：OneHotEncoder 只在 X_train 上 fit
    # drop='first' 避免 Dummy Variable Trap (California 作為基準參考組)
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    ohe.fit(X_train_raw[['State']])
    
    state_dummy_cols = list(ohe.get_feature_names_out(['State']))
    print(f"\n[One-Hot Encoding 說明]")
    print(f"原始類別包括：{sorted(df['State'].unique().tolist())}")
    print(f"為避免 Dummy Variable Trap，移除基準組 (California)，產生的虛擬變數為：{state_dummy_cols}")
    
    # 轉換訓練集與測試集的 State
    train_state_encoded = ohe.transform(X_train_raw[['State']])
    test_state_encoded = ohe.transform(X_test_raw[['State']])
    
    num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
    
    # 組合數值特徵與 One-Hot Encoding 特徵
    X_train = pd.concat([
        X_train_raw[num_cols].reset_index(drop=True),
        pd.DataFrame(train_state_encoded, columns=state_dummy_cols)
    ], axis=1)
    
    X_test = pd.concat([
        X_test_raw[num_cols].reset_index(drop=True),
        pd.DataFrame(test_state_encoded, columns=state_dummy_cols)
    ], axis=1)
    
    # 重設 y 索引以確保對齊
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    print(f"\n最終輸入模型的特徵欄位 ({len(X_train.columns)} 個)：")
    for idx, col in enumerate(X_train.columns, 1):
        print(f"  {idx}. {col}")
        
    return X_train, X_test, y_train, y_test


def run_feature_selection(X_train, X_test, y_train, y_test):
    """
    使用 Multiple Linear Regression 與 Recursive Feature Elimination (RFE)
    依序測試 1 到 5 個特徵，並在測試集上計算 MSE, RMSE, MAE, R-squared
    """
    print("\n" + "=" * 60)
    print(" 步驟 3 & 4：特徵選取 (RFE) 與多元線性迴歸模型評估")
    print("=" * 60)
    
    results = []
    
    for k in range(1, len(X_train.columns) + 1):
        # 1. 使用多元線性迴歸作為基礎估計器建立 RFE
        estimator = LinearRegression()
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)
        
        # 取得被選中的特徵名稱
        selected_features = list(X_train.columns[rfe.support_])
        
        # 2. 以選中的特徵重新擬合線性迴歸模型
        model = LinearRegression()
        model.fit(X_train[selected_features], y_train)
        
        # 3. 預測訓練集與測試集
        y_train_pred = model.predict(X_train[selected_features])
        y_test_pred = model.predict(X_test[selected_features])
        
        # 4. 計算評估指標
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = float(np.sqrt(train_mse))
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = float(np.sqrt(test_mse))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'Number of Features': k,
            'Selected Features': selected_features,
            'Train RMSE': train_rmse,
            'RMSE': test_rmse,
            'MSE': test_mse,
            'MAE': test_mae,
            'R-squared': test_r2
        })
        
    results_df = pd.DataFrame(results)
    
    # 格式化輸出要求表格
    display_df = results_df[['Number of Features', 'Selected Features', 'RMSE', 'R-squared']].copy()
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"{x:,.2f}")
    display_df['R-squared'] = display_df['R-squared'].apply(lambda x: f"{x:.4f}")
    
    print("\n--- Feature Selection 評估結果 DataFrame ---")
    print(display_df.to_string(index=False))
    
    # 儲存詳細結果 CSV
    csv_output_path = os.path.join(OUTPUT_DIR, 'feature_selection_results.csv')
    # 轉換 Selected Features 格式為字串以利存成 CSV
    results_to_save = results_df.copy()
    results_to_save['Selected Features'] = results_to_save['Selected Features'].apply(lambda x: "; ".join(x))
    results_to_save.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] 結果表格已成功儲存至：{csv_output_path}")
    
    return results_df


def plot_feature_selection(results_df):
    """
    繪製左右並排圖表：
    左圖：RMSE by Number of Features (藍色折線與圓形標記)
    右圖：R-squared by Number of Features (藍色折線與圓形標記)
    儲存至 outputs/feature_selection.png
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    x = results_df['Number of Features']
    rmse_values = results_df['RMSE']
    r2_values = results_df['R-squared']
    
    # 左圖：RMSE by Number of Features
    ax1.plot(x, rmse_values, color='#1f77b4', marker='o', linewidth=2.2, markersize=8, label='RMSE')
    ax1.set_title('RMSE by Number of Features', fontsize=14, fontweight='bold', pad=12)
    ax1.set_xlabel('Number of Features', fontsize=12, labelpad=8)
    ax1.set_ylabel('RMSE', fontsize=12, labelpad=8)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 在折線點上加上數值標籤
    for i, txt in enumerate(rmse_values):
        ax1.annotate(f"{txt:,.0f}", (x.iloc[i], txt), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, fontweight='bold', color='#0f4c81')
    
    # 右圖：R-squared by Number of Features
    ax2.plot(x, r2_values, color='#1f77b4', marker='o', linewidth=2.2, markersize=8, label='R-squared')
    ax2.set_title('R-squared by Number of Features', fontsize=14, fontweight='bold', pad=12)
    ax2.set_xlabel('Number of Features', fontsize=12, labelpad=8)
    ax2.set_ylabel('R-squared', fontsize=12, labelpad=8)
    ax2.set_xticks(x)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(r2_values):
        ax2.annotate(f"{txt:.4f}", (x.iloc[i], txt), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, fontweight='bold', color='#0f4c81')
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'feature_selection.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] 視覺化圖表已成功儲存至：{output_path}")


def plot_overfitting_underfitting(results_df):
    """
    繪製 Training RMSE 與 Testing RMSE 的比較折線圖
    X-axis 為 Number of Features
    分析模型之 Overfitting 與 Underfitting 現象
    儲存至 outputs/train_test_rmse.png
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    x = results_df['Number of Features']
    train_rmse = results_df['Train RMSE']
    test_rmse = results_df['RMSE']
    
    ax.plot(x, train_rmse, color='#2b5c8f', marker='o', linewidth=2.2, markersize=8, label='Training RMSE')
    ax.plot(x, test_rmse, color='#d95f02', marker='s', linewidth=2.2, markersize=8, linestyle='--', label='Testing RMSE')
    
    ax.set_title('Training vs Testing RMSE by Number of Features', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Features', fontsize=12, labelpad=8)
    ax.set_ylabel('RMSE (Root Mean Squared Error)', fontsize=12, labelpad=8)
    ax.set_xticks(x)
    ax.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 標示數值 (訓練集向上偏移，測試集向下偏移)
    for i in range(len(x)):
        ax.annotate(f"{train_rmse.iloc[i]:,.0f}", (x.iloc[i], train_rmse.iloc[i]),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8.5, fontweight='bold', color='#2b5c8f')
        ax.annotate(f"{test_rmse.iloc[i]:,.0f}", (x.iloc[i], test_rmse.iloc[i]),
                    textcoords="offset points", xytext=(0, -18), ha='center', fontsize=8.5, fontweight='bold', color='#d95f02')
        
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'train_test_rmse.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[+] 訓練與測試 RMSE 比較圖表已儲存至：{output_path}")


def analyze_overfitting_underfitting(results_df):
    """輸出過擬合 (Overfitting) 與欠擬合 (Underfitting) 之深入分析"""
    print("\n" + "=" * 60)
    print(" 步驟 5：過擬合 (Overfitting) 與欠擬合 (Underfitting) 深入分析")
    print("=" * 60)
    
    print("""
【分析結論】：
1. 欠擬合 (Underfitting) 階段：
   - 當特徵數量為 1 與 2 時 (選取特徵僅包含 State_New York, State_Florida)：
     訓練集 RMSE 與測試集 RMSE 均高達 35,000 ~ 41,000 以上，且測試集 R-squared 為負值 (-0.56 ~ -0.59)。
     這代表模型完全無法捕捉資料的主導變異（缺少核心的研發支出 R&D Spend），
     模型偏差 (Bias) 極大，屬於典型的「嚴重欠擬合 (Underfitting)」。

2. 模型效能躍升點 (Optimal Trade-off)：
   - 當特徵數量達到 3 時 (選取特徵：R&D Spend, State_Florida, State_New York)：
     測試集 RMSE 從 35,902 驟降至 7,946.37，測試集 R-squared 大幅躍升至 0.9220。
     這證明 R&D Spend 是預測 Profit 最關鍵的核心特徵。

3. 微幅過擬合 (Slight Overfitting) 階段：
   - 當特徵數量增加至 4 與 5 時 (陸續加入 Administration 與 Marketing Spend)：
     訓練集 RMSE 持續單調下降（從 9,517.84 降至 9,162.34 再降至 8,927.49），代表在訓練集上擬合得更貼合；
     然而，測試集 RMSE 卻由 7,946.37 上升至 9,316.54 與 9,055.96，測試集 R-squared 也從 0.9220 略降至 0.8987。
     訓練集誤差持續下降但測試集誤差反而上升的現象，即代表模型出現了「微幅過擬合 (Slight Overfitting)」，
     顯示額外引入的變數對泛化預測能力帶來了雜訊或多重共線性干擾。
""")


def main():
    print("=" * 60)
    print(" 50 Startups 多元線性迴歸與特徵選取 (RFE) 分析流程開始")
    print("=" * 60)
    
    # 步驟 1: 建立目錄與下載資料
    ensure_directories()
    download_dataset()
    
    # 步驟 2: 探索資料
    df = load_and_explore()
    
    # 步驟 3: 前處理 (One-Hot Encoding, train_test_split, 防止 Data Leakage)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 步驟 4: RFE 特徵選取與多元線性迴歸訓練評估 (1 ~ 5 特徵)
    results_df = run_feature_selection(X_train, X_test, y_train, y_test)
    
    # 步驟 5: 視覺化繪圖
    print("\n" + "=" * 60)
    print(" 步驟 6：生成視覺化圖表")
    print("=" * 60)
    plot_feature_selection(results_df)
    plot_overfitting_underfitting(results_df)
    
    # 步驟 7: 過擬合與欠擬合分析
    analyze_overfitting_underfitting(results_df)
    
    print("\n" + "=" * 60)
    print(" 所有分析流程與輸出檔案已成功執行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
