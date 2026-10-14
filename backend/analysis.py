import os
import gzip
import pickle
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', '50_Startups.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_dataset(csv_path=DATA_PATH):
    """讀取 50 Startups 資料集並計算探索性統計"""
    df = pd.read_csv(csv_path)
    
    # 數值摘要
    num_cols = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    stats_dict = df[num_cols].describe().round(2).to_dict()
    
    # 相關係數矩陣 (Correlation Matrix)
    corr_matrix = df[num_cols].corr().round(4).to_dict()
    
    # 州別分布
    state_counts = df['State'].value_counts().to_dict()
    
    return {
        'dataframe': df,
        'records': df.to_dict(orient='records'),
        'shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'columns': list(df.columns),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': stats_dict,
        'correlation_matrix': corr_matrix,
        'state_distribution': state_counts
    }


def prepare_data(df, random_state=42):
    """
    CRISP-DM Step 2: 前處理
    - train_test_split (80/20)
    - OneHotEncoder 僅 fit 於訓練集，防範 Data Leakage
    - drop='first' (California 作為基準組) 避免 Dummy Variable Trap
    """
    X = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State']]
    y = df['Profit']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    ohe.fit(X_train_raw[['State']])
    dummy_cols = list(ohe.get_feature_names_out(['State']))
    
    num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
    
    train_encoded = pd.DataFrame(ohe.transform(X_train_raw[['State']]), columns=dummy_cols)
    test_encoded = pd.DataFrame(ohe.transform(X_test_raw[['State']]), columns=dummy_cols)
    
    X_train = pd.concat([X_train_raw[num_cols].reset_index(drop=True), train_encoded], axis=1)
    X_test = pd.concat([X_test_raw[num_cols].reset_index(drop=True), test_encoded], axis=1)
    
    # 儲存對應的原始索引以便追蹤真實個案
    train_indices = list(X_train_raw.index)
    test_indices = list(X_test_raw.index)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'feature_names': list(X_train.columns),
        'ohe': ohe,
        'random_state': random_state
    }


def compute_metrics(y_true, y_pred, p_features, n_samples):
    """計算回歸評估指標 (RMSE, MSE, MAE, R2, Adjusted R2)"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # Adjusted R-squared: 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
    if n_samples > p_features + 1:
        adj_r2 = 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - p_features - 1))
    else:
        adj_r2 = r2
        
    return {
        'mse': round(float(mse), 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4),
        'adj_r2': round(float(adj_r2), 4)
    }


def compute_vif_and_stats(X_train_sub, y_train, selected_features):
    """
    使用 statsmodels OLS 進行統計推論與計算 VIF
    """
    X_with_const = sm.add_constant(X_train_sub)
    ols_model = sm.OLS(y_train, X_with_const).fit()
    
    coefficients = []
    
    # Intercept
    ci = ols_model.conf_int().loc['const']
    coefficients.append({
        'feature': 'Intercept (β₀)',
        'coef': round(float(ols_model.params['const']), 2),
        'stderr': round(float(ols_model.bse['const']), 2),
        'tstat': round(float(ols_model.tvalues['const']), 3),
        'pvalue': float(ols_model.pvalues['const']),
        'ci_lower': round(float(ci[0]), 2),
        'ci_upper': round(float(ci[1]), 2),
        'vif': None
    })
    
    # VIF 計算 (若特徵數為 1 則 VIF 為 1.0)
    vif_dict = {}
    if len(selected_features) > 1:
        for idx, feat in enumerate(selected_features):
            try:
                # 在包含常數項的設計矩陣中計算 VIF
                feat_idx = X_with_const.columns.get_loc(feat)
                val = variance_inflation_factor(X_with_const.values, feat_idx)
                vif_dict[feat] = round(float(val), 3) if not np.isinf(val) else 999.0
            except Exception:
                vif_dict[feat] = 1.0
    else:
        vif_dict[selected_features[0]] = 1.0
        
    for feat in selected_features:
        ci = ols_model.conf_int().loc[feat]
        coefficients.append({
            'feature': feat,
            'coef': round(float(ols_model.params[feat]), 4),
            'stderr': round(float(ols_model.bse[feat]), 4),
            'tstat': round(float(ols_model.tvalues[feat]), 3),
            'pvalue': float(ols_model.pvalues[feat]),
            'ci_lower': round(float(ci[0]), 4),
            'ci_upper': round(float(ci[1]), 4),
            'vif': vif_dict.get(feat, 1.0)
        })
        
    f_stat = round(float(ols_model.fvalue), 2) if not np.isnan(ols_model.fvalue) else None
    f_pvalue = float(ols_model.f_pvalue) if not np.isnan(ols_model.f_pvalue) else None
    
    return coefficients, f_stat, f_pvalue, ols_model


def run_feature_selection_algorithms(prep_data):
    """
    CRISP-DM Step 3 & 4:
    實作 6 大特徵選擇與建模演算法：
    1. Baseline OLS (Correlation / Standard OLS order)
    2. Forward Sequential Feature Selection (Forward SFS)
    3. Recursive Feature Elimination (RFE)
    4. Lasso (L1) with Standardized features & cross-validation
    5. ElasticNet (L1 + L2) with Standardized features
    6. Exhaustive Feature Search (Best Subset Selection via CV RMSE)
    """
    X_train = prep_data['X_train']
    X_test = prep_data['X_test']
    y_train = prep_data['y_train']
    y_test = prep_data['y_test']
    test_indices = prep_data['test_indices']
    feature_names = prep_data['feature_names']
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    algorithms = [
        'Forward SFS',
        'RFE',
        'Exhaustive Search',
        'Baseline OLS',
        'Lasso',
        'ElasticNet'
    ]
    
    # 預先標準化用於 Lasso 與 ElasticNet
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    
    for algo in algorithms:
        results[algo] = []
        
        for k in range(1, len(feature_names) + 1):
            if algo == 'Forward SFS':
                if k == len(feature_names):
                    selected = list(feature_names)
                else:
                    sfs = SequentialFeatureSelector(
                        LinearRegression(), n_features_to_select=k, direction='forward', cv=cv, scoring='neg_root_mean_squared_error'
                    )
                    sfs.fit(X_train, y_train)
                    selected = list(X_train.columns[sfs.support_])
                
            elif algo == 'RFE':
                # 標準化後之 RFE 具備尺度不變性
                rfe = RFE(LinearRegression(), n_features_to_select=k, step=1)
                rfe.fit(X_train_scaled, y_train)
                selected = list(X_train.columns[rfe.support_])
                
            elif algo == 'Exhaustive Search':
                best_sub = None
                best_cv = float('inf')
                for combo in itertools.combinations(feature_names, k):
                    lr = LinearRegression()
                    scores = cross_val_score(lr, X_train[list(combo)], y_train, cv=cv, scoring='neg_root_mean_squared_error')
                    mean_rmse = -np.mean(scores)
                    if mean_rmse < best_cv:
                        best_cv = mean_rmse
                        best_sub = list(combo)
                selected = best_sub
                
            elif algo == 'Baseline OLS':
                # 按單變數相關係數由大至小排序
                corrs = [abs(np.corrcoef(X_train[f], y_train)[0, 1]) for f in feature_names]
                sorted_feats = [f for _, f in sorted(zip(corrs, feature_names), reverse=True)]
                selected = sorted_feats[:k]
                
            elif algo == 'Lasso':
                # Lasso 根據正則化係數路徑選取重要特徵
                # 測試一組 alpha 找出恰好有 k 個非零係數的最佳特徵集合
                best_features = None
                alphas = np.logspace(-2, 5, 200)
                for a in alphas:
                    las = Lasso(alpha=a, random_state=42, max_iter=10000)
                    las.fit(X_train_scaled, y_train)
                    non_zero = [f for f, c in zip(feature_names, las.coef_) if abs(c) > 1e-4]
                    if len(non_zero) == k:
                        best_features = non_zero
                        break
                if not best_features:
                    # 若無法恰好匹配，取 abs(coef) 最大的前 k 個
                    las = Lasso(alpha=100.0, random_state=42, max_iter=10000)
                    las.fit(X_train_scaled, y_train)
                    order = np.argsort(np.abs(las.coef_))[::-1]
                    best_features = [feature_names[i] for i in order[:k]]
                selected = best_features
                
            elif algo == 'ElasticNet':
                best_features = None
                alphas = np.logspace(-2, 5, 100)
                for a in alphas:
                    en = ElasticNet(alpha=a, l1_ratio=0.5, random_state=42, max_iter=10000)
                    en.fit(X_train_scaled, y_train)
                    non_zero = [f for f, c in zip(feature_names, en.coef_) if abs(c) > 1e-4]
                    if len(non_zero) == k:
                        best_features = non_zero
                        break
                if not best_features:
                    en = ElasticNet(alpha=10.0, l1_ratio=0.5, random_state=42, max_iter=10000)
                    en.fit(X_train_scaled, y_train)
                    order = np.argsort(np.abs(en.coef_))[::-1]
                    best_features = [feature_names[i] for i in order[:k]]
                selected = best_features
                
            # 建立評估模型 (使用選取特徵在訓練集擬合 OLS)
            model = LinearRegression()
            model.fit(X_train[selected], y_train)
            
            # 計算 5-Fold Cross Validation RMSE
            cv_scores = cross_val_score(LinearRegression(), X_train[selected], y_train, cv=cv, scoring='neg_root_mean_squared_error')
            cv_rmse_mean = round(float(-np.mean(cv_scores)), 2)
            cv_rmse_std = round(float(np.std(cv_scores)), 2)
            
            # 預測
            y_train_pred = model.predict(X_train[selected])
            y_test_pred = model.predict(X_test[selected])
            
            train_metrics = compute_metrics(y_train, y_train_pred, k, len(y_train))
            test_metrics = compute_metrics(y_test, y_test_pred, k, len(y_test))
            
            train_test_gap = round(float(test_metrics['rmse'] - train_metrics['rmse']), 2)
            
            # 統計推論與 VIF (statsmodels)
            coefficients, f_stat, f_pvalue, ols_res = compute_vif_and_stats(X_train[selected], y_train, selected)
            
            # 殘差與 Top Outliers
            residuals = []
            for i, idx in enumerate(test_indices):
                act = float(y_test.iloc[i])
                pred_val = float(y_test_pred[i])
                res = act - pred_val
                abs_err = abs(res)
                pct_err = (abs_err / act * 100.0) if act != 0 else 0.0
                residuals.append({
                    'index': int(idx),
                    'actual': round(act, 2),
                    'predicted': round(pred_val, 2),
                    'residual': round(res, 2),
                    'abs_error': round(abs_err, 2),
                    'pct_error': round(pct_err, 2)
                })
                
            # 依絕對誤差降冪排序
            outliers = sorted(residuals, key=lambda x: x['abs_error'], reverse=True)
            for rank, item in enumerate(outliers, 1):
                item['rank'] = rank
                
            # 迴歸方程式字串
            eq_parts = [f"{round(float(model.intercept_), 2):,}"]
            for feat, coef in zip(selected, model.coef_):
                sign = "+" if coef >= 0 else "-"
                eq_parts.append(f"{sign} {abs(coef):.4f} × [{feat}]")
            equation = f"Profit = {' '.join(eq_parts)}"
            
            results[algo].append({
                'k': k,
                'algorithm': algo,
                'features': selected,
                'equation': equation,
                'intercept': round(float(model.intercept_), 2),
                'weights': {f: round(float(c), 4) for f, c in zip(selected, model.coef_)},
                'train_rmse': train_metrics['rmse'],
                'test_rmse': test_metrics['rmse'],
                'train_test_gap': train_test_gap,
                'cv_rmse_mean': cv_rmse_mean,
                'cv_rmse_std': cv_rmse_std,
                'rmse': test_metrics['rmse'],  # Alias for compatibility
                'mse': test_metrics['mse'],
                'mae': test_metrics['mae'],
                'r2': test_metrics['r2'],
                'adj_r2': test_metrics['adj_r2'],
                'train_r2': train_metrics['r2'],
                'f_stat': f_stat,
                'f_pvalue': f_pvalue,
                'coefficients': coefficients,
                'residuals': residuals,
                'outliers': outliers
            })
            
    return results


def run_optuna_optimization(prep_data, n_trials=50):
    """
    CRISP-DM Step 5: Optuna 最佳化模組
    在訓練資料上透過 5-Fold CV 尋找 Ridge, Lasso, ElasticNet 之最佳超參數
    """
    X_train = prep_data['X_train']
    y_train = prep_data['y_train']
    X_test = prep_data['X_test']
    y_test = prep_data['y_test']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    trials_history = []
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['Ridge', 'Lasso', 'ElasticNet'])
        alpha = trial.suggest_float('alpha', 1e-4, 1e4, log=True)
        
        if model_type == 'Ridge':
            model = Ridge(alpha=alpha, random_state=42)
            l1_ratio = None
        elif model_type == 'Lasso':
            model = Lasso(alpha=alpha, random_state=42, max_iter=10000)
            l1_ratio = 1.0
        else:
            l1_ratio = trial.suggest_float('l1_ratio', 0.05, 0.95)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000)
            
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error')
        cv_rmse = float(-np.mean(scores))
        
        trials_history.append({
            'trial_number': trial.number,
            'model_type': model_type,
            'alpha': round(alpha, 6),
            'l1_ratio': round(l1_ratio, 4) if l1_ratio is not None else None,
            'cv_rmse': round(cv_rmse, 2)
        })
        
        return cv_rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_cv_rmse = round(float(study.best_value), 2)
    
    # 使用最佳超參數訓練最終模型並於測試集評估
    b_type = best_params['model_type']
    b_alpha = best_params['alpha']
    if b_type == 'Ridge':
        best_model = Ridge(alpha=b_alpha, random_state=42)
    elif b_type == 'Lasso':
        best_model = Lasso(alpha=b_alpha, random_state=42, max_iter=10000)
    else:
        best_model = ElasticNet(alpha=b_alpha, l1_ratio=best_params['l1_ratio'], random_state=42, max_iter=10000)
        
    best_model.fit(X_train_scaled, y_train)
    y_test_pred = best_model.predict(X_test_scaled)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = round(float(np.sqrt(test_mse)), 2)
    test_r2 = round(float(r2_score(y_test, y_test_pred)), 4)
    test_mae = round(float(mean_absolute_error(y_test, y_test_pred)), 2)
    
    # 儲存最佳模型
    save_payload = {
        'model': best_model,
        'scaler': scaler,
        'best_params': best_params,
        'best_cv_rmse': best_cv_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'feature_names': prep_data['feature_names'],
        'ohe': prep_data['ohe']
    }
    
    optuna_save_path = os.path.join(MODELS_DIR, 'optuna_best_model.pkl.gz')
    with gzip.open(optuna_save_path, 'wb') as f:
        pickle.dump(save_payload, f)
        
    return {
        'best_params': best_params,
        'best_cv_rmse': best_cv_rmse,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'trials_count': len(study.trials),
        'trials_history': sorted(trials_history, key=lambda x: x['cv_rmse'])[:20],
        'saved_model_path': optuna_save_path
    }


def save_pipeline_models(prep_data, fs_results):
    """
    CRISP-DM Step 6:
    使用 pickle + gzip 保存預處理管線與訓練完成之特徵選取模型
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    payload = {
        'ohe': prep_data['ohe'],
        'feature_names': prep_data['feature_names'],
        'random_state': prep_data['random_state'],
        'fs_results': fs_results
    }
    
    pipeline_path = os.path.join(MODELS_DIR, 'model_pipeline.pkl.gz')
    with gzip.open(pipeline_path, 'wb') as f:
        pickle.dump(payload, f)
        
    return pipeline_path


def load_pipeline_models(pipeline_path=None):
    """載入 gzip + pickle 模型管線"""
    if pipeline_path is None:
        pipeline_path = os.path.join(MODELS_DIR, 'model_pipeline.pkl.gz')
        
    with gzip.open(pipeline_path, 'rb') as f:
        payload = pickle.load(f)
    return payload


def predict_profit(rd_spend, admin, marketing, state, algorithm='Forward SFS', k=3, pipeline=None):
    """
    即時獲利預測推論函式
    """
    if pipeline is None:
        pipeline = load_pipeline_models()
        
    ohe = pipeline['ohe']
    fs_results = pipeline['fs_results']
    
    # 取得指定演算法與 k 的模型資訊
    algo_models = fs_results.get(algorithm, fs_results['Forward SFS'])
    target_model = next((m for m in algo_models if m['k'] == k), algo_models[2])
    
    # 前處理輸入
    state_df = pd.DataFrame([{'State': state}])
    encoded_state = ohe.transform(state_df[['State']])[0]
    dummy_cols = list(ohe.get_feature_names_out(['State']))
    
    sample = {
        'R&D Spend': float(rd_spend),
        'Administration': float(admin),
        'Marketing Spend': float(marketing)
    }
    for col_name, val in zip(dummy_cols, encoded_state):
        sample[col_name] = float(val)
        
    # 計算預測值
    predicted = target_model['intercept']
    breakdown = {'Intercept': target_model['intercept']}
    
    for feat, weight in target_model['weights'].items():
        val = sample.get(feat, 0.0)
        contrib = round(val * weight, 2)
        predicted += contrib
        breakdown[feat] = contrib
        
    return {
        'predicted_profit': round(predicted, 2),
        'algorithm': algorithm,
        'k': k,
        'selected_features': target_model['features'],
        'equation': target_model['equation'],
        'breakdown': breakdown,
        'input_data': sample
    }


if __name__ == '__main__':
    print("Executing CRISP-DM Machine Learning Pipeline...")
    data_bundle = load_dataset()
    print(f"Loaded {data_bundle['shape']['rows']} rows.")
    
    prep = prepare_data(data_bundle['dataframe'], random_state=42)
    print("Preprocessing completed (One-Hot Encoded, Data Leakage prevented).")
    
    fs_res = run_feature_selection_algorithms(prep)
    print("Feature Selection evaluated across 6 algorithms.")
    
    saved_path = save_pipeline_models(prep, fs_res)
    print(f"Saved models to: {saved_path}")
    
    opt_res = run_optuna_optimization(prep, n_trials=30)
    print(f"Optuna Best Params: {opt_res['best_params']}, Best CV RMSE: {opt_res['best_cv_rmse']}")
    
    pred_res = predict_profit(100000, 120000, 250000, 'New York', algorithm='Forward SFS', k=3)
    print(f"Sample Prediction for NY: ${pred_res['predicted_profit']:,.2f}")
