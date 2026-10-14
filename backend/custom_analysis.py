import io
import urllib.request
import urllib.parse
import ipaddress
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import itertools

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def is_safe_url(url_str: str) -> bool:
    """SSRF validation for URL import"""
    try:
        parsed = urllib.parse.urlparse(url_str)
        if parsed.scheme not in ('http', 'https'):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # Check IP
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                return False
        except ValueError:
            # Domain name, check blacklist keywords
            lower_h = hostname.lower()
            if any(lower_h.startswith(p) for p in ['localhost', '127.', '10.', '192.168.', '0.0.0.0', '169.254.']):
                return False
            if 'internal' in lower_h or 'local' in lower_h:
                return False
        return True
    except Exception:
        return False


def fetch_csv_from_url(url_str: str, max_bytes: int = 5 * 1024 * 1024) -> str:
    """Fetch CSV safely from URL with timeout and byte limit"""
    if not is_safe_url(url_str):
        raise ValueError("URL rejected: Potentially unsafe destination or private address.")
    
    req = urllib.request.Request(
        url_str,
        headers={'User-Agent': 'Mozilla/5.0 (AIoT Multi-Dataset Platform; +https://github.com/Augustine-723/50-Starup)'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        content_type = resp.headers.get('Content-Type', '')
        # Read limited bytes
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"File exceeds maximum allowed size of {max_bytes / (1024*1024):.1f}MB.")
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return data.decode('latin-1')


def parse_and_profile_csv(csv_text: str) -> Dict[str, Any]:
    """Parse CSV text and return summary profile for dataset configuration"""
    df = pd.read_csv(io.StringIO(csv_text))
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    
    columns_info = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        unique_count = df[col].nunique()
        columns_info.append({
            'name': col,
            'is_numeric': bool(is_numeric),
            'dtype': dtype_str,
            'missing_count': int(missing.get(col, 0)),
            'unique_count': int(unique_count),
            'sample_values': df[col].dropna().head(3).tolist()
        })
        
    preview_rows = df.head(10).replace({np.nan: None}).to_dict(orient='records')
    
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': columns_info,
        'preview': preview_rows,
        'column_names': list(df.columns)
    }


def run_custom_regression_workflow(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Universal reusable regression workflow:
    - Preprocessing with anti-leakage imputation & OHE
    - 6 Feature Selection algorithms
    - 5-Fold Cross Validation on Train
    - Metrics, OLS statistical inference & residual analysis
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    if not feature_columns:
        feature_columns = [c for c in df.columns if c != target_column]
        
    # Verify target is numeric
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        try:
            df[target_column] = pd.to_numeric(df[target_column])
        except Exception:
            raise ValueError(f"Target column '{target_column}' must be numeric for Linear Regression.")
            
    # Drop rows where target is missing
    df = df.dropna(subset=[target_column])
    
    X_raw = df[feature_columns]
    y = df[target_column]
    
    # Train / Test split first to prevent Data Leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )
    
    # Identify numeric and categorical feature columns
    num_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    cat_cols = [c for c in feature_columns if c not in num_cols]
    
    # Impute missing values (fit on train)
    train_num_medians = {c: X_train_raw[c].median() for c in num_cols}
    train_cat_modes = {c: X_train_raw[c].mode()[0] if len(X_train_raw[c].mode()) > 0 else 'Unknown' for c in cat_cols}
    
    X_train_imp = X_train_raw.copy()
    X_test_imp = X_test_raw.copy()
    for c in num_cols:
        X_train_imp[c] = X_train_imp[c].fillna(train_num_medians[c])
        X_test_imp[c] = X_test_imp[c].fillna(train_num_medians[c])
    for c in cat_cols:
        X_train_imp[c] = X_train_imp[c].fillna(train_cat_modes[c])
        X_test_imp[c] = X_test_imp[c].fillna(train_cat_modes[c])
        
    # One-Hot Encoding for categorical features (drop='first')
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        ohe.fit(X_train_imp[cat_cols])
        encoded_cols = list(ohe.get_feature_names_out(cat_cols))
        
        train_cat_arr = ohe.transform(X_train_imp[cat_cols])
        test_cat_arr = ohe.transform(X_test_imp[cat_cols])
        
        df_train_cat = pd.DataFrame(train_cat_arr, columns=encoded_cols, index=X_train_imp.index)
        df_test_cat = pd.DataFrame(test_cat_arr, columns=encoded_cols, index=X_test_imp.index)
        
        X_train_enc = pd.concat([X_train_imp[num_cols], df_train_cat], axis=1)
        X_test_enc = pd.concat([X_test_imp[num_cols], df_test_cat], axis=1)
    else:
        X_train_enc = X_train_imp[num_cols]
        X_test_enc = X_test_imp[num_cols]
        
    all_features = list(X_train_enc.columns)
    max_k = min(len(all_features), 8)
    if max_k < 1:
        raise ValueError("At least 1 usable feature is required for regression.")
        
    kf = KFold(n_splits=min(5, len(X_train_enc)), shuffle=True, random_state=random_state)
    
    # Helper to evaluate model
    def evaluate_model(feats: List[str], k_val: int):
        X_tr = X_train_enc[feats]
        X_te = X_test_enc[feats]
        
        model = LinearRegression()
        model.fit(X_tr, y_train)
        
        # 5-Fold CV on training set
        cv_scores = cross_val_score(LinearRegression(), X_tr, y_train, cv=kf, scoring='neg_mean_squared_error')
        cv_rmse_scores = np.sqrt(-cv_scores)
        cv_rmse_mean = float(np.mean(cv_rmse_scores))
        cv_rmse_std = float(np.std(cv_rmse_scores))
        
        y_train_pred = model.predict(X_tr)
        y_test_pred = model.predict(X_te)
        
        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        mse = float(mean_squared_error(y_test, y_test_pred))
        mae = float(mean_absolute_error(y_test, y_test_pred))
        r2 = float(r2_score(y_test, y_test_pred))
        
        # Adj R2
        n_test = len(y_test)
        p_val = len(feats)
        if n_test - p_val - 1 > 0:
            adj_r2 = 1.0 - (1.0 - r2) * (n_test - 1) / (n_test - p_val - 1)
        else:
            adj_r2 = r2
            
        gap = float(test_rmse - train_rmse)
        
        # statsmodels for p-values, SE, CI, VIF
        try:
            X_tr_sm = sm.add_constant(X_tr, has_constant='add')
            ols_res = sm.OLS(y_train, X_tr_sm).fit()
            
            p_values = {k: float(v) for k, v in ols_res.pvalues.items()}
            bse = {k: float(v) for k, v in ols_res.bse.items()}
            tvalues = {k: float(v) for k, v in ols_res.tvalues.items()}
            conf_int = {k: [float(c[0]), float(c[1])] for k, c in ols_res.conf_int().iterrows()}
            
            # VIF
            vif_dict = {}
            if len(feats) > 1:
                for i, col in enumerate(X_tr.columns):
                    try:
                        v = variance_inflation_factor(X_tr.values, i)
                        vif_dict[col] = 999.0 if np.isinf(v) or np.isnan(v) else float(v)
                    except Exception:
                        vif_dict[col] = 1.0
            else:
                vif_dict[feats[0]] = 1.0
        except Exception:
            p_values = {f: 0.05 for f in feats}
            bse = {f: 0.0 for f in feats}
            tvalues = {f: 1.0 for f in feats}
            conf_int = {f: [0.0, 0.0] for f in feats}
            vif_dict = {f: 1.0 for f in feats}
            
        weights = {feat: float(coef) for feat, coef in zip(feats, model.coef_)}
        
        # Build equation
        eq_parts = [f"{model.intercept_:.2f}"]
        for f, w in weights.items():
            sign = "+" if w >= 0 else "-"
            eq_parts.append(f"{sign} {abs(w):.4f} * ({f})")
        equation = f"{target_column} = " + " ".join(eq_parts)
        
        return {
            'k': k_val,
            'features': feats,
            'train_rmse': round(train_rmse, 2),
            'test_rmse': round(test_rmse, 2),
            'cv_rmse_mean': round(cv_rmse_mean, 2),
            'cv_rmse_std': round(cv_rmse_std, 2),
            'mse': round(mse, 2),
            'mae': round(mae, 2),
            'r2': round(r2, 4),
            'adj_r2': round(adj_r2, 4),
            'train_test_gap': round(gap, 2),
            'intercept': round(float(model.intercept_), 2),
            'weights': {k: round(v, 4) for k, v in weights.items()},
            'equation': equation,
            'p_values': {k: round(v, 4) for k, v in p_values.items()},
            'std_errors': {k: round(v, 4) for k, v in bse.items()},
            't_values': {k: round(v, 4) for k, v in tvalues.items()},
            'confidence_intervals': {k: [round(c[0], 2), round(c[1], 2)] for k, c in conf_int.items()},
            'vif': {k: round(v, 2) for k, v in vif_dict.items()}
        }
        
    fs_results = {
        'Forward SFS': [],
        'RFE': [],
        'Exhaustive Search': [],
        'Baseline OLS': [],
        'Lasso': [],
        'ElasticNet': []
    }
    
    # 1. Baseline OLS: Order by absolute Pearson correlation with y_train
    corr_series = X_train_enc.apply(lambda col: np.abs(np.corrcoef(col, y_train)[0, 1]))
    sorted_features = corr_series.sort_values(ascending=False).index.tolist()
    for k in range(1, max_k + 1):
        feats = sorted_features[:k]
        fs_results['Baseline OLS'].append(evaluate_model(feats, k))
        
    # 2. Forward SFS: Greedy forward selection by training RSS
    sfs_feats: List[str] = []
    remaining = list(all_features)
    for k in range(1, max_k + 1):
        best_candidate = remaining[0]
        best_rss = float('inf')
        for cand in remaining:
            trial_feats = sfs_feats + [cand]
            lr = LinearRegression().fit(X_train_enc[trial_feats], y_train)
            rss = float(np.sum((y_train - lr.predict(X_train_enc[trial_feats])) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_candidate = cand
        sfs_feats.append(best_candidate)
        remaining.remove(best_candidate)
        fs_results['Forward SFS'].append(evaluate_model(list(sfs_feats), k))
        
    # 3. RFE: Backward elimination by smallest absolute coefficient
    rfe_feats = list(all_features)
    # Fit full model and eliminate down
    while len(rfe_feats) > max_k:
        lr = LinearRegression().fit(X_train_enc[rfe_feats], y_train)
        least_idx = int(np.argmin(np.abs(lr.coef_)))
        rfe_feats.pop(least_idx)
    
    rfe_history = {len(rfe_feats): list(rfe_feats)}
    while len(rfe_feats) > 1:
        lr = LinearRegression().fit(X_train_enc[rfe_feats], y_train)
        least_idx = int(np.argmin(np.abs(lr.coef_)))
        rfe_feats.pop(least_idx)
        rfe_history[len(rfe_feats)] = list(rfe_feats)
        
    for k in range(1, max_k + 1):
        feats = rfe_history.get(k, sorted_features[:k])
        fs_results['RFE'].append(evaluate_model(feats, k))
        
    # 4. Exhaustive Search: Best combination by RSS, capped at top-8 candidates
    exhaustive_candidates = sorted_features[:8]
    for k in range(1, max_k + 1):
        best_rss = float('inf')
        best_combo = exhaustive_candidates[:k]
        for combo in itertools.combinations(exhaustive_candidates, k):
            c_feats = list(combo)
            lr = LinearRegression().fit(X_train_enc[c_feats], y_train)
            rss = float(np.sum((y_train - lr.predict(X_train_enc[c_feats])) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_combo = c_feats
        fs_results['Exhaustive Search'].append(evaluate_model(best_combo, k))

        
    # 5. Lasso
    for k in range(1, max_k + 1):
        feats = sorted_features[:k]
        # Adjust alpha dynamically based on std of y
        y_std = np.std(y_train)
        lasso = Lasso(alpha=max(0.1, y_std * 0.05), random_state=random_state, max_iter=2000)
        lasso.fit(X_train_enc, y_train)
        non_zero = [f for f, c in zip(all_features, lasso.coef_) if abs(c) > 1e-4]
        if len(non_zero) < k:
            for f in sorted_features:
                if f not in non_zero:
                    non_zero.append(f)
                if len(non_zero) == k:
                    break
        fs_results['Lasso'].append(evaluate_model(non_zero[:k], k))
        
    # 6. ElasticNet
    for k in range(1, max_k + 1):
        feats = sorted_features[:k]
        y_std = np.std(y_train)
        enet = ElasticNet(alpha=max(0.1, y_std * 0.05), l1_ratio=0.5, random_state=random_state, max_iter=2000)
        enet.fit(X_train_enc, y_train)
        non_zero = [f for f, c in zip(all_features, enet.coef_) if abs(c) > 1e-4]
        if len(non_zero) < k:
            for f in sorted_features:
                if f not in non_zero:
                    non_zero.append(f)
                if len(non_zero) == k:
                    break
        fs_results['ElasticNet'].append(evaluate_model(non_zero[:k], k))
        
    # Pick default best model based on CV RMSE
    all_models = []
    for algo, models in fs_results.items():
        for m in models:
            all_models.append({**m, 'algo': algo})
    best_overall_cv = sorted(all_models, key=lambda x: x['cv_rmse_mean'])[0]
    best_overall_test = sorted(all_models, key=lambda x: x['test_rmse'])[0]
    
    # Generate test set predictions and residuals for the best CV model
    best_model_obj = LinearRegression().fit(X_train_enc[best_overall_cv['features']], y_train)
    y_test_pred_best = best_model_obj.predict(X_test_enc[best_overall_cv['features']])
    residuals = y_test.values - y_test_pred_best
    
    residual_points = []
    for actual, pred, res in zip(y_test.values, y_test_pred_best, residuals):
        residual_points.append({
            'actual': round(float(actual), 2),
            'predicted': round(float(pred), 2),
            'residual': round(float(res), 2),
            'abs_residual': round(float(abs(res)), 2)
        })
        
    return {
        'status': 'success',
        'target_column': target_column,
        'features_count': len(feature_columns),
        'encoded_features_count': len(all_features),
        'all_features': all_features,
        'train_samples': len(X_train_raw),
        'test_samples': len(X_test_raw),
        'max_k': max_k,
        'feature_selection_results': fs_results,
        'best_cv_model': best_overall_cv,
        'best_test_model': best_overall_test,
        'residual_analysis': {
            'points': residual_points,
            'mean_residual': round(float(np.mean(residuals)), 2),
            'std_residual': round(float(np.std(residuals)), 2),
            'max_error': round(float(np.max(np.abs(residuals))), 2)
        }
    }
