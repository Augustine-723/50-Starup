import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import (
    load_dataset,
    prepare_data,
    run_feature_selection_algorithms,
    run_optuna_optimization,
    save_pipeline_models,
    load_pipeline_models,
    predict_profit
)

app = FastAPI(
    title="50 Startups ML Regression & Feature Selection API",
    description="Backend API powering the 50 Startups Multiple Linear Regression Explorer",
    version="2.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache initialized pipelines in memory for fast response
CACHE: Dict[int, Any] = {}
OPTUNA_CACHE: Optional[Dict[str, Any]] = None


def get_cached_pipeline(random_state: int = 42):
    """取得或初始化指定 random_state 的管線結果"""
    global CACHE, OPTUNA_CACHE
    if random_state not in CACHE:
        data_bundle = load_dataset()
        prep = prepare_data(data_bundle['dataframe'], random_state=random_state)
        fs_res = run_feature_selection_algorithms(prep)
        save_pipeline_models(prep, fs_res)
        
        CACHE[random_state] = {
            'data_bundle': data_bundle,
            'prep': prep,
            'fs_results': fs_res
        }
        
    return CACHE[random_state]


@app.on_event("startup")
def startup_event():
    """伺服器啟動時預先暖機預設 random_state=42 與 random_state=0"""
    print("[*] Initializing pipelines for random_state=42 and random_state=0...")
    get_cached_pipeline(42)
    get_cached_pipeline(0)
    print("[+] Startup initialization completed.")


class PredictionRequest(BaseModel):
    rd_spend: float = Field(..., ge=0, description="R&D Spend in USD")
    administration: float = Field(..., ge=0, description="Administration Spend in USD")
    marketing_spend: float = Field(..., ge=0, description="Marketing Spend in USD")
    state: str = Field(..., description="State name: New York, California, or Florida")
    algorithm: Optional[str] = Field("Forward SFS", description="Algorithm name")
    k: Optional[int] = Field(3, ge=1, le=5, description="Number of features (1-5)")
    random_state: Optional[int] = Field(42, description="Random state seed (42 or 0)")


@app.get("/")
def root():
    return {
        "message": "50 Startups Multiple Linear Regression API is running.",
        "docs_url": "/docs",
        "endpoints": [
            "GET /api/health",
            "GET /api/results",
            "GET /api/models",
            "GET /api/optuna",
            "POST /api/predict",
            "POST /api/custom-analyze",
            "GET /api/proxy-csv"
        ]
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "backend_mode": "external_fastapi",
        "message": "FastAPI ML Backend is healthy and connected"
    }



@app.get("/api/results")
def get_results(random_state: int = Query(42, description="Train-Test Split Seed")):
    """
    獲取完整的 CRISP-DM 分析數據：
    - 資料集基本統計與相關矩陣
    - 6 種特徵選擇演算法在 k=1..5 之完整評估指標
    - 各特徵之係數、p-value、95% CI 與 VIF
    - 殘差分佈與 Top Outliers
    """
    try:
        pipe = get_cached_pipeline(random_state)
        data_bundle = pipe['data_bundle']
        fs_results = pipe['fs_results']
        
        # 移除不可 JSON 序列化之 dataframe 物件
        clean_data_info = {k: v for k, v in data_bundle.items() if k != 'dataframe'}
        
        return {
            'status': 'success',
            'random_state': random_state,
            'data_info': clean_data_info,
            'feature_selection_results': fs_results,
            'benchmark_note': (
                "當 random_state=0 時，k=2 之 R&D Spend + Marketing Spend 測試集 RMSE 恰為 8,198.80，"
                "與參考網站 (AIoT-DIC2-MultipleLR) 數值 100% 吻合。"
                if random_state == 0 else
                "預設遵循作業規範 random_state=42，k=2 測試集 RMSE 為 8,206.33；最佳模型為 k=3 (RMSE: 7,946.37, R²: 0.9220)。"
            )
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
def get_models_metadata(random_state: int = Query(42)):
    """返回支援的演算法、特徵欄位與各模型精簡清單"""
    pipe = get_cached_pipeline(random_state)
    fs_results = pipe['fs_results']
    
    summary = {}
    for algo, models in fs_results.items():
        summary[algo] = [
            {
                'k': m['k'],
                'features': m['features'],
                'test_rmse': m.get('test_rmse', m.get('rmse')),
                'test_r2': m.get('test_r2', m.get('r2')),
                'train_test_gap': m.get('train_test_gap', 0),
                'equation': m['equation']
            }
            for m in models
        ]
        
    return {
        'status': 'success',
        'random_state': random_state,
        'algorithms': list(fs_results.keys()),
        'candidate_features': pipe['prep']['feature_names'],
        'models_summary': summary
    }


@app.get("/api/optuna")
def get_optuna_results(n_trials: int = Query(30, ge=10, le=100)):
    """執行或返回 Optuna 超參數最佳化成果"""
    global OPTUNA_CACHE
    if OPTUNA_CACHE is None:
        pipe = get_cached_pipeline(42)
        OPTUNA_CACHE = run_optuna_optimization(pipe['prep'], n_trials=n_trials)
        
    return {
        'status': 'success',
        'optuna': OPTUNA_CACHE
    }


@app.post("/api/predict")
def predict(req: PredictionRequest):
    """
    即時獲利預測 API
    接收使用者各項支出輸入與州別，返回預測利潤與詳細係數分解
    """
    valid_states = ['New York', 'California', 'Florida']
    if req.state not in valid_states:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state '{req.state}'. Allowed states: {valid_states}"
        )
        
    pipe = get_cached_pipeline(req.random_state or 42)
    pipeline_obj = {
        'ohe': pipe['prep']['ohe'],
        'fs_results': pipe['fs_results']
    }
    
    algo = req.algorithm or "Forward SFS"
    if algo not in pipe['fs_results']:
        algo = "Forward SFS"
        
    result = predict_profit(
        rd_spend=req.rd_spend,
        admin=req.administration,
        marketing=req.marketing_spend,
        state=req.state,
        algorithm=algo,
        k=req.k or 3,
        pipeline=pipeline_obj
    )
    
    return {
        'status': 'success',
        'prediction': result
    }


class CustomAnalysisRequest(BaseModel):
    csv_text: Optional[str] = Field(None, description="Raw CSV string")
    url: Optional[str] = Field(None, description="Public CSV URL")
    target_column: Optional[str] = Field(None, description="Target column name")
    feature_columns: Optional[List[str]] = Field(None, description="List of feature column names")
    test_size: Optional[float] = Field(0.2, ge=0.05, le=0.5, description="Test split fraction")
    random_state: Optional[int] = Field(42, description="Random seed")
    action: Optional[str] = Field("analyze", description="'profile' or 'analyze'")


@app.get("/api/proxy-csv")
def proxy_csv(url: str = Query(..., description="Public CSV URL to fetch safely")):
    """SSRF-safe CSV proxy endpoint"""
    try:
        from custom_analysis import fetch_csv_from_url
        content = fetch_csv_from_url(url)
        return {
            'status': 'success',
            'url': url,
            'size': len(content),
            'csv_text': content
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/custom-analyze")
def custom_analyze(req: CustomAnalysisRequest):
    """Universal ML regression workflow endpoint for arbitrary tabular datasets"""
    try:
        from custom_analysis import fetch_csv_from_url, parse_and_profile_csv, run_custom_regression_workflow
        import pandas as pd
        import io

        csv_str = req.csv_text
        if not csv_str and req.url:
            csv_str = fetch_csv_from_url(req.url)

        if not csv_str:
            raise HTTPException(status_code=400, detail="Either 'csv_text' or 'url' must be provided.")

        if req.action == 'profile':
            profile = parse_and_profile_csv(csv_str)
            return {'status': 'success', 'profile': profile}

        # Otherwise run full analysis
        df = pd.read_csv(io.StringIO(csv_str))
        target_col = req.target_column
        if not target_col:
            # Pick last numeric column by default
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not num_cols:
                raise HTTPException(status_code=400, detail="No numeric columns found in dataset.")
            target_col = num_cols[-1]

        feats = req.feature_columns
        if not feats:
            feats = [c for c in df.columns if c != target_col]

        res = run_custom_regression_workflow(
            df=df,
            target_column=target_col,
            feature_columns=feats,
            test_size=req.test_size or 0.2,
            random_state=req.random_state or 42
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    print("[*] Starting FastAPI server on http://localhost:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)

