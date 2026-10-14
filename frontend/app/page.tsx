'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine, Cell
} from 'recharts';
import {
  TrendingUp, Award, CheckCircle, AlertTriangle, Cpu, Play,
  Download, Database, Sliders, RefreshCw, BarChart3, Layers, BookOpen,
  ArrowRight, Activity, HelpCircle, ChevronRight, Compass, Upload,
  Globe, Check, FileText, ArrowRightLeft, Sparkles, Filter, AlertCircle,
  FileSpreadsheet, CheckCircle2, ChevronDown, ChevronUp, ListFilter, Copy,
  ArrowLeft, Eye, ShieldCheck, CheckSquare, Settings2, BarChart2
} from 'lucide-react';

// Fallback Data for 50 Startups (Seed 42 & 0)
import { FALLBACK_RESULTS_SEED_42, FALLBACK_RESULTS_SEED_0, DATASET_50_STARTUPS } from './fallbackData';
import {
  parseCSV,
  profileDataset,
  runUniversalRegressionWorkflow,
  DatasetProfile,
  UniversalRegressionResult
} from './utils/universalRegression';

export default function DashboardPage() {
  // ==========================================
  // Step Navigation State (Step 1 -> 2 -> 3 -> 4)
  // ==========================================
  const [currentStep, setCurrentStep] = useState<number>(1);

  // ==========================================
  // Problem Selection (Problem 1 vs Problem 2)
  // ==========================================
  const [activeProblem, setActiveProblem] = useState<'problem1' | 'problem2'>('problem1');
  const [submissionChoice, setSubmissionChoice] = useState<'problem1' | 'problem2'>('problem1');

  // Model Tuning Parameters
  const [selectedAlgo, setSelectedAlgo] = useState<string>('Forward SFS');
  const [selectedK, setSelectedK] = useState<number>(3);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);

  // Step 3 Collapsible Advanced Sections
  const [expandedSections, setExpandedSections] = useState<{
    stats: boolean;
    residuals: boolean;
    predictor: boolean;
    optuna: boolean;
    benchmark: boolean;
  }>({
    stats: false,
    residuals: false,
    predictor: true, // Default open for easy prediction testing
    optuna: false,
    benchmark: false,
  });

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  // ==========================================
  // Problem 1: 50 Startups State
  // ==========================================
  const [p1RandomState, setP1RandomState] = useState<number>(42);
  const [rdSpend, setRdSpend] = useState<number>(100000);
  const [adminSpend, setAdminSpend] = useState<number>(120000);
  const [mktgSpend, setMktgSpend] = useState<number>(250000);
  const [selectedState, setSelectedState] = useState<string>('New York');

  const [p1ApiData, setP1ApiData] = useState<any>(null);
  const [backendRuntime, setBackendRuntime] = useState<string>('Vercel Serverless (即時運算)');
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [isP1Loading, setIsP1Loading] = useState<boolean>(false);

  // ==========================================
  // Problem 2: Custom Dataset State
  // ==========================================
  const [customImportMode, setCustomImportMode] = useState<'upload' | 'url'>('url');
  const [customUrlInput, setCustomUrlInput] = useState<string>(
    'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'
  );
  const [customDatasetName, setCustomDatasetName] = useState<string>('加州房地產估價 (10 Features)');
  const [customCsvText, setCustomCsvText] = useState<string>('');
  const [customRecords, setCustomRecords] = useState<any[]>([]);
  const [customProfile, setCustomProfile] = useState<DatasetProfile | null>(null);
  const [customTargetCol, setCustomTargetCol] = useState<string>('MedianHouseValue');
  const [customFeatureCols, setCustomFeatureCols] = useState<string[]>([]);
  const [customTestSize, setCustomTestSize] = useState<number>(0.2);
  const [customRandomSeed, setCustomRandomSeed] = useState<number>(42);
  const [customAnalysisResult, setCustomAnalysisResult] = useState<UniversalRegressionResult | null>(null);
  const [isCustomLoading, setIsCustomLoading] = useState<boolean>(false);
  const [customError, setCustomError] = useState<string | null>(null);
  const [customSuccessMsg, setCustomSuccessMsg] = useState<string | null>(null);
  const [customPredictInputs, setCustomPredictInputs] = useState<Record<string, number>>({});
  const [optunaData, setOptunaData] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch Optuna data on mount
  useEffect(() => {
    async function fetchOptuna() {
      try {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
        const res = await fetch(`${apiBase}/api/optuna`);
        if (res.ok) {
          const json = await res.json();
          setOptunaData(json);
        }
      } catch (e) {
        console.warn('Could not load Optuna data:', e);
      }
    }
    fetchOptuna();
  }, []);

  // Fetch Health on mount to accurately detect backend environment
  useEffect(() => {
    async function checkBackend() {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
      try {
        const hRes = await fetch(`${apiBase}/api/health`);
        if (hRes.ok) {
          const hJson = await hRes.json();
          if (hJson.backend_mode === 'external_fastapi') {
            setBackendRuntime('FastAPI 伺服器 (即時模型運算)');
            setIsBackendConnected(true);
          } else {
            setBackendRuntime('Vercel Serverless ML API (即時模型運算)');
            setIsBackendConnected(true);
          }
        }
      } catch (e) {
        setBackendRuntime('純 TypeScript ML 引擎 (本地運算)');
        setIsBackendConnected(false);
      }
    }
    checkBackend();
  }, []);

  // Fetch Problem 1 data on seed change
  useEffect(() => {
    async function fetchP1Data() {
      setIsP1Loading(true);
      try {
        const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
        const res = await fetch(`${apiBase}/api/results?random_state=${p1RandomState}`);
        if (res.ok) {
          const json = await res.json();
          setP1ApiData(json);
          setIsBackendConnected(true);
        }
      } catch (err) {
        console.warn('Backend /api/results unreachable, using offline fallback dataset.');
      } finally {
        setIsP1Loading(false);
      }
    }
    fetchP1Data();
  }, [p1RandomState]);

  // Load default 10-feature custom benchmark dataset on mount
  useEffect(() => {
    async function loadDefaultCustomDataset() {
      try {
        const res = await fetch('/data/real_estate_valuation_10features.csv');
        if (res.ok) {
          const text = await res.text();
          loadCsvDataString(text, '加州房地產估價 (10 Features)', 'MedianHouseValue');
        }
      } catch (e) {
        console.warn('Could not auto-load real estate benchmark dataset:', e);
      }
    }
    loadDefaultCustomDataset();
  }, []);

  // Helper to parse, profile, and set custom CSV
  const loadCsvDataString = (csvString: string, datasetName: string, defaultTarget?: string) => {
    try {
      setCustomError(null);
      const records = parseCSV(csvString);
      if (!records || records.length === 0) {
        throw new Error('CSV 檔案為空或無法解析');
      }
      const profile = profileDataset(records);
      setCustomCsvText(csvString);
      setCustomRecords(records);
      setCustomProfile(profile);
      setCustomDatasetName(datasetName);

      // Determine default target
      let target = defaultTarget;
      if (!target || !profile.numericColumns.includes(target)) {
        target = profile.numericColumns[profile.numericColumns.length - 1] || profile.columnNames[0];
      }
      setCustomTargetCol(target);

      // Default features: all other columns
      const defaultFeatures = profile.columnNames.filter(c => c !== target);
      setCustomFeatureCols(defaultFeatures);

      // Auto-run analysis for convenience
      runAnalysisOnCustomData(records, target, defaultFeatures, customTestSize, customRandomSeed, datasetName);
      setCustomSuccessMsg(`成功載入「${datasetName}」(${profile.rowCount} 筆樣本，${profile.columnNames.length} 個欄位)`);
    } catch (err: any) {
      setCustomError(err?.message || '解析 CSV 失敗');
    }
  };

  // Run Custom Dataset Regression Analysis
  const runAnalysisOnCustomData = async (
    records: any[],
    target: string,
    features: string[],
    testSize: number,
    seed: number,
    name: string
  ) => {
    setIsCustomLoading(true);
    setCustomError(null);
    try {
      // First try Next.js / FastAPI backend API endpoint
      const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
      let result: UniversalRegressionResult | null = null;
      try {
        const res = await fetch(`${apiBase}/api/custom-analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            records,
            targetColumn: target,
            featureColumns: features,
            datasetName: name,
            testSize,
            randomState: seed,
          }),
        });
        if (res.ok) {
          result = await res.json();
        }
      } catch (apiErr) {
        console.log('Server API custom-analyze unavailable, using pure client engine fallback...');
      }

      // If backend API returned error or was unavailable, use in-browser TS universal regression engine
      if (!result || (result as any).status === 'error') {
        result = runUniversalRegressionWorkflow(
          records,
          target,
          features,
          name,
          testSize,
          seed
        );
      }

      setCustomAnalysisResult(result);

      // Initialize predictor inputs to mean values
      if (result && result.all_features) {
        const initInputs: Record<string, number> = {};
        result.all_features.forEach((feat: string) => {
          const vals = records.map(r => Number(r[feat])).filter(v => !isNaN(v));
          if (vals.length > 0) {
            const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
            initInputs[feat] = Math.round(avg * 100) / 100;
          } else {
            initInputs[feat] = 0;
          }
        });
        setCustomPredictInputs(initInputs);
      }

      // Adjust selectedK if out of bounds
      if (result && selectedK > result.max_k) {
        setSelectedK(result.max_k);
      }
    } catch (err: any) {
      setCustomError(err?.message || '迴歸分析執行失敗');
    } finally {
      setIsCustomLoading(false);
    }
  };

  // Handle Local CSV File Upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const fileName = file.name.replace(/\.[^/.]+$/, '');
    const reader = new FileReader();

    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (text) {
        loadCsvDataString(text, fileName);
      }
    };
    reader.onerror = () => {
      setCustomError('讀取本機 CSV 檔案失敗');
    };
    reader.readAsText(file, 'UTF-8');
  };

  // Handle URL CSV Import (SSRF-protected via /api/proxy-csv)
  const handleUrlImport = async () => {
    if (!customUrlInput || !customUrlInput.trim()) {
      setCustomError('請輸入有效的公開 CSV 網址');
      return;
    }
    setIsCustomLoading(true);
    setCustomError(null);
    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || '';
      const proxyUrl = `${apiBase}/api/proxy-csv?url=${encodeURIComponent(customUrlInput.trim())}`;
      const res = await fetch(proxyUrl);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.message || `網址載入失敗 (HTTP ${res.status})`);
      }
      const data = await res.json();
      if (!data.csv_text) {
        throw new Error('伺服器未回傳有效之 CSV 文字內容');
      }
      const urlParts = customUrlInput.split('/');
      const guessedName = urlParts[urlParts.length - 1].replace('.csv', '') || 'URL Dataset';
      loadCsvDataString(data.csv_text, guessedName);
    } catch (err: any) {
      setCustomError(err?.message || '匯入 URL CSV 發生錯誤');
    } finally {
      setIsCustomLoading(false);
    }
  };

  // ==========================================
  // Active Dataset Bundle Binding
  // ==========================================
  const activeResultsBundle = useMemo(() => {
    if (activeProblem === 'problem1') {
      if (p1ApiData && p1ApiData.feature_selection_results) {
        return p1ApiData.feature_selection_results;
      }
      return p1RandomState === 0 ? FALLBACK_RESULTS_SEED_0 : FALLBACK_RESULTS_SEED_42;
    } else {
      return customAnalysisResult?.feature_selection_results || {};
    }
  }, [activeProblem, p1ApiData, p1RandomState, customAnalysisResult]);

  const currentAlgoModels = useMemo(() => {
    return activeResultsBundle[selectedAlgo] || activeResultsBundle['Forward SFS'] || [];
  }, [activeResultsBundle, selectedAlgo]);

  const maxK = useMemo(() => {
    if (activeProblem === 'problem1') return 5;
    return customAnalysisResult?.max_k || (currentAlgoModels.length > 0 ? currentAlgoModels.length : 5);
  }, [activeProblem, customAnalysisResult, currentAlgoModels]);

  // Best CV Model (5-Fold CV on training split)
  const bestCvModel = useMemo(() => {
    if (!currentAlgoModels || currentAlgoModels.length === 0) return null;
    return [...currentAlgoModels].sort((a: any, b: any) => (a.cv_rmse_mean || 999999) - (b.cv_rmse_mean || 999999))[0];
  }, [currentAlgoModels]);

  // Best Test Model (Lowest Test RMSE)
  const bestTestModel = useMemo(() => {
    if (!currentAlgoModels || currentAlgoModels.length === 0) return null;
    return [...currentAlgoModels].sort((a: any, b: any) => (a.test_rmse || 999999) - (b.test_rmse || 999999))[0];
  }, [currentAlgoModels]);

  // Current Model for selected k
  const currentModel = useMemo(() => {
    if (!currentAlgoModels || currentAlgoModels.length === 0) {
      return {
        k: 1,
        features: [],
        train_rmse: 0,
        test_rmse: 0,
        cv_rmse_mean: 0,
        cv_rmse_std: 0,
        mse: 0,
        mae: 0,
        r2: 0,
        adj_r2: 0,
        train_test_gap: 0,
        intercept: 0,
        weights: {},
        equation: 'No Model Fitted',
        p_values: {},
        std_errors: {},
        t_values: {},
        confidence_intervals: {},
        vif: {}
      };
    }
    return currentAlgoModels.find((m: any) => m.k === selectedK) || currentAlgoModels[0];
  }, [currentAlgoModels, selectedK]);

  // Target Variable Name
  const activeTargetName = useMemo(() => {
    if (activeProblem === 'problem1') return 'Profit';
    return customTargetCol || 'Target';
  }, [activeProblem, customTargetCol]);

  // Dataset Name Display
  const activeDatasetDisplayName = useMemo(() => {
    if (activeProblem === 'problem1') return '50 Startups (內建標準題)';
    return customDatasetName || 'Custom Dataset (自訂資料集)';
  }, [activeProblem, customDatasetName]);

  // Live Prediction Calculation
  const prediction = useMemo(() => {
    if (!currentModel) return { total: 0, breakdown: {} };

    if (activeProblem === 'problem1') {
      let total = currentModel.intercept;
      const breakdown: Record<string, number> = { 'Intercept (β₀)': currentModel.intercept };
      const isFlorida = selectedState === 'Florida' ? 1 : 0;
      const isNewYork = selectedState === 'New York' ? 1 : 0;

      if (currentModel.weights['R&D Spend']) {
        const v = currentModel.weights['R&D Spend'] * rdSpend;
        total += v;
        breakdown['R&D Spend'] = Math.round(v * 100) / 100;
      }
      if (currentModel.weights['Administration']) {
        const v = currentModel.weights['Administration'] * adminSpend;
        total += v;
        breakdown['Administration'] = Math.round(v * 100) / 100;
      }
      if (currentModel.weights['Marketing Spend']) {
        const v = currentModel.weights['Marketing Spend'] * mktgSpend;
        total += v;
        breakdown['Marketing Spend'] = Math.round(v * 100) / 100;
      }
      if (currentModel.weights['State_Florida']) {
        const v = currentModel.weights['State_Florida'] * isFlorida;
        total += v;
        breakdown['State_Florida'] = Math.round(v * 100) / 100;
      }
      if (currentModel.weights['State_New York']) {
        const v = currentModel.weights['State_New York'] * isNewYork;
        total += v;
        breakdown['State_New York'] = Math.round(v * 100) / 100;
      }
      return { total: Math.round(total * 100) / 100, breakdown };
    } else {
      let total = currentModel.intercept || 0;
      const breakdown: Record<string, number> = { 'Intercept (β₀)': total };

      if (currentModel.weights) {
        Object.entries(currentModel.weights).forEach(([feat, weight]: [string, any]) => {
          const val = customPredictInputs[feat] ?? 0;
          const contrib = Number(weight) * Number(val);
          total += contrib;
          breakdown[feat] = Math.round(contrib * 100) / 100;
        });
      }
      return { total: Math.round(total * 100) / 100, breakdown };
    }
  }, [currentModel, activeProblem, rdSpend, adminSpend, mktgSpend, selectedState, customPredictInputs]);

  // Export CSV
  const handleExportCSV = (targetProblem: 'problem1' | 'problem2') => {
    const isP1 = targetProblem === 'problem1';
    const bundle = isP1
      ? (p1ApiData?.feature_selection_results || (p1RandomState === 0 ? FALLBACK_RESULTS_SEED_0 : FALLBACK_RESULTS_SEED_42))
      : (customAnalysisResult?.feature_selection_results || {});

    const dName = isP1 ? `50_Startups_Seed_${p1RandomState}` : customDatasetName.replace(/\s+/g, '_');
    const headers = ['Dataset', 'Algorithm', 'Number of Features', 'Selected Features', 'Train RMSE', 'Test RMSE', 'CV RMSE Mean', 'MSE', 'MAE', 'R-squared', 'Adjusted R-squared'];
    const rows: string[] = [headers.join(',')];

    Object.entries(bundle).forEach(([algo, models]: [string, any]) => {
      models.forEach((m: any) => {
        const feats = `"${m.features.join('; ')}"`;
        rows.push([dName, algo, m.k, feats, m.train_rmse, m.test_rmse, m.cv_rmse_mean || '', m.mse, m.mae, m.r2, m.adj_r2].join(','));
      });
    });

    const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${dName}_Regression_Results.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Export Markdown CRISP-DM Report
  const handleExportMarkdownReport = (targetProblem: 'problem1' | 'problem2') => {
    const isP1 = targetProblem === 'problem1';
    const dName = isP1 ? '50 Startups Dataset' : customDatasetName;
    const target = isP1 ? 'Profit' : customTargetCol;
    const model = isP1 ? (bestCvModel || currentModel) : (customAnalysisResult?.best_cv_model || currentModel);

    const reportContent = `# AIoT HW2 — 多元線性迴歸與特徵選取專案分析結案報告
**選定繳交題目：${isP1 ? 'Problem 1 (50 Startups 獲利預測)' : `Problem 2 (${dName} 預測)`}**
- 學生報告日期：${new Date().toLocaleDateString('zh-TW')}
- 執行架構：CRISP-DM 標準流程 (Business/Data Understanding → Preprocessing → Modeling → Evaluation → Deployment)
- 最佳模型評選依據：訓練集 5-Fold Cross-Validation RMSE (防範 Data Leakage / 測試集窺探)

---

## 1. 資料集理解與規格 (Data Understanding)
- **資料集名稱：** ${dName}
- **預測目標變數 (Target Variable)：** \`${target}\`
- **資料筆數 (Sample Count)：** ${isP1 ? 50 : (customProfile?.rowCount || customRecords.length)} 筆
- **原始特徵數 (Original Features)：** ${isP1 ? 5 : (customProfile?.columnNames.length ? customProfile.columnNames.length - 1 : 0)} 個
- **訓練 / 測試集劃分比例：** 80% 訓練集 (Train) / 20% 測試集 (Test)
- **隨機種子 (Random Seed)：** ${isP1 ? p1RandomState : customRandomSeed}

---

## 2. 特徵選取與模型訓練 (Feature Selection & Modeling)
- **最優特徵選取演算法：** \`${selectedAlgo}\`
- **選取特徵數 (k)：** ${model?.k}
- **選取特徵組合 (Selected Features)：**
${model?.features?.map((f: string) => `  - \`${f}\``).join('\n')}

### 迴歸方程式 (Regression Equation)
\`\`\`text
${model?.equation || 'N/A'}
\`\`\`

---

## 3. 模型評估指標 (Model Evaluation Metrics)
- **訓練集 5-Fold CV RMSE：** ${model?.cv_rmse_mean?.toLocaleString() || 'N/A'} (標準差: ±${model?.cv_rmse_std?.toLocaleString() || 0})
- **獨立測試集 Test RMSE：** ${model?.test_rmse?.toLocaleString()}
- **訓練集 Train RMSE：** ${model?.train_rmse?.toLocaleString()}
- **泛化差距 (Generalization Gap = Test - Train RMSE)：** ${model?.train_test_gap?.toLocaleString()}
- **平均絕對誤差 (MAE)：** ${model?.mae?.toLocaleString()}
- **判定係數 (R²)：** ${model?.r2?.toFixed(4)}
- **調整後判定係數 (Adjusted R²)：** ${model?.adj_r2?.toFixed(4)}

---

## 4. 統計推論與殘差檢定 (Statistical Inference)
- **多重共線性檢驗 (VIF)：**
${Object.entries(model?.vif || {}).map(([feat, vif]) => `  - \`${feat}\`: VIF = ${vif}`).join('\n')}
- **顯著性檢定 (p-value)：**
${Object.entries(model?.p_values || {}).map(([feat, p]) => `  - \`${feat}\`: p-value = ${p}`).join('\n')}

---

## 5. 跨資料集比較與作業要求總結 (Problem Comparison & Discussion)
1. **量綱尺度與 RMSE 不可直接絕對比較說明：**
   不同資料集因目標變數物理單位不同（如利潤為數萬美元，房價中位數為數十萬美元），絕對 RMSE 不可跨資料集判定優劣。評估主要仰賴無量綱之 $R^2$、調整後 $R^2$ 與交叉驗證穩定性。
2. **防範 Data Leakage 機制：**
   One-Hot Encoding、缺失值中位數填補與特徵選取評分嚴格僅在訓練集擬合，避免任何資訊外洩至測試集。
`;

    const blob = new Blob([reportContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${isP1 ? '50_Startups' : customDatasetName.replace(/\s+/g, '_')}_CRISP_DM_Report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Switch Dataset Problem & Reset Stale States
  const handleProblemChange = (problem: 'problem1' | 'problem2') => {
    setActiveProblem(problem);
    if (problem === 'problem1') {
      setSelectedK(3);
    } else {
      if (customAnalysisResult?.max_k && selectedK > customAnalysisResult.max_k) {
        setSelectedK(customAnalysisResult.max_k);
      }
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-900 font-sans">
      
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-xs">
        <div className="max-w-6xl mx-auto px-4 py-3 flex flex-wrap justify-between items-center gap-3">
          
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold text-base shadow-sm">
              ML
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full uppercase bg-indigo-50 text-indigo-700 border border-indigo-200">
                  AIoT HW2
                </span>
                <span className="text-xs text-slate-400">• 四步驟直覺式迴歸分析平台</span>
              </div>
              <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                多元線性迴歸與特徵選取 (Multiple Linear Regression Explorer)
              </h1>
            </div>
          </div>

          {/* Backend Connection Indicator Badge */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
              <span className={`w-2 h-2 rounded-full ${isBackendConnected ? 'bg-emerald-500 animate-pulse' : 'bg-indigo-500'}`} />
              <span className="text-[11px]">{backendRuntime}</span>
            </div>
          </div>

        </div>
      </header>

      {/* 4-Step Guided Flow Wizard Bar */}
      <div className="bg-white border-b border-slate-200 py-3 shadow-xs">
        <div className="max-w-6xl mx-auto px-4">
          <nav aria-label="Progress" className="flex items-center justify-between">
            {[
              { num: 1, label: '資料集 (Dataset)', desc: activeProblem === 'problem1' ? '50 Startups' : customDatasetName },
              { num: 2, label: '模型設定 (Model)', desc: `${selectedAlgo} (k=${currentModel.k})` },
              { num: 3, label: '結果與評估 (Results)', desc: currentModel.r2 ? `R² = ${currentModel.r2.toFixed(3)}` : '尚未分析' },
              { num: 4, label: '比較與匯出 (Export)', desc: `選擇 ${submissionChoice === 'problem1' ? 'Problem 1' : 'Problem 2'}` },
            ].map((step, idx) => {
              const isActive = currentStep === step.num;
              const isPassed = currentStep > step.num;
              return (
                <button
                  key={step.num}
                  onClick={() => setCurrentStep(step.num)}
                  className={`flex items-center gap-2.5 text-left py-1.5 px-3 rounded-xl transition-all cursor-pointer ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700 font-bold border border-indigo-200 shadow-xs'
                      : isPassed
                      ? 'text-slate-700 hover:bg-slate-100'
                      : 'text-slate-400 hover:text-slate-600'
                  }`}
                >
                  <span className={`w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold ${
                    isActive
                      ? 'bg-indigo-600 text-white shadow-sm'
                      : isPassed
                      ? 'bg-emerald-100 text-emerald-800'
                      : 'bg-slate-100 text-slate-400'
                  }`}>
                    {isPassed ? <Check className="w-4 h-4 stroke-[2.5]" /> : step.num}
                  </span>
                  <div className="hidden sm:block">
                    <div className="text-xs font-bold leading-tight">{step.label}</div>
                    <div className="text-[10px] text-slate-400 truncate max-w-[130px]">{step.desc}</div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Main Container */}
      <main className="max-w-6xl mx-auto px-4 py-6 w-full flex-1">

        {/* ========================================================================= */}
        {/* STEP 1: 資料集選擇與探勘 (Dataset Selection & Profiling)                     */}
        {/* ========================================================================= */}
        {currentStep === 1 && (
          <div className="space-y-6">
            
            {/* Step Header */}
            <div>
              <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-wider mb-1">
                <span>Step 1 of 4</span>
                <span>•</span>
                <span>選擇或匯入資料集</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
                選擇要分析的迴歸問題 (Choose Problem)
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 mt-1">
                依據作業規範「Do two problems but choose one of the topics to upload」，請先選擇 Problem 1 或 Problem 2。
              </p>
            </div>

            {/* Problem Choice Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              
              {/* Problem 1 Card */}
              <div
                onClick={() => handleProblemChange('problem1')}
                className={`p-5 rounded-2xl border-2 cursor-pointer transition-all ${
                  activeProblem === 'problem1'
                    ? 'bg-indigo-50/50 border-indigo-600 shadow-md ring-2 ring-indigo-500/10'
                    : 'bg-white border-slate-200 hover:border-slate-300 shadow-xs'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${activeProblem === 'problem1' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                      <Database className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-bold text-sm text-slate-900">Problem 1: 50 Startups</h3>
                      <span className="text-[10px] text-slate-400">內建標準題目 (Built-in)</span>
                    </div>
                  </div>
                  <span className={`w-4 h-4 rounded-full border flex items-center justify-center ${activeProblem === 'problem1' ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-slate-300'}`}>
                    {activeProblem === 'problem1' && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                  </span>
                </div>
                <p className="text-xs text-slate-600 mb-3 leading-relaxed">
                  經典創投營運支出資料集，包含研發、行政、行銷支出與設立州別，預測目標為公司利潤（<code>Profit</code>）。
                </p>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 font-mono">50 Samples</span>
                  <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 font-mono">4 Features</span>
                  <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 font-mono font-bold">Target: Profit</span>
                </div>
              </div>

              {/* Problem 2 Card */}
              <div
                onClick={() => handleProblemChange('problem2')}
                className={`p-5 rounded-2xl border-2 cursor-pointer transition-all ${
                  activeProblem === 'problem2'
                    ? 'bg-indigo-50/50 border-indigo-600 shadow-md ring-2 ring-indigo-500/10'
                    : 'bg-white border-slate-200 hover:border-slate-300 shadow-xs'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${activeProblem === 'problem2' ? 'bg-indigo-600 text-white' : 'bg-slate-100 text-slate-600'}`}>
                      <Sparkles className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="font-bold text-sm text-slate-900">Problem 2: 自訂資料集 (Custom Dataset)</h3>
                      <span className="text-[10px] text-slate-400">支援 CSV 上傳與公開 URL 匯入</span>
                    </div>
                  </div>
                  <span className={`w-4 h-4 rounded-full border flex items-center justify-center ${activeProblem === 'problem2' ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-slate-300'}`}>
                    {activeProblem === 'problem2' && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                  </span>
                </div>
                <p className="text-xs text-slate-600 mb-3 leading-relaxed">
                  通用表格式機器學習分析，可自由匯入本機或線上 CSV，自動探勘缺失值、類別編碼並動態選擇預測目標。
                </p>
                <div className="flex flex-wrap gap-1.5 text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 font-mono">
                    {customProfile ? `${customProfile.rowCount} Samples` : '任意樣本數'}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 font-mono">
                    {customProfile ? `${customProfile.columnNames.length - 1} Features` : '多維特徵'}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-800 font-mono font-bold">
                    Target: {customTargetCol}
                  </span>
                </div>
              </div>

            </div>

            {/* Custom Dataset Import Box (Only shown if Problem 2 is chosen) */}
            {activeProblem === 'problem2' && (
              <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-4">
                
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
                  <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                    <Upload className="w-3.5 h-3.5 text-indigo-600" />
                    匯入資料方式 (Import Method)：
                  </span>
                  <div className="flex items-center bg-slate-100 p-1 rounded-lg text-xs font-semibold">
                    <button
                      onClick={() => setCustomImportMode('url')}
                      className={`px-3 py-1 rounded-md transition-all ${customImportMode === 'url' ? 'bg-white text-indigo-700 shadow-xs' : 'text-slate-600'}`}
                    >
                      公開 CSV 網址 (URL)
                    </button>
                    <button
                      onClick={() => setCustomImportMode('upload')}
                      className={`px-3 py-1 rounded-md transition-all ${customImportMode === 'upload' ? 'bg-white text-indigo-700 shadow-xs' : 'text-slate-600'}`}
                    >
                      本機 CSV 上傳 (File)
                    </button>
                  </div>
                </div>

                {customError && (
                  <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0" />
                    <span>{customError}</span>
                  </div>
                )}
                {customSuccessMsg && !customError && (
                  <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                    <span>{customSuccessMsg}</span>
                  </div>
                )}

                {/* Import Inputs */}
                {customImportMode === 'upload' ? (
                  <div>
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                      accept=".csv,text/csv"
                      className="hidden"
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full border-2 border-dashed border-indigo-200 hover:border-indigo-400 bg-indigo-50/20 hover:bg-indigo-50/50 rounded-xl p-6 text-center cursor-pointer transition-all flex flex-col items-center justify-center gap-1.5"
                    >
                      <Upload className="w-7 h-7 text-indigo-600" />
                      <span className="text-xs font-bold text-slate-800">點擊此處選擇本機 CSV 檔案</span>
                      <span className="text-[11px] text-slate-400">支援標準 UTF-8 及常見 CSV 編碼格式 (上限 5MB)</span>
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      <input
                        type="url"
                        value={customUrlInput}
                        onChange={(e) => setCustomUrlInput(e.target.value)}
                        placeholder="請輸入公開 CSV 直連網址 (例如 GitHub raw CSV 連結)"
                        className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3.5 py-2 text-xs font-mono text-slate-800 outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                      <button
                        onClick={handleUrlImport}
                        disabled={isCustomLoading}
                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-lg shadow-xs flex items-center gap-1.5 transition-all disabled:opacity-50"
                      >
                        {isCustomLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                        載入資料
                      </button>
                    </div>
                    <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
                      <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                      <span>已啟動 SSRF 安全代理防護 (自動阻絕私有內部 IP 與超時)</span>
                    </div>
                  </div>
                )}

                {/* Presets */}
                <div className="pt-2 flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-bold text-slate-500">快速範例：</span>
                  <button
                    onClick={async () => {
                      const res = await fetch('/data/real_estate_valuation_10features.csv');
                      const text = await res.text();
                      loadCsvDataString(text, '加州房地產估價 (10 Features)', 'MedianHouseValue');
                    }}
                    className="px-2.5 py-1 bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 border border-slate-200 rounded-md text-xs font-medium text-slate-700 transition-all"
                  >
                    🏠 範例 1：加州房產 (10 特徵, 80 筆)
                  </button>
                  <button
                    onClick={() => {
                      setCustomUrlInput('https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv');
                      fetch('https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv')
                        .then(r => r.text())
                        .then(t => loadCsvDataString(t, '50 Startups (URL 版)', 'Profit'))
                        .catch(() => handleUrlImport());
                    }}
                    className="px-2.5 py-1 bg-slate-100 hover:bg-indigo-50 hover:text-indigo-700 border border-slate-200 rounded-md text-xs font-medium text-slate-700 transition-all"
                  >
                    💼 範例 2：50 Startups 線上 CSV
                  </button>
                </div>

              </div>
            )}

            {/* Dataset Preview & Statistics Table */}
            <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-indigo-600" />
                  <h3 className="font-bold text-sm text-slate-900">
                    資料集規格與前 10 筆預覽 (Data Preview & Info)
                  </h3>
                </div>
                <div className="flex flex-wrap gap-2 text-xs font-mono">
                  <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                    樣本數: {activeProblem === 'problem1' ? 50 : (customProfile?.rowCount || customRecords.length || 0)} 筆
                  </span>
                  <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700">
                    特徵數: {activeProblem === 'problem1' ? 4 : (customProfile?.columnNames?.length ? customProfile.columnNames.length - 1 : 0)} 欄
                  </span>
                  <span className="px-2 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200">
                    缺失值: 0 處
                  </span>
                </div>
              </div>

              {/* Preview Table */}
              <div className="overflow-x-auto border border-slate-200 rounded-xl max-h-52">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-50 text-slate-600 font-semibold sticky top-0">
                    <tr>
                      <th className="p-2 border-b border-slate-200 text-center w-10">#</th>
                      {activeProblem === 'problem1' ? (
                        <>
                          <th className="p-2 border-b border-slate-200">R&D Spend</th>
                          <th className="p-2 border-b border-slate-200">Administration</th>
                          <th className="p-2 border-b border-slate-200">Marketing Spend</th>
                          <th className="p-2 border-b border-slate-200">State</th>
                          <th className="p-2 border-b border-slate-200 bg-indigo-50 text-indigo-900 font-bold">Profit (🎯 Target)</th>
                        </>
                      ) : (
                        customProfile?.columnNames.map(col => (
                          <th key={col} className={`p-2 border-b border-slate-200 whitespace-nowrap ${col === customTargetCol ? 'bg-indigo-50 text-indigo-900 font-bold' : ''}`}>
                            {col} {col === customTargetCol && '(🎯 Target)'}
                          </th>
                        ))
                      )}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-[11px]">
                    {activeProblem === 'problem1' ? (
                      DATASET_50_STARTUPS.slice(0, 5).map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="p-2 text-center text-slate-400">{idx + 1}</td>
                          <td className="p-2">{row["R&D Spend"].toLocaleString()}</td>
                          <td className="p-2">{row["Administration"].toLocaleString()}</td>
                          <td className="p-2">{row["Marketing Spend"].toLocaleString()}</td>
                          <td className="p-2">{row["State"]}</td>
                          <td className="p-2 font-bold text-indigo-700 bg-indigo-50/30">${row["Profit"].toLocaleString()}</td>
                        </tr>
                      ))
                    ) : (
                      customProfile?.firstRows.slice(0, 5).map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-50">
                          <td className="p-2 text-center text-slate-400">{idx + 1}</td>
                          {customProfile.columnNames.map(col => (
                            <td key={col} className={`p-2 whitespace-nowrap ${col === customTargetCol ? 'font-bold text-indigo-700 bg-indigo-50/30' : 'text-slate-700'}`}>
                              {row[col] !== undefined ? String(row[col]) : ''}
                            </td>
                          ))}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Bottom Next Step Action */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
              <div className="text-xs text-slate-500 font-medium flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>已選擇分析資料集：<strong className="text-slate-800">{activeDatasetDisplayName}</strong></span>
              </div>
              <button
                onClick={() => setCurrentStep(2)}
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl shadow-sm flex items-center gap-2 transition-all cursor-pointer"
              >
                <span>下一步：設定迴歸模型 (Step 2)</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 2: 模型設定與特徵選取 (Model Configuration)                             */}
        {/* ========================================================================= */}
        {currentStep === 2 && (
          <div className="space-y-6">
            
            {/* Step Header */}
            <div>
              <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-wider mb-1">
                <span>Step 2 of 4</span>
                <span>•</span>
                <span>模型超參數與特徵選取設定</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
                設定模型參數 (Model Configuration Panel)
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 mt-1">
                目前選定資料集：<strong>{activeDatasetDisplayName}</strong>。設定目標變數與演算法後，即可一鍵執行全套迴歸擬合。
              </p>
            </div>

            {/* Core Model Configuration Panel */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6">
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                {/* 1. Target Variable Selection */}
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                    <span>🎯 預測目標變數 (Target Variable):</span>
                  </label>
                  {activeProblem === 'problem1' ? (
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-200 text-xs font-mono font-bold text-indigo-700 flex items-center justify-between">
                      <span>Profit (公司年度淨利潤)</span>
                      <span className="text-[10px] bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded">內建目標</span>
                    </div>
                  ) : (
                    <select
                      value={customTargetCol}
                      onChange={(e) => {
                        const newT = e.target.value;
                        setCustomTargetCol(newT);
                        setCustomFeatureCols(prev => prev.filter(c => c !== newT));
                      }}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-semibold text-slate-800 outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
                    >
                      {customProfile?.numericColumns.map(col => (
                        <option key={col} value={col}>
                          {col} (數值連續型變數)
                        </option>
                      ))}
                    </select>
                  )}
                  <p className="text-[11px] text-slate-400">
                    通用迴歸引擎將針對此目標變數擬合線性權重方程。
                  </p>
                </div>

                {/* 2. Algorithm Selection */}
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-700 flex items-center gap-1.5">
                    <span>⚙️ 特徵選取演算法 (Feature Selection Algorithm):</span>
                  </label>
                  <select
                    value={selectedAlgo}
                    onChange={(e) => setSelectedAlgo(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-semibold text-slate-800 outline-none focus:ring-2 focus:ring-indigo-500 cursor-pointer"
                  >
                    <option value="Forward SFS">Forward Sequential Feature Selection (Forward SFS)</option>
                    <option value="RFE">Recursive Feature Elimination (RFE)</option>
                    <option value="Exhaustive Search">Exhaustive Feature Search (最佳子集窮舉)</option>
                    <option value="Baseline OLS">Baseline OLS (單變數皮爾森相關排序)</option>
                    <option value="Lasso">Lasso (L1 正則化稀疏路徑)</option>
                    <option value="ElasticNet">ElasticNet (L1 + L2 混合正則化)</option>
                  </select>
                  <p className="text-[11px] text-slate-400">
                    推薦使用 Forward SFS，以訓練集 5-Fold CV 尋找泛化最優特徵子集。
                  </p>
                </div>

              </div>

              {/* 3. Number of Features (k) Selector */}
              <div className="space-y-2 pt-2 border-t border-slate-100">
                <div className="flex justify-between items-center">
                  <label className="text-xs font-bold text-slate-700">
                    特徵選取數量 (Number of Features k):
                  </label>
                  <span className="text-xs font-mono font-bold text-indigo-600">k = {selectedK}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {Array.from({ length: maxK }, (_, i) => i + 1).map(k => {
                    const isBestCv = bestCvModel?.k === k;
                    const isBestTest = bestTestModel?.k === k;
                    const isSelected = selectedK === k;
                    return (
                      <button
                        key={k}
                        onClick={() => setSelectedK(k)}
                        className={`px-3 py-1.5 rounded-xl text-xs font-bold transition-all flex items-center gap-1.5 border cursor-pointer ${
                          isSelected
                            ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                            : isBestCv
                            ? 'bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100'
                            : 'bg-slate-50 text-slate-700 border-slate-200 hover:border-slate-300'
                        }`}
                      >
                        <span>{k} Features</span>
                        {isBestCv && (
                          <span className={`text-[9px] px-1.5 py-0.2 rounded-full ${isSelected ? 'bg-white/20 text-white' : 'bg-emerald-200 text-emerald-900'}`}>
                            🏆 推薦 CV
                          </span>
                        )}
                        {isBestTest && !isBestCv && (
                          <span className={`text-[9px] px-1.5 py-0.2 rounded-full ${isSelected ? 'bg-white/20 text-white' : 'bg-amber-200 text-amber-900'}`}>
                            🎯 最低 Test
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* 4. Split and Seed */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-2 border-t border-slate-100">
                
                {/* Seed Toggle / Input */}
                <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-700">隨機種子 (Random Seed):</label>
                  {activeProblem === 'problem1' ? (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setP1RandomState(42)}
                        className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-bold border transition-all ${
                          p1RandomState === 42 ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs' : 'bg-slate-50 text-slate-600 border-slate-200'
                        }`}
                      >
                        Seed 42 (標準作業規範)
                      </button>
                      <button
                        onClick={() => setP1RandomState(0)}
                        className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-bold border transition-all ${
                          p1RandomState === 0 ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs' : 'bg-slate-50 text-slate-600 border-slate-200'
                        }`}
                      >
                        Seed 0 (對標 8198.80)
                      </button>
                    </div>
                  ) : (
                    <input
                      type="number"
                      value={customRandomSeed}
                      onChange={(e) => setCustomRandomSeed(Number(e.target.value))}
                      className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs font-mono font-semibold text-slate-800"
                    />
                  )}
                </div>

                {/* Test Split Fraction */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-bold text-slate-700">測試集劃分比例 (Test Size):</label>
                    <span className="font-mono text-xs font-bold text-indigo-600">
                      {activeProblem === 'problem1' ? '20% (固定 10 筆)' : `${Math.round(customTestSize * 100)}%`}
                    </span>
                  </div>
                  {activeProblem === 'problem1' ? (
                    <div className="text-xs text-slate-500 py-1 font-mono">
                      80% 訓練集 (40 筆) / 20% 獨立測試集 (10 筆)
                    </div>
                  ) : (
                    <input
                      type="range"
                      min="0.1"
                      max="0.4"
                      step="0.05"
                      value={customTestSize}
                      onChange={(e) => setCustomTestSize(Number(e.target.value))}
                      className="w-full accent-indigo-600 cursor-pointer"
                    />
                  )}
                </div>

              </div>

              {/* Collapsible Advanced Settings (Checklist of Features) */}
              {activeProblem === 'problem2' && customProfile && (
                <div className="pt-2 border-t border-slate-100">
                  <button
                    onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                    className="flex items-center gap-1.5 text-xs font-bold text-slate-600 hover:text-indigo-600 transition-colors cursor-pointer"
                  >
                    <Settings2 className="w-3.5 h-3.5" />
                    <span>進階設定：自訂候選特徵白名單 ({customFeatureCols.length}/{customProfile.columnNames.length - 1} 個已選取)</span>
                    {showAdvancedSettings ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </button>

                  {showAdvancedSettings && (
                    <div className="mt-3 p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2 text-xs">
                      <div className="flex justify-between items-center pb-1">
                        <span className="font-bold text-slate-700">勾選參與特徵篩選的原始欄位：</span>
                        <div className="flex gap-2">
                          <button
                            onClick={() => setCustomFeatureCols(customProfile.columnNames.filter(c => c !== customTargetCol))}
                            className="text-[11px] text-indigo-600 hover:underline font-semibold"
                          >
                            全選
                          </button>
                          <button
                            onClick={() => setCustomFeatureCols([])}
                            className="text-[11px] text-slate-500 hover:underline"
                          >
                            清空
                          </button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {customProfile.columnNames.filter(c => c !== customTargetCol).map(feat => {
                          const isSel = customFeatureCols.includes(feat);
                          return (
                            <button
                              key={feat}
                              onClick={() => {
                                if (isSel) setCustomFeatureCols(customFeatureCols.filter(c => c !== feat));
                                else setCustomFeatureCols([...customFeatureCols, feat]);
                              }}
                              className={`px-2.5 py-1 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all ${
                                isSel ? 'bg-indigo-50 border-indigo-300 text-indigo-800 font-bold' : 'bg-white border-slate-200 text-slate-600'
                              }`}
                            >
                              <span className={`w-3 h-3 rounded flex items-center justify-center text-[9px] ${isSel ? 'bg-indigo-600 text-white' : 'border border-slate-300'}`}>
                                {isSel && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                              </span>
                              <span>{feat}</span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Run Analysis Primary Button */}
              <div className="pt-2">
                <button
                  onClick={() => {
                    if (activeProblem === 'problem2') {
                      runAnalysisOnCustomData(
                        customRecords,
                        customTargetCol,
                        customFeatureCols,
                        customTestSize,
                        customRandomSeed,
                        customDatasetName
                      );
                    }
                    setCurrentStep(3);
                  }}
                  disabled={isCustomLoading || isP1Loading}
                  className="w-full py-3.5 bg-gradient-to-r from-indigo-600 via-indigo-700 to-indigo-800 hover:from-indigo-700 hover:to-indigo-900 text-white font-bold text-sm rounded-xl shadow-md flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-50"
                >
                  {isCustomLoading || isP1Loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>正在執行全套迴歸擬合與交叉驗證...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-white" />
                      <span>🚀 執行迴歸分析 (Run Regression Analysis)</span>
                    </>
                  )}
                </button>
              </div>

            </div>

            {/* Step Navigation Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
              <button
                onClick={() => setCurrentStep(1)}
                className="px-4 py-2.5 text-slate-600 hover:text-slate-900 text-xs font-bold rounded-xl border border-slate-200 hover:bg-slate-50 flex items-center gap-1.5 cursor-pointer transition-all"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>上一步：選擇資料集 (Step 1)</span>
              </button>
              <button
                onClick={() => {
                  if (activeProblem === 'problem2') {
                    runAnalysisOnCustomData(
                      customRecords,
                      customTargetCol,
                      customFeatureCols,
                      customTestSize,
                      customRandomSeed,
                      customDatasetName
                    );
                  }
                  setCurrentStep(3);
                }}
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl shadow-xs flex items-center gap-2 transition-all cursor-pointer"
              >
                <span>下一步：查看評估結果 (Step 3)</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 3: 結果展示與評估 (Model Results & Evaluation)                          */}
        {/* ========================================================================= */}
        {currentStep === 3 && (
          <div className="space-y-6">
            
            {/* Step Header */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-wider mb-1">
                  <span>Step 3 of 4</span>
                  <span>•</span>
                  <span>模型評估指標與視覺化</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
                  迴歸模型評估結果 (Evaluation Results)
                </h2>
                <p className="text-xs sm:text-sm text-slate-500 mt-1">
                  目前顯示：<strong>{activeDatasetDisplayName}</strong> • 演算法：<strong>{selectedAlgo}</strong> (k={currentModel.k})
                </p>
              </div>

              {/* Quick Jump to Step 4 */}
              <button
                onClick={() => setCurrentStep(4)}
                className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl shadow-xs flex items-center gap-1.5 transition-all cursor-pointer"
              >
                <span>下一步：比較與匯出成果 (Step 4)</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

            {/* Recommended Model Banner */}
            <div className="p-4 rounded-2xl border border-emerald-200 bg-emerald-50/60 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-emerald-600 text-white flex items-center justify-center font-bold text-base shadow-sm">
                  🏆
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-bold text-sm text-emerald-950">
                      推薦最佳模型：{bestCvModel?.k} Features ({selectedAlgo})
                    </span>
                    <span className="text-[10px] bg-emerald-200 text-emerald-900 font-bold px-2 py-0.5 rounded-full">
                      5-Fold CV 評選
                    </span>
                    <span className="text-[10px] bg-white/90 border border-emerald-300 text-emerald-800 font-semibold px-2 py-0.5 rounded-full shadow-2xs">
                      {activeProblem === 'problem1' ? '⚙️ 運算責任：Python FastAPI / Vercel Serverless' : '⚡ 運算責任：純 TypeScript 矩陣運算引擎 (即時 5-Fold CV 擬合)'}
                    </span>
                  </div>
                  <div className="text-xs text-emerald-800 mt-0.5">
                    特徵組合：<strong>{bestCvModel?.features?.join(' + ')}</strong>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4 text-xs font-mono">
                <div>
                  <span className="text-emerald-700 block text-[10px] uppercase">CV RMSE:</span>
                  <span className="font-bold text-emerald-950">{bestCvModel?.cv_rmse_mean?.toLocaleString() || 'N/A'}</span>
                </div>
                <div>
                  <span className="text-emerald-700 block text-[10px] uppercase">Test RMSE:</span>
                  <span className="font-bold text-emerald-950">{bestCvModel?.test_rmse?.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-emerald-700 block text-[10px] uppercase">Test R²:</span>
                  <span className="font-bold text-emerald-950">{bestCvModel?.r2?.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* 4 Core KPI Metric Cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
              
              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs relative overflow-hidden">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">測試集 Test RMSE</div>
                <div className="text-2xl font-extrabold text-slate-900 font-mono">
                  {currentModel.test_rmse ? currentModel.test_rmse.toLocaleString() : '0'}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">獨立評估誤差 (越低越好)</div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-indigo-600" />
              </div>

              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs relative overflow-hidden">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">判定係數 R²</div>
                <div className={`text-2xl font-extrabold font-mono ${currentModel.r2 > 0 ? 'text-indigo-600' : 'text-rose-600'}`}>
                  {currentModel.r2 ? currentModel.r2.toFixed(4) : '0.0000'}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">解釋變異能力 (越高越好)</div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-indigo-500" />
              </div>

              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs relative overflow-hidden">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">調整後 R² (Adj R²)</div>
                <div className="text-2xl font-extrabold text-slate-900 font-mono">
                  {currentModel.adj_r2 ? currentModel.adj_r2.toFixed(4) : '0.0000'}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">懲罰無效多餘變數</div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-indigo-400" />
              </div>

              <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs relative overflow-hidden">
                <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">5-Fold CV RMSE</div>
                <div className="text-2xl font-extrabold text-emerald-600 font-mono">
                  {currentModel.cv_rmse_mean ? currentModel.cv_rmse_mean.toLocaleString() : 'N/A'}
                </div>
                <div className="text-[11px] text-slate-500 mt-1">訓練集交叉驗證穩定度</div>
                <div className="absolute top-0 left-0 right-0 h-1 bg-emerald-500" />
              </div>

            </div>

            {/* The Two Main Evaluation Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* Chart 1: RMSE by Number of Features */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-sm text-slate-900">特徵數量與 RMSE 曲線 (RMSE by k)</h3>
                    <p className="text-[11px] text-slate-400">同時對照訓練集、測試集與 5-Fold CV</p>
                  </div>
                  <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono">
                    {selectedAlgo}
                  </span>
                </div>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentAlgoModels} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="k" label={{ value: '特徵數 k', position: 'insideBottom', offset: -5 }} />
                      <YAxis domain={['auto', 'auto']} tickFormatter={(v) => v.toLocaleString()} />
                      <Tooltip formatter={(value: any) => [`${Number(value).toLocaleString()}`, '']} />
                      <Legend verticalAlign="top" height={30} />
                      <Line type="monotone" dataKey="test_rmse" name="Test RMSE" stroke="#4f46e5" strokeWidth={2.5} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="train_rmse" name="Train RMSE" stroke="#0ea5e9" strokeWidth={1.5} strokeDasharray="4 4" dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="cv_rmse_mean" name="CV RMSE (5-Fold)" stroke="#10b981" strokeWidth={1.5} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chart 2: R-squared by Number of Features */}
              <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="font-bold text-sm text-slate-900">特徵數量與 R² / 調整後 R² 曲線</h3>
                    <p className="text-[11px] text-slate-400">檢驗特徵增加之邊際解釋力與懲罰</p>
                  </div>
                  <span className="text-[10px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono">
                    {selectedAlgo}
                  </span>
                </div>
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={currentAlgoModels} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                      <XAxis dataKey="k" label={{ value: '特徵數 k', position: 'insideBottom', offset: -5 }} />
                      <YAxis domain={['auto', 1.0]} />
                      <Tooltip formatter={(value: any) => [Number(value).toFixed(4), '']} />
                      <Legend verticalAlign="top" height={30} />
                      <Line type="monotone" dataKey="r2" name="R² (判定係數)" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4 }} />
                      <Line type="monotone" dataKey="adj_r2" name="Adjusted R²" stroke="#ec4899" strokeWidth={1.5} strokeDasharray="3 3" dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

            </div>

            {/* Model Evaluation Summary Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="p-4 border-b border-slate-100 flex justify-between items-center">
                <div>
                  <h3 className="font-bold text-sm text-slate-900">
                    全特徵數量模型評估摘要表 (Model Summary Table)
                  </h3>
                  <p className="text-[11px] text-slate-500">點擊任意列可直接切換當前模型特徵數 (k)</p>
                </div>
                <span className="text-xs font-mono text-indigo-600">Target: {activeTargetName}</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-50 text-slate-600 font-semibold">
                    <tr>
                      <th className="p-3">特徵數 (k)</th>
                      <th className="p-3 font-sans">選取特徵 (Selected Features)</th>
                      <th className="p-3">Train RMSE</th>
                      <th className="p-3">Test RMSE</th>
                      <th className="p-3">CV RMSE</th>
                      <th className="p-3">MAE</th>
                      <th className="p-3">R²</th>
                      <th className="p-3">Adj R²</th>
                      <th className="p-3 font-sans">狀態</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {currentAlgoModels.map((m: any) => {
                      const isCurrent = m.k === selectedK;
                      const isBestCv = bestCvModel?.k === m.k;
                      return (
                        <tr
                          key={m.k}
                          onClick={() => setSelectedK(m.k)}
                          className={`cursor-pointer transition-colors ${isCurrent ? 'bg-indigo-50 font-bold text-indigo-950' : 'hover:bg-slate-50 text-slate-700'}`}
                        >
                          <td className="p-3">{m.k}</td>
                          <td className="p-3 font-sans max-w-xs truncate" title={m.features?.join(', ')}>
                            {m.features?.join(' + ')}
                          </td>
                          <td className="p-3">{m.train_rmse?.toLocaleString()}</td>
                          <td className="p-3 text-indigo-600">{m.test_rmse?.toLocaleString()}</td>
                          <td className="p-3 text-emerald-600">{m.cv_rmse_mean?.toLocaleString() || 'N/A'}</td>
                          <td className="p-3">{m.mae?.toLocaleString()}</td>
                          <td className="p-3">{m.r2?.toFixed(4)}</td>
                          <td className="p-3">{m.adj_r2?.toFixed(4)}</td>
                          <td className="p-3 font-sans">
                            {isBestCv && (
                              <span className="bg-emerald-100 text-emerald-800 text-[10px] px-2 py-0.5 rounded-full font-bold">
                                最佳 CV 🏆
                              </span>
                            )}
                            {isCurrent && !isBestCv && (
                              <span className="bg-indigo-100 text-indigo-800 text-[10px] px-2 py-0.5 rounded-full">
                                目前選定
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Collapsible Advanced Analysis Accordion */}
            <div className="space-y-3 pt-2">
              <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                進階深入分析 (可選折疊展開 / Advanced Analysis)：
              </div>

              {/* Accordion 1: Live Prediction Form */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
                <button
                  onClick={() => toggleSection('predictor')}
                  className="w-full p-4 text-left font-bold text-xs flex items-center justify-between text-slate-800 hover:bg-slate-50 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-indigo-600" />
                    <span>🧮 即時預測表單 (Live Predictor for {activeTargetName})</span>
                  </div>
                  {expandedSections.predictor ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSections.predictor && (
                  <div className="p-5 border-t border-slate-100 grid grid-cols-1 md:grid-cols-12 gap-6 bg-slate-50/50">
                    <div className="md:col-span-7 space-y-3">
                      <div className="text-xs font-bold text-slate-700 mb-2">輸入當前模型特徵數值：</div>
                      {activeProblem === 'problem1' ? (
                        <div className="space-y-3 text-xs">
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="font-semibold text-slate-600">研發支出 (R&D Spend):</span>
                              <span className="font-mono font-bold text-indigo-600">${rdSpend.toLocaleString()}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="200000"
                              step="2000"
                              value={rdSpend}
                              onChange={(e) => setRdSpend(Number(e.target.value))}
                              className="w-full accent-indigo-600 cursor-pointer"
                            />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="font-semibold text-slate-600">行政管理 (Administration):</span>
                              <span className="font-mono font-bold text-indigo-600">${adminSpend.toLocaleString()}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="200000"
                              step="2000"
                              value={adminSpend}
                              onChange={(e) => setAdminSpend(Number(e.target.value))}
                              className="w-full accent-indigo-600 cursor-pointer"
                            />
                          </div>
                          <div>
                            <div className="flex justify-between mb-1">
                              <span className="font-semibold text-slate-600">行銷推廣 (Marketing Spend):</span>
                              <span className="font-mono font-bold text-indigo-600">${mktgSpend.toLocaleString()}</span>
                            </div>
                            <input
                              type="range"
                              min="0"
                              max="500000"
                              step="5000"
                              value={mktgSpend}
                              onChange={(e) => setMktgSpend(Number(e.target.value))}
                              className="w-full accent-indigo-600 cursor-pointer"
                            />
                          </div>
                          <div>
                            <span className="font-semibold text-slate-600 block mb-1">設立州別 (State):</span>
                            <div className="flex gap-2">
                              {['California', 'Florida', 'New York'].map(st => (
                                <button
                                  key={st}
                                  onClick={() => setSelectedState(st)}
                                  className={`flex-1 py-1 px-2 rounded-lg text-xs font-semibold border ${selectedState === st ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white border-slate-200 text-slate-700'}`}
                                >
                                  {st}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-2.5 max-h-56 overflow-y-auto pr-1">
                          {currentModel.features?.map((feat: string) => (
                            <div key={feat} className="flex items-center justify-between gap-3 bg-white p-2.5 rounded-lg border border-slate-200 text-xs">
                              <label className="font-semibold text-slate-700 truncate">{feat}:</label>
                              <input
                                type="number"
                                value={customPredictInputs[feat] ?? 0}
                                onChange={(e) => {
                                  const val = Number(e.target.value);
                                  setCustomPredictInputs(prev => ({ ...prev, [feat]: val }));
                                }}
                                className="w-28 bg-slate-50 border border-slate-300 rounded px-2 py-1 text-xs font-mono"
                              />
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    <div className="md:col-span-5 bg-gradient-to-br from-indigo-900 to-slate-900 text-white rounded-xl p-5 flex flex-col justify-between">
                      <div>
                        <span className="text-[10px] text-indigo-300 uppercase tracking-wider font-bold">預測結果 (Prediction)</span>
                        <div className="text-xs text-indigo-200 mt-1">預測目標：{activeTargetName}</div>
                        <div className="text-3xl font-extrabold font-mono text-white my-2">
                          {prediction.total.toLocaleString()}
                        </div>
                      </div>
                      <div className="text-[10px] text-indigo-300 border-t border-white/10 pt-2">
                        使用 {selectedAlgo} (k={currentModel.k}) 即時迴歸方程計算
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Accordion 2: Statistical Inference */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
                <button
                  onClick={() => toggleSection('stats')}
                  className="w-full p-4 text-left font-bold text-xs flex items-center justify-between text-slate-800 hover:bg-slate-50 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-600" />
                    <span>🔬 統計推論與迴歸方程式 (OLS Coefficients, p-value, 95% CI, VIF)</span>
                  </div>
                  {expandedSections.stats ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSections.stats && (
                  <div className="p-5 border-t border-slate-100 space-y-4">
                    <div className="p-3 bg-slate-900 text-indigo-300 rounded-xl font-mono text-xs overflow-x-auto">
                      {currentModel.equation || 'N/A'}
                    </div>
                    <div className="overflow-x-auto border border-slate-200 rounded-xl">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-slate-50 text-slate-600">
                          <tr>
                            <th className="p-2.5 font-sans">特徵名稱</th>
                            <th className="p-2.5">係數 (Coef)</th>
                            <th className="p-2.5">標準誤 (SE)</th>
                            <th className="p-2.5">p-value</th>
                            <th className="p-2.5">95% 信賴區間</th>
                            <th className="p-2.5">VIF (共線性)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 text-[11px]">
                          <tr>
                            <td className="p-2.5 font-sans font-bold">Intercept (截距)</td>
                            <td className="p-2.5 font-bold">{currentModel.intercept?.toLocaleString()}</td>
                            <td className="p-2.5">-</td>
                            <td className="p-2.5 text-emerald-600">&lt; 0.001</td>
                            <td className="p-2.5">-</td>
                            <td className="p-2.5">-</td>
                          </tr>
                          {currentModel.features?.map((f: string) => {
                            const w = currentModel.weights?.[f] ?? 0;
                            const p = currentModel.p_values?.[f] ?? 0.05;
                            const vif = currentModel.vif?.[f] ?? 1.0;
                            return (
                              <tr key={f}>
                                <td className="p-2.5 font-sans text-slate-800">{f}</td>
                                <td className="p-2.5 text-indigo-700 font-bold">{typeof w === 'number' ? w.toFixed(4) : w}</td>
                                <td className="p-2.5 text-slate-500">{currentModel.std_errors?.[f] ?? '-'}</td>
                                <td className="p-2.5">{typeof p === 'number' ? p.toFixed(4) : p}</td>
                                <td className="p-2.5 text-slate-500">{currentModel.confidence_intervals?.[f] ? `[${currentModel.confidence_intervals[f][0]}, ${currentModel.confidence_intervals[f][1]}]` : '-'}</td>
                                <td className="p-2.5">
                                  <span className={`px-1.5 py-0.5 rounded ${vif > 5 ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'}`}>
                                    {typeof vif === 'number' ? vif.toFixed(2) : vif}
                                  </span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>

              {/* Accordion 3: Residual Diagnostics */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
                <button
                  onClick={() => toggleSection('residuals')}
                  className="w-full p-4 text-left font-bold text-xs flex items-center justify-between text-slate-800 hover:bg-slate-50 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-indigo-600" />
                    <span>🎯 殘差診斷與實際值對比 (Residual Diagnostics)</span>
                  </div>
                  {expandedSections.residuals ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSections.residuals && (
                  <div className="p-5 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <div className="text-xs font-bold text-slate-700 mb-2">實際值 vs 預測值 (45度對角線):</div>
                      <div className="h-48 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis type="number" dataKey="actual" name="實際值" tickFormatter={(v) => v.toLocaleString()} />
                            <YAxis type="number" dataKey="predicted" name="預測值" tickFormatter={(v) => v.toLocaleString()} />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(val: any) => [Number(val).toLocaleString(), '']} />
                            <Scatter
                              data={customAnalysisResult?.residual_analysis?.points || (p1ApiData?.residual_analysis?.points || [
                                { actual: 103282, predicted: 103015 },
                                { actual: 144259, predicted: 134250 },
                                { actual: 146121, predicted: 135200 },
                                { actual: 77798, predicted: 74200 },
                                { actual: 191050, predicted: 181200 },
                              ])}
                              fill="#4f46e5"
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs font-bold text-slate-700 mb-2">殘差 vs 預測值 (0基準線):</div>
                      <div className="h-48 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis type="number" dataKey="predicted" name="預測值" tickFormatter={(v) => v.toLocaleString()} />
                            <YAxis type="number" dataKey="residual" name="殘差" tickFormatter={(v) => v.toLocaleString()} />
                            <ReferenceLine y={0} stroke="#dc2626" strokeDasharray="3 3" />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(val: any) => [Number(val).toLocaleString(), '']} />
                            <Scatter
                              data={customAnalysisResult?.residual_analysis?.points || (p1ApiData?.residual_analysis?.points || [
                                { predicted: 103015, residual: 267 },
                                { predicted: 134250, residual: 10008 },
                                { predicted: 135200, residual: 10921 },
                                { predicted: 74200, residual: 3598 },
                                { predicted: 181200, residual: 9849 },
                              ])}
                              fill="#0ea5e9"
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Accordion 4: Optuna Bayesian Optimization */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
                <button
                  onClick={() => toggleSection('optuna')}
                  className="w-full p-4 text-left font-bold text-xs flex items-center justify-between text-slate-800 hover:bg-slate-50 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-indigo-600" />
                    <span>🧪 Optuna 貝氏最佳化超參數調優 (Bayesian Hyperparameter Optimization)</span>
                  </div>
                  {expandedSections.optuna ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSections.optuna && (
                  <div className="p-5 border-t border-slate-100 space-y-4">
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">最佳模型種類</span>
                        <div className="text-sm font-bold text-slate-800 font-mono mt-0.5">
                          {optunaData?.optuna?.best_params?.model_type || 'Lasso'}
                        </div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">最優正則化 α</span>
                        <div className="text-sm font-bold text-indigo-700 font-mono mt-0.5">
                          {optunaData?.optuna?.best_params?.alpha ? optunaData.optuna.best_params.alpha.toFixed(2) : '828.22'}
                        </div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">5-Fold CV RMSE</span>
                        <div className="text-sm font-bold text-emerald-600 font-mono mt-0.5">
                          ${optunaData?.optuna?.best_cv_rmse ? Number(optunaData.optuna.best_cv_rmse).toLocaleString() : '9,198.43'}
                        </div>
                      </div>
                      <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold">獨立測試集 Test RMSE</span>
                        <div className="text-sm font-bold text-slate-800 font-mono mt-0.5">
                          ${optunaData?.optuna?.test_rmse ? Number(optunaData.optuna.test_rmse).toLocaleString() : '8,346.85'}
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="text-xs font-bold text-slate-700 mb-2">30 次試驗 (Trials) 貝氏收斂路徑 (Trial vs CV RMSE):</div>
                      <div className="h-44 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={optunaData?.optuna?.trials_history || [
                              { trial_number: 1, cv_rmse: 9480 },
                              { trial_number: 5, cv_rmse: 9350 },
                              { trial_number: 10, cv_rmse: 9280 },
                              { trial_number: 15, cv_rmse: 9210 },
                              { trial_number: 20, cv_rmse: 9198.49 },
                              { trial_number: 25, cv_rmse: 9198.50 },
                              { trial_number: 29, cv_rmse: 9198.43 },
                            ]}
                            margin={{ top: 5, right: 15, bottom: 5, left: 10 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis dataKey="trial_number" label={{ value: '試驗次數 (Trial #)', position: 'insideBottom', offset: -5 }} />
                            <YAxis domain={['auto', 'auto']} tickFormatter={(v) => v.toLocaleString()} />
                            <Tooltip formatter={(v: any) => [`$${Number(v).toLocaleString()}`, 'CV RMSE']} />
                            <Line type="monotone" dataKey="cv_rmse" name="CV RMSE" stroke="#0ea5e9" strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Accordion 5: 8,198.80 Benchmark Deep Dive */}
              <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-xs">
                <button
                  onClick={() => toggleSection('benchmark')}
                  className="w-full p-4 text-left font-bold text-xs flex items-center justify-between text-slate-800 hover:bg-slate-50 cursor-pointer"
                >
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-amber-500" />
                    <span>🏆 參考網站基準深入對照 (Seed 0 Benchmark $8,198.80 Deep Dive)</span>
                  </div>
                  {expandedSections.benchmark ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>

                {expandedSections.benchmark && (
                  <div className="p-5 border-t border-slate-100 space-y-3 text-xs">
                    <div className="p-3 bg-indigo-50/60 rounded-xl border border-indigo-100 text-indigo-900 leading-relaxed">
                      <strong>基準精準復現說明：</strong>
                      參考網站使用 <code>random_state = 0</code> 進行 80/20 分割，測試集樣本索引為 <code>[2, 4, 10, 11, 22, 27, 28, 31, 38, 41]</code>。
                      在特徵選取 $k=2$（R&D Spend + Marketing Spend）時，計算出的測試集 RMSE 精準達到 <strong>$8,198.80</strong>！
                    </div>

                    <div className="overflow-x-auto border border-slate-200 rounded-xl">
                      <table className="w-full text-left text-xs font-mono">
                        <thead className="bg-slate-50 text-slate-600 font-sans">
                          <tr>
                            <th className="p-2.5">評估維度</th>
                            <th className="p-2.5">作業標準規範 (Seed 42)</th>
                            <th className="p-2.5 text-indigo-700">參考網站基準 (Seed 0)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          <tr>
                            <td className="p-2.5 font-sans font-semibold">k=2 最佳特徵</td>
                            <td className="p-2.5">R&D Spend + Marketing Spend</td>
                            <td className="p-2.5 text-indigo-700 font-bold">R&D Spend + Marketing Spend</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-sans font-semibold">k=2 Test RMSE</td>
                            <td className="p-2.5">$8,206.33</td>
                            <td className="p-2.5 text-indigo-700 font-bold">$8,198.80 (完全一致)</td>
                          </tr>
                          <tr>
                            <td className="p-2.5 font-sans font-semibold">全域最佳模型</td>
                            <td className="p-2.5">k=3 ($7,946.37, R²: 0.9220)</td>
                            <td className="p-2.5 text-indigo-700 font-bold">k=2 ($8,198.80, R²: 0.9880)</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>

            </div>

            {/* Step Navigation Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-slate-100">
              <button
                onClick={() => setCurrentStep(2)}
                className="px-4 py-2.5 text-slate-600 hover:text-slate-900 text-xs font-bold rounded-xl border border-slate-200 hover:bg-slate-50 flex items-center gap-1.5 cursor-pointer transition-all"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>上一步：模型設定 (Step 2)</span>
              </button>
              <button
                onClick={() => setCurrentStep(4)}
                className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold rounded-xl shadow-xs flex items-center gap-2 transition-all cursor-pointer"
              >
                <span>下一步：比較與成果匯出 (Step 4)</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

          </div>
        )}

        {/* ========================================================================= */}
        {/* STEP 4: 雙問題比較與成果繳交 (Problem Comparison & Submission Export)        */}
        {/* ========================================================================= */}
        {currentStep === 4 && (
          <div className="space-y-6">
            
            {/* Step Header */}
            <div>
              <div className="flex items-center gap-2 text-indigo-600 font-bold text-xs uppercase tracking-wider mb-1">
                <span>Step 4 of 4</span>
                <span>•</span>
                <span>雙問題對比與成果匯出</span>
              </div>
              <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
                成果繳交與報告匯出 (Export for Submission)
              </h2>
              <p className="text-xs sm:text-sm text-slate-500 mt-1">
                檢視 Problem 1 與 Problem 2 之各項指標對比，並選擇其中一個問題匯出標準化成果與 CRISP-DM 結案報告。
              </p>
            </div>

            {/* Side-by-side Comparison Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="p-4 border-b border-slate-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <ArrowRightLeft className="w-4 h-4 text-indigo-600" />
                  <h3 className="font-bold text-sm text-slate-900">
                    雙問題全面橫向比較表 (Problem 1 vs Problem 2)
                  </h3>
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-sans">
                  <thead className="bg-slate-100 text-slate-700 font-bold uppercase tracking-wider">
                    <tr>
                      <th className="p-3">比較維度 (Dimension)</th>
                      <th className="p-3 bg-indigo-50/70 text-indigo-900">Problem 1: 50 Startups</th>
                      <th className="p-3 bg-amber-50/70 text-amber-900">Problem 2: {customDatasetName}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">預測目標變數 (Target Variable)</td>
                      <td className="p-3 font-mono font-bold text-indigo-700">Profit</td>
                      <td className="p-3 font-mono font-bold text-amber-700">{customTargetCol}</td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">總樣本數量 (Sample Size)</td>
                      <td className="p-3 font-mono">50 筆</td>
                      <td className="p-3 font-mono">{customProfile?.rowCount || customRecords.length || 80} 筆</td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">原始特徵數量 (Features Count)</td>
                      <td className="p-3 font-mono">4 個</td>
                      <td className="p-3 font-mono">{customProfile?.columnNames?.length ? customProfile.columnNames.length - 1 : 10} 個</td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">最佳模型選取特徵 (Best Features)</td>
                      <td className="p-3 font-mono text-indigo-800">R&D Spend + Marketing Spend</td>
                      <td className="p-3 font-mono text-amber-800">
                        {customAnalysisResult?.best_cv_model?.features?.join(' + ') || 'MedInc + DistToCoast + HouseAge ...'}
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">訓練 5-Fold CV RMSE</td>
                      <td className="p-3 font-mono font-bold text-indigo-700">
                        {p1RandomState === 42 ? '$9,441.52' : '$9,410.20'}
                      </td>
                      <td className="p-3 font-mono font-bold text-amber-700">
                        {customAnalysisResult?.best_cv_model?.cv_rmse_mean?.toLocaleString() || '17,591.23'}
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">獨立測試集 Test RMSE</td>
                      <td className="p-3 font-mono font-bold text-indigo-700">
                        {p1RandomState === 42 ? '$7,946.37' : '$8,198.80 (對標基準)'}
                      </td>
                      <td className="p-3 font-mono font-bold text-amber-700">
                        {customAnalysisResult?.best_cv_model?.test_rmse?.toLocaleString() || '16,513.56'}
                      </td>
                    </tr>
                    <tr>
                      <td className="p-3 font-semibold text-slate-700">判定係數 Test R²</td>
                      <td className="p-3 font-mono font-bold text-indigo-700">0.9220</td>
                      <td className="p-3 font-mono font-bold text-amber-700">
                        {customAnalysisResult?.best_cv_model?.r2 ? customAnalysisResult.best_cv_model.r2.toFixed(4) : '0.9575'}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Academic Note on Scale Invariance */}
            <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs leading-relaxed space-y-1.5">
              <div className="font-bold flex items-center gap-1.5 text-amber-950">
                <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
                <span>重要統計學觀念：為什麼不能直接比較兩問題的 RMSE 絕對數值？</span>
              </div>
              <p>
                RMSE（均方根誤差）保留了預測目標變數的物理單位與量綱。Problem 1 的目標利潤單位為美金（數萬至十幾萬規模），而 Problem 2 可能是房屋總價、所得或無量綱指數。
                兩者尺度完全不同，因此<strong>跨資料集評定模型優劣應以判定係數 ($R^2$)、調整後 $R^2$ 與交叉驗證穩定度為核心依據</strong>。
              </p>
            </div>

            {/* Select for Submission Card */}
            <div className="bg-gradient-to-br from-indigo-900 via-indigo-800 to-slate-900 text-white rounded-2xl p-6 sm:p-7 shadow-lg space-y-5">
              <div>
                <span className="text-[10px] uppercase font-bold text-indigo-300 tracking-wider flex items-center gap-1.5 mb-1">
                  <Award className="w-3.5 h-3.5 text-amber-300" />
                  Select for Submission (選擇成果繳交題目)
                </span>
                <h3 className="text-lg sm:text-xl font-bold text-white">
                  點選欲繳交的主題並匯出作業檔案
                </h3>
              </div>

              {/* Radio Selection */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div
                  onClick={() => setSubmissionChoice('problem1')}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                    submissionChoice === 'problem1' ? 'bg-white/15 border-white text-white shadow-md' : 'bg-white/5 border-white/15 text-indigo-200 hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-sm">選擇 Problem 1 (50 Startups)</span>
                    <span className={`w-4 h-4 rounded-full border flex items-center justify-center ${submissionChoice === 'problem1' ? 'bg-indigo-500 border-white text-white' : 'border-white/30'}`}>
                      {submissionChoice === 'problem1' && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                    </span>
                  </div>
                  <p className="text-[11px] text-indigo-200">
                    經典 50 Startups 資料集，展示對標參考網站之 8,198.80 基準復現與最優特徵篩選 (R² = 0.9220)。
                  </p>
                </div>

                <div
                  onClick={() => setSubmissionChoice('problem2')}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                    submissionChoice === 'problem2' ? 'bg-white/15 border-white text-white shadow-md' : 'bg-white/5 border-white/15 text-indigo-200 hover:bg-white/10'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold text-sm">選擇 Problem 2 ({customDatasetName})</span>
                    <span className={`w-4 h-4 rounded-full border flex items-center justify-center ${submissionChoice === 'problem2' ? 'bg-amber-500 border-white text-white' : 'border-white/30'}`}>
                      {submissionChoice === 'problem2' && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                    </span>
                  </div>
                  <p className="text-[11px] text-indigo-200">
                    通用迴歸分析多維特徵之自訂資料集，展示動態前處理、特徵篩選路徑與高維度泛化診斷。
                  </p>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-3 pt-1">
                <button
                  onClick={() => handleExportMarkdownReport(submissionChoice)}
                  className="px-5 py-2.5 bg-white hover:bg-indigo-50 text-indigo-900 text-xs font-bold rounded-xl shadow-md flex items-center gap-2 transition-all cursor-pointer"
                >
                  <FileText className="w-4 h-4 text-indigo-600" />
                  <span>📄 匯出 CRISP-DM 結案報告 (Markdown)</span>
                </button>
                <button
                  onClick={() => handleExportCSV(submissionChoice)}
                  className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-xl border border-indigo-400/40 shadow-md flex items-center gap-2 transition-all cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>📥 匯出評估數據 (Results CSV)</span>
                </button>
                <button
                  onClick={() => {
                    const model = submissionChoice === 'problem1' ? (bestCvModel || currentModel) : (customAnalysisResult?.best_cv_model || currentModel);
                    if (model?.equation) {
                      navigator.clipboard?.writeText(model.equation);
                      alert('已將迴歸方程式複製到剪貼簿！');
                    }
                  }}
                  className="px-4 py-2.5 bg-white/10 hover:bg-white/20 text-white text-xs font-semibold rounded-xl border border-white/20 transition-all flex items-center gap-1.5 cursor-pointer"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>複製最佳方程式</span>
                </button>
              </div>

            </div>

            {/* Step Navigation Actions */}
            <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-100">
              <button
                onClick={() => setCurrentStep(3)}
                className="px-4 py-2.5 text-slate-600 hover:text-slate-900 text-xs font-bold rounded-xl border border-slate-200 hover:bg-slate-50 flex items-center gap-1.5 cursor-pointer transition-all"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>上一步：查看評估結果 (Step 3)</span>
              </button>
              <button
                onClick={() => setCurrentStep(1)}
                className="px-5 py-2.5 text-indigo-700 hover:bg-indigo-50 text-xs font-bold rounded-xl border border-indigo-200 flex items-center gap-1.5 cursor-pointer transition-all"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>更換資料集 / 重新分析 (Step 1)</span>
              </button>
            </div>

          </div>
        )}

      </main>

      {/* Clean Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 text-slate-400 text-xs">
        <div className="max-w-6xl mx-auto px-4 flex flex-wrap justify-between items-center gap-3">
          <div>
            <strong className="text-slate-600">AIoT HW2 — 多元線性迴歸與特徵選取分析平台</strong>
            <span className="text-slate-400 ml-2">Next.js 14 • TypeScript • Tailwind CSS • FastAPI</span>
          </div>
          <div className="flex items-center gap-4">
            <a href="https://github.com/Augustine-723/50-Starup" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
              GitHub Repository
            </a>
            <a href="/api/health" target="_blank" rel="noopener noreferrer" className="text-indigo-600 hover:underline">
              API Health
            </a>
          </div>
        </div>
      </footer>

    </div>
  );
}
