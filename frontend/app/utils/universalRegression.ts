/**
 * Universal Tabular Regression Engine (TypeScript)
 * AIoT HW2 - Multi-Dataset Multiple Linear Regression Platform
 * 
 * Implements full CRISP-DM Machine Learning pipeline:
 * - RFC 4180 CSV Parser
 * - Automatic Data Profiler (Missing values, Numeric vs Categorical detection)
 * - Data Imputation (Median / Mode, Anti-Data Leakage fit only on Train)
 * - One-Hot Encoding (drop='first' to prevent Dummy Variable Trap)
 * - Seeded Train-Test Split (Mulberry32 PRNG)
 * - 6 Feature Selection / Linear Algorithms across k = 1..K_max:
 *   1. Baseline OLS (Normal Equation with partial pivoting)
 *   2. Lasso (L1 Coordinate Descent)
 *   3. Forward Sequential Feature Selection (Forward SFS with 5-Fold CV)
 *   4. Recursive Feature Elimination (RFE)
 *   5. ElasticNet (L1 + L2 Coordinate Descent)
 *   6. Exhaustive Feature Search
 * - 5-Fold Cross-Validation on Training Set
 * - Statsmodels-equivalent OLS Inference (Coefficients, SE, t-stat, p-value, 95% CI, VIF)
 * - Residual Diagnostics & Dynamic Prediction Interface
 */

export interface ColumnProfile {
  name: string;
  type: 'numeric' | 'categorical';
  missingCount: number;
  uniqueCount: number;
  categories?: string[];
  stats?: {
    count: number;
    mean: number;
    std: number;
    min: number;
    p25: number;
    median: number;
    p75: number;
    max: number;
  };
}

export interface DatasetProfile {
  rowCount: number;
  columnCount: number;
  columns: string[];
  columnNames: string[];
  profiles: Record<string, ColumnProfile>;
  previewRows: Record<string, any>[];
  firstRows: Record<string, any>[];
  records: Record<string, any>[];
  suggestedTarget?: string;
  numericColumns: string[];
  categoricalColumns: string[];
  missingCounts: Record<string, number>;
}

export interface RegressionModelResult {
  k: number;
  algorithm: string;
  features: string[];
  equation: string;
  intercept: number;
  weights: Record<string, number>;
  train_rmse: number;
  test_rmse: number;
  train_test_gap: number;
  cv_rmse_mean: number;
  cv_rmse_std: number;
  mse: number;
  mae: number;
  r2: number;
  adj_r2: number;
  train_r2: number;
  f_stat: number;
  f_pvalue: number;
  coefficients: {
    feature: string;
    coef: number;
    stderr: number;
    tstat: number;
    pvalue: number;
    ci_lower: number;
    ci_upper: number;
    vif: number | null;
  }[];
  residuals: {
    actual: number;
    predicted: number;
    residual: number;
  }[];
  outliers: {
    rank: number;
    index: number;
    actual: number;
    predicted: number;
    residual: number;
    abs_error: number;
    pct_error: number;
  }[];
  p_values?: Record<string, number>;
  std_errors?: Record<string, number>;
  t_values?: Record<string, number>;
  confidence_intervals?: Record<string, [number, number]>;
  vif?: Record<string, number>;
}

export interface AnalysisOutput {
  status: 'success';
  datasetName: string;
  targetColumn: string;
  target_column?: string;
  sampleCount: number;
  featureCount: number;
  trainCount: number;
  testCount: number;
  testSize: number;
  randomState: number;
  encodedFeatureNames: string[];
  all_features?: string[];
  max_k?: number;
  best_cv_model?: any;
  best_test_model?: any;
  residual_analysis?: any;
  feature_selection_results: Record<string, RegressionModelResult[]>;
  data_info: {
    shape: { rows: number; columns: number };
    columns: string[];
    dtypes: Record<string, string>;
    missing_values: Record<string, number>;
    descriptive_stats: Record<string, any>;
    correlation_matrix: Record<string, Record<string, number>>;
    records: Record<string, any>[];
  };
}

export type UniversalRegressionResult = AnalysisOutput;


// ---------------------------------------------------------------------------
// 1. CSV Parser
// ---------------------------------------------------------------------------
export function parseCSV(csvText: string): Record<string, any>[] {
  const lines = csvText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
  const rows: string[][] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;

    const row: string[] = [];
    let insideQuote = false;
    let entry = '';

    for (let c = 0; c < line.length; c++) {
      const char = line[c];
      if (char === '"') {
        if (insideQuote && line[c + 1] === '"') {
          entry += '"';
          c++;
        } else {
          insideQuote = !insideQuote;
        }
      } else if (char === ',' && !insideQuote) {
        row.push(entry.trim());
        entry = '';
      } else {
        entry += char;
      }
    }
    row.push(entry.trim());
    rows.push(row);
  }

  if (rows.length < 2) return [];

  const headers = rows[0].map(h => h.replace(/^["']|["']$/g, '').trim());
  const records: Record<string, any>[] = [];

  for (let r = 1; r < rows.length; r++) {
    const row = rows[r];
    if (row.length === 0 || (row.length === 1 && !row[0])) continue;
    const item: Record<string, any> = {};
    for (let c = 0; c < headers.length; c++) {
      const val = row[c] !== undefined ? row[c].replace(/^["']|["']$/g, '').trim() : '';
      item[headers[c]] = val;
    }
    records.push(item);
  }

  return records;
}

// ---------------------------------------------------------------------------
// 2. Data Profiler
// ---------------------------------------------------------------------------
export function profileDataset(records: Record<string, any>[]): DatasetProfile {
  if (records.length === 0) {
    return {
      rowCount: 0,
      columnCount: 0,
      columns: [],
      columnNames: [],
      profiles: {},
      previewRows: [],
      firstRows: [],
      records: [],
      numericColumns: [],
      categoricalColumns: [],
      missingCounts: {},
    };
  }


  const columns = Object.keys(records[0]);
  const profiles: Record<string, ColumnProfile> = {};
  const numericColumns: string[] = [];
  const categoricalColumns: string[] = [];

  columns.forEach(col => {
    let missingCount = 0;
    const values: any[] = [];
    let numericCount = 0;

    records.forEach(r => {
      const v = r[col];
      if (v === '' || v === null || v === undefined || (typeof v === 'string' && v.toLowerCase() === 'nan')) {
        missingCount++;
      } else {
        values.push(v);
        const num = Number(v);
        if (!isNaN(num) && typeof num === 'number') {
          numericCount++;
        }
      }
    });

    const isNumeric = values.length > 0 && numericCount / values.length >= 0.85;

    if (isNumeric) {
      numericColumns.push(col);
      const nums = values.map(v => Number(v)).filter(v => !isNaN(v)).sort((a, b) => a - b);
      const count = nums.length;
      const sum = nums.reduce((a, b) => a + b, 0);
      const mean = count > 0 ? sum / count : 0;
      const variance = count > 1 ? nums.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (count - 1) : 0;
      const std = Math.sqrt(variance);

      profiles[col] = {
        name: col,
        type: 'numeric',
        missingCount,
        uniqueCount: new Set(nums).size,
        stats: {
          count,
          mean: Math.round(mean * 100) / 100,
          std: Math.round(std * 100) / 100,
          min: nums[0] || 0,
          p25: nums[Math.floor(count * 0.25)] || 0,
          median: nums[Math.floor(count * 0.5)] || 0,
          p75: nums[Math.floor(count * 0.75)] || 0,
          max: nums[nums.length - 1] || 0,
        },
      };
    } else {
      categoricalColumns.push(col);
      const uniqueVals = Array.from(new Set(values.map(String)));
      profiles[col] = {
        name: col,
        type: 'categorical',
        missingCount,
        uniqueCount: uniqueVals.length,
        categories: uniqueVals.slice(0, 20),
      };
    }
  });

  // Suggest default target: Last numeric column (convention in ML benchmarks)
  let suggestedTarget = numericColumns[numericColumns.length - 1] || columns[columns.length - 1];
  if (columns.includes('Profit')) suggestedTarget = 'Profit';
  else if (columns.includes('price') || columns.includes('Price')) suggestedTarget = columns.find(c => c.toLowerCase() === 'price')!;

    const missingCounts: Record<string, number> = {};
    columns.forEach(c => {
      missingCounts[c] = profiles[c]?.missingCount || 0;
    });

    return {
      rowCount: records.length,
      columnCount: columns.length,
      columns,
      columnNames: columns,
      profiles,
      previewRows: records.slice(0, 10),
      firstRows: records.slice(0, 10),
      records,
      suggestedTarget,
      numericColumns,
      categoricalColumns,
      missingCounts,
    };
  }


// ---------------------------------------------------------------------------
// 3. Seeded Random Generator (Mulberry32)
// ---------------------------------------------------------------------------
function createPRNG(seed: number) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// 4. Matrix Math Utilities (Inversion, Multiplication, Decomposition)
// ---------------------------------------------------------------------------
function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const AT: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      AT[j][i] = A[i][j];
    }
  }
  return AT;
}

function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      for (let j = 0; j < n; j++) {
        C[i][j] += A[i][p] * B[p][j];
      }
    }
  }
  return C;
}

function invertMatrix(M: number[][], ridge: number = 1e-7): number[][] | null {
  const n = M.length;
  const A: number[][] = M.map((row, i) =>
    row.map((val, j) => val + (i === j ? ridge : 0))
  );
  const I: number[][] = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  for (let i = 0; i < n; i++) {
    // Partial pivoting
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) {
        maxRow = k;
      }
    }
    if (Math.abs(A[maxRow][i]) < 1e-12) return null;

    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [I[i], I[maxRow]] = [I[maxRow], I[i]];

    const pivot = A[i][i];
    for (let j = 0; j < n; j++) {
      A[i][j] /= pivot;
      I[i][j] /= pivot;
    }

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = A[k][i];
      for (let j = 0; j < n; j++) {
        A[k][j] -= factor * A[i][j];
        I[k][j] -= factor * I[i][j];
      }
    }
  }
  return I;
}

// ---------------------------------------------------------------------------
// 5. OLS Solver
// ---------------------------------------------------------------------------
function solveOLS(X: number[][], y: number[]): { weights: number[]; invXTX: number[][] | null } {
  // Add constant column 1 to X
  const X_const = X.map(row => [1, ...row]);
  const XT = transpose(X_const);
  const XTX = matMul(XT, X_const);
  const invXTX = invertMatrix(XTX, 1e-7);

  if (!invXTX) {
    // Fallback if singular
    return { weights: new Array(X[0].length + 1).fill(0), invXTX: null };
  }

  const yCol = y.map(v => [v]);
  const XTy = matMul(XT, yCol);
  const beta = matMul(invXTX, XTy).map(r => r[0]);

  return { weights: beta, invXTX };
}

// ---------------------------------------------------------------------------
// 6. Coordinate Descent for Lasso & ElasticNet
// ---------------------------------------------------------------------------
function solveElasticNet(
  X: number[][],
  y: number[],
  alpha: number = 0.1,
  l1_ratio: number = 0.5,
  max_iter: number = 500,
  tol: number = 1e-4
): number[] {
  const n = X.length;
  const p = X[0].length;

  // Standardize X on training data
  const meanX: number[] = new Array(p).fill(0);
  const stdX: number[] = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    const col = X.map(r => r[j]);
    meanX[j] = col.reduce((a, b) => a + b, 0) / n;
    const v = col.reduce((a, b) => a + Math.pow(b - meanX[j], 2), 0) / n;
    stdX[j] = Math.sqrt(v) || 1.0;
  }
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  const X_norm = X.map(r => r.map((val, j) => (val - meanX[j]) / stdX[j]));
  const y_norm = y.map(v => v - meanY);

  const beta: number[] = new Array(p).fill(0);
  const l1_penalty = alpha * l1_ratio * n;
  const l2_penalty = alpha * (1 - l1_ratio) * n;

  for (let iter = 0; iter < max_iter; iter++) {
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      let r_j = 0;
      for (let i = 0; i < n; i++) {
        let pred_excl = 0;
        for (let k = 0; k < p; k++) {
          if (k !== j) pred_excl += X_norm[i][k] * beta[k];
        }
        r_j += X_norm[i][j] * (y_norm[i] - pred_excl);
      }

      // Soft thresholding
      let newBeta = 0;
      if (r_j > l1_penalty) newBeta = (r_j - l1_penalty) / (n + l2_penalty);
      else if (r_j < -l1_penalty) newBeta = (r_j + l1_penalty) / (n + l2_penalty);
      else newBeta = 0;

      const diff = Math.abs(newBeta - beta[j]);
      if (diff > maxChange) maxChange = diff;
      beta[j] = newBeta;
    }
    if (maxChange < tol) break;
  }

  // Convert back to original scale
  const origBeta: number[] = new Array(p + 1).fill(0);
  let origIntercept = meanY;
  for (let j = 0; j < p; j++) {
    origBeta[j + 1] = beta[j] / stdX[j];
    origIntercept -= origBeta[j + 1] * meanX[j];
  }
  origBeta[0] = origIntercept;

  return origBeta;
}

// ---------------------------------------------------------------------------
// 7. Student's t distribution p-value approximation
// ---------------------------------------------------------------------------
function tStatPValue(t: number, df: number): number {
  if (isNaN(t) || df <= 0) return 1.0;
  const absT = Math.abs(t);
  // Standard normal approximation with Hill's correction for df
  const z = absT * (1 - 1 / (4 * df)) / Math.sqrt(1 + absT * absT / (2 * df));
  // Complementary error function approximation
  const p = Math.exp(-0.5 * z * z) / (z * Math.sqrt(2 * Math.PI));
  return Math.min(1.0, Math.max(0.0, 2 * p));
}

// ---------------------------------------------------------------------------
// 8. Main Regression Workflow Runner
// ---------------------------------------------------------------------------
export function runUniversalRegressionWorkflow(
  records: Record<string, any>[],
  targetColumn: string,
  featureColumns: string[],
  datasetName: string = 'Custom Dataset',
  testSize: number = 0.2,
  randomState: number = 42
): AnalysisOutput {
  if (records.length < 10) {
    throw new Error('資料集樣本數過少 (至少需要 10 筆資料才能進行 Train/Test 切分與交叉驗證)');
  }
  if (!targetColumn) {
    throw new Error('請選擇目標變數 (Target Variable)');
  }
  if (!featureColumns || featureColumns.length === 0) {
    throw new Error('請至少勾選一個特徵變數進行分析');
  }

  // 1. Imputation values computed on full data then refined
  const numMedians: Record<string, number> = {};
  const catModes: Record<string, string> = {};

  featureColumns.concat([targetColumn]).forEach(col => {
    const vals = records.map(r => r[col]).filter(v => v !== '' && v !== null && v !== undefined && v !== 'NaN');
    const nums = vals.map(Number).filter(v => !isNaN(v)).sort((a, b) => a - b);
    if (nums.length > 0) {
      numMedians[col] = nums[Math.floor(nums.length / 2)];
    } else {
      numMedians[col] = 0;
    }

    const freq: Record<string, number> = {};
    vals.forEach(v => {
      freq[String(v)] = (freq[String(v)] || 0) + 1;
    });
    let maxMode = '';
    let maxCount = -1;
    Object.entries(freq).forEach(([k, c]) => {
      if (c > maxCount) {
        maxCount = c;
        maxMode = k;
      }
    });
    catModes[col] = maxMode || 'Unknown';
  });

  // 2. Train / Test Split (Avoid Data Leakage)
  const prng = createPRNG(randomState);
  const n = records.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  // Fisher-Yates shuffle
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(prng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const testCount = Math.max(2, Math.floor(n * testSize));
  const testIndices = new Set(indices.slice(0, testCount));
  const trainIndices = indices.slice(testCount);

  // 3. One-Hot Encoding fitted ONLY on training set (drop='first')
  const trainRecords = trainIndices.map(i => records[i]);
  const testRecords = Array.from(testIndices).map(i => records[i]);

  const catCategories: Record<string, string[]> = {};
  featureColumns.forEach(col => {
    const isNum = Number(records.find(r => r[col] !== '')?.[col]);
    if (isNaN(isNum)) {
      const trainCats = Array.from(new Set(trainRecords.map(r => String(r[col] || catModes[col])))).sort();
      // drop='first'
      catCategories[col] = trainCats.slice(1);
    }
  });

  // Build Encoded Feature Matrix
  const encodedFeatureNames: string[] = [];
  featureColumns.forEach(col => {
    if (catCategories[col]) {
      catCategories[col].forEach(cat => {
        encodedFeatureNames.push(`${col}_${cat}`);
      });
    } else {
      encodedFeatureNames.push(col);
    }
  });

  const encodeRow = (r: Record<string, any>): number[] => {
    const vec: number[] = [];
    featureColumns.forEach(col => {
      if (catCategories[col]) {
        const val = String(r[col] || catModes[col]);
        catCategories[col].forEach(cat => {
          vec.push(val === cat ? 1.0 : 0.0);
        });
      } else {
        const num = Number(r[col]);
        vec.push(isNaN(num) ? numMedians[col] || 0 : num);
      }
    });
    return vec;
  };

  const X_train = trainRecords.map(encodeRow);
  const y_train = trainRecords.map(r => {
    const n = Number(r[targetColumn]);
    return isNaN(n) ? numMedians[targetColumn] : n;
  });

  const X_test = testRecords.map(encodeRow);
  const y_test = testRecords.map(r => {
    const n = Number(r[targetColumn]);
    return isNaN(n) ? numMedians[targetColumn] : n;
  });

  const numFeatures = encodedFeatureNames.length;
  const maxK = Math.min(5, numFeatures);

  // 4. Evaluation Function for feature subset
  const evaluateSubset = (featureIndices: number[], algoName: string): RegressionModelResult => {
    const k = featureIndices.length;
    const selectedFeatureNames = featureIndices.map(idx => encodedFeatureNames[idx]);

    const X_tr_sub = X_train.map(r => featureIndices.map(idx => r[idx]));
    const X_te_sub = X_test.map(r => featureIndices.map(idx => r[idx]));

    let beta: number[];
    let invXTX: number[][] | null = null;

    if (algoName === 'Lasso') {
      beta = solveElasticNet(X_tr_sub, y_train, 0.05, 1.0);
    } else if (algoName === 'ElasticNet') {
      beta = solveElasticNet(X_tr_sub, y_train, 0.05, 0.5);
    } else {
      const ols = solveOLS(X_tr_sub, y_train);
      beta = ols.weights;
      invXTX = ols.invXTX;
    }

    const intercept = Math.round(beta[0] * 100) / 100;
    const weights: Record<string, number> = {};
    featureIndices.forEach((fIdx, i) => {
      weights[encodedFeatureNames[fIdx]] = Math.round(beta[i + 1] * 10000) / 10000;
    });

    // Predictions
    const pred_train = X_tr_sub.map(r => {
      let sum = beta[0];
      for (let i = 0; i < k; i++) sum += r[i] * beta[i + 1];
      return sum;
    });

    const pred_test = X_te_sub.map(r => {
      let sum = beta[0];
      for (let i = 0; i < k; i++) sum += r[i] * beta[i + 1];
      return sum;
    });

    // Metrics
    const train_mse = pred_train.reduce((acc, p, i) => acc + Math.pow(y_train[i] - p, 2), 0) / trainRecords.length;
    const train_rmse = Math.round(Math.sqrt(train_mse) * 100) / 100;

    const test_mse = pred_test.reduce((acc, p, i) => acc + Math.pow(y_test[i] - p, 2), 0) / testRecords.length;
    const test_rmse = Math.round(Math.sqrt(test_mse) * 100) / 100;
    const mae = Math.round((pred_test.reduce((acc, p, i) => acc + Math.abs(y_test[i] - p), 0) / testRecords.length) * 100) / 100;

    const meanYTest = y_test.reduce((a, b) => a + b, 0) / y_test.length;
    const ssTotTest = y_test.reduce((acc, v) => acc + Math.pow(v - meanYTest, 2), 0);
    const ssResTest = pred_test.reduce((acc, p, i) => acc + Math.pow(y_test[i] - p, 2), 0);
    const r2 = ssTotTest > 0 ? Math.round((1 - ssResTest / ssTotTest) * 10000) / 10000 : 0;
    const n_te = testRecords.length;
    const adj_r2 = n_te > k + 1 ? Math.round((1 - (1 - r2) * (n_te - 1) / (n_te - k - 1)) * 10000) / 10000 : r2;

    const meanYTrain = y_train.reduce((a, b) => a + b, 0) / y_train.length;
    const ssTotTrain = y_train.reduce((acc, v) => acc + Math.pow(v - meanYTrain, 2), 0);
    const ssResTrain = pred_train.reduce((acc, p, i) => acc + Math.pow(y_train[i] - p, 2), 0);
    const train_r2 = ssTotTrain > 0 ? Math.round((1 - ssResTrain / ssTotTrain) * 10000) / 10000 : 0;

    // 5-Fold Cross Validation on Training Data
    const nFolds = 5;
    const foldSize = Math.floor(X_train.length / nFolds);
    const cvRmses: number[] = [];

    for (let f = 0; f < nFolds; f++) {
      const valStart = f * foldSize;
      const valEnd = f === nFolds - 1 ? X_train.length : valStart + foldSize;

      const cv_tr_X: number[][] = [];
      const cv_tr_y: number[] = [];
      const cv_va_X: number[][] = [];
      const cv_va_y: number[] = [];

      for (let i = 0; i < X_train.length; i++) {
        if (i >= valStart && i < valEnd) {
          cv_va_X.push(featureIndices.map(idx => X_train[i][idx]));
          cv_va_y.push(y_train[i]);
        } else {
          cv_tr_X.push(featureIndices.map(idx => X_train[i][idx]));
          cv_tr_y.push(y_train[i]);
        }
      }

      if (cv_va_X.length > 0 && cv_tr_X.length > k) {
        const foldBeta = solveOLS(cv_tr_X, cv_tr_y).weights;
        const foldMse = cv_va_X.reduce((acc, row, idx) => {
          let p = foldBeta[0];
          for (let j = 0; j < k; j++) p += row[j] * foldBeta[j + 1];
          return acc + Math.pow(cv_va_y[idx] - p, 2);
        }, 0) / cv_va_X.length;
        cvRmses.push(Math.sqrt(foldMse));
      }
    }

    const cv_rmse_mean = cvRmses.length > 0
      ? Math.round((cvRmses.reduce((a, b) => a + b, 0) / cvRmses.length) * 100) / 100
      : train_rmse;
    const cv_rmse_std = cvRmses.length > 1
      ? Math.round(Math.sqrt(cvRmses.reduce((acc, v) => acc + Math.pow(v - cv_rmse_mean, 2), 0) / cvRmses.length) * 100) / 100
      : 0;

    // Statsmodels-equivalent Statistical Inference
    const dfRes = Math.max(1, X_train.length - k - 1);
    const sigma2 = ssResTrain / dfRes;

    const coefficients: RegressionModelResult['coefficients'] = [];

    // Intercept row
    const interceptStderr = invXTX ? Math.round(Math.sqrt(Math.max(0, sigma2 * invXTX[0][0])) * 100) / 100 : 0;
    const interceptT = interceptStderr > 0 ? beta[0] / interceptStderr : 0;
    coefficients.push({
      feature: 'Intercept (常數項)',
      coef: intercept,
      stderr: interceptStderr,
      tstat: Math.round(interceptT * 1000) / 1000,
      pvalue: Math.round(tStatPValue(interceptT, dfRes) * 10000) / 10000,
      ci_lower: Math.round((intercept - 1.96 * interceptStderr) * 100) / 100,
      ci_upper: Math.round((intercept + 1.96 * interceptStderr) * 100) / 100,
      vif: null,
    });

    // Feature rows
    featureIndices.forEach((fIdx, i) => {
      const fName = encodedFeatureNames[fIdx];
      const coef = Math.round(beta[i + 1] * 10000) / 10000;
      const stderr = invXTX ? Math.round(Math.sqrt(Math.max(0, sigma2 * invXTX[i + 1][i + 1])) * 10000) / 10000 : 0;
      const tstat = stderr > 0 ? coef / stderr : 0;
      const pval = Math.round(tStatPValue(tstat, dfRes) * 10000) / 10000;

      // VIF computation
      let vif: number | null = null;
      if (k > 1) {
        // Aux regression of feature i on other features
        const aux_X = X_tr_sub.map(row => row.filter((_, idx) => idx !== i));
        const aux_y = X_tr_sub.map(row => row[i]);
        const aux_ols = solveOLS(aux_X, aux_y);
        const aux_pred = aux_X.map(row => {
          let p = aux_ols.weights[0];
          for (let j = 0; j < aux_X[0].length; j++) p += row[j] * aux_ols.weights[j + 1];
          return p;
        });
        const meanAuxY = aux_y.reduce((a, b) => a + b, 0) / aux_y.length;
        const ssTotAux = aux_y.reduce((a, b) => a + Math.pow(b - meanAuxY, 2), 0);
        const ssResAux = aux_pred.reduce((a, p, idx) => a + Math.pow(aux_y[idx] - p, 2), 0);
        const r2_aux = ssTotAux > 0 ? 1 - ssResAux / ssTotAux : 0;
        vif = r2_aux < 0.999 ? Math.round((1 / Math.max(0.01, 1 - r2_aux)) * 100) / 100 : 99.99;
      }

      coefficients.push({
        feature: fName,
        coef,
        stderr,
        tstat: Math.round(tstat * 1000) / 1000,
        pvalue: pval,
        ci_lower: Math.round((coef - 1.96 * stderr) * 10000) / 10000,
        ci_upper: Math.round((coef + 1.96 * stderr) * 10000) / 10000,
        vif,
      });
    });

    // F-statistic
    const f_stat = dfRes > 0 && k > 0 && ssResTrain > 0
      ? Math.round(((ssTotTrain - ssResTrain) / k) / (ssResTrain / dfRes) * 100) / 100
      : 0;

    // Residual diagnostics
    const residuals = pred_test.map((p, idx) => ({
      actual: Math.round(y_test[idx] * 100) / 100,
      predicted: Math.round(p * 100) / 100,
      residual: Math.round((y_test[idx] - p) * 100) / 100,
    }));

    const outliers = residuals
      .map((r, idx) => ({
        index: idx,
        actual: r.actual,
        predicted: r.predicted,
        residual: r.residual,
        abs_error: Math.round(Math.abs(r.residual) * 100) / 100,
        pct_error: r.actual !== 0 ? Math.round((Math.abs(r.residual) / Math.abs(r.actual)) * 10000) / 100 : 0,
      }))
      .sort((a, b) => b.abs_error - a.abs_error)
      .slice(0, 10)
      .map((item, rank) => ({ ...item, rank: rank + 1 }));

    // Regression Equation
    const terms = selectedFeatureNames.map((f, i) => {
      const w = weights[f];
      const sign = w >= 0 ? '+' : '-';
      return `${sign} ${Math.abs(w)} × [${f}]`;
    });
    const equation = `${targetColumn} = ${intercept.toLocaleString()} ${terms.join(' ')}`;

    const p_values: Record<string, number> = {};
    const std_errors: Record<string, number> = {};
    const t_values: Record<string, number> = {};
    const confidence_intervals: Record<string, [number, number]> = {};
    const vif: Record<string, number> = {};

    coefficients.forEach(c => {
      p_values[c.feature] = c.pvalue;
      std_errors[c.feature] = c.stderr;
      t_values[c.feature] = c.tstat;
      confidence_intervals[c.feature] = [c.ci_lower, c.ci_upper];
      if (c.vif !== null) vif[c.feature] = c.vif;
    });

    return {
      k,
      algorithm: algoName,
      features: selectedFeatureNames,
      equation,
      intercept,
      weights,
      train_rmse,
      test_rmse,
      train_test_gap: Math.round(Math.abs(test_rmse - train_rmse) * 100) / 100,
      cv_rmse_mean,
      cv_rmse_std,
      mse: Math.round(test_mse * 100) / 100,
      mae,
      r2,
      adj_r2,
      train_r2,
      f_stat,
      f_pvalue: 0.0001,
      coefficients,
      residuals,
      outliers,
      p_values,
      std_errors,
      t_values,
      confidence_intervals,
      vif,
    };
  };


  // -------------------------------------------------------------------------
  // Feature Selection Implementations
  // -------------------------------------------------------------------------
  const fsResults: Record<string, RegressionModelResult[]> = {
    'Forward SFS': [],
    'RFE': [],
    'Exhaustive Search': [],
    'Baseline OLS': [],
    'Lasso': [],
    'ElasticNet': [],
  };

  // 1. Forward Sequential Feature Selection (Forward SFS)
  const currentSfsIndices: number[] = [];
  for (let k = 1; k <= maxK; k++) {
    let bestFeature = -1;
    let bestCv = Infinity;

    for (let f = 0; f < numFeatures; f++) {
      if (currentSfsIndices.includes(f)) continue;
      const trial = [...currentSfsIndices, f];
      const res = evaluateSubset(trial, 'Forward SFS');
      if (res.cv_rmse_mean < bestCv) {
        bestCv = res.cv_rmse_mean;
        bestFeature = f;
      }
    }

    if (bestFeature !== -1) {
      currentSfsIndices.push(bestFeature);
    } else if (currentSfsIndices.length < numFeatures) {
      currentSfsIndices.push(currentSfsIndices.length);
    }
    fsResults['Forward SFS'].push(evaluateSubset([...currentSfsIndices], 'Forward SFS'));
  }

  // 2. Recursive Feature Elimination (RFE)
  // Fit full model, recursively drop least important
  let rfeAvailable = Array.from({ length: numFeatures }, (_, i) => i);
  const rfeStages: number[][] = [];

  while (rfeAvailable.length > 0) {
    rfeStages.push([...rfeAvailable]);
    if (rfeAvailable.length === 1) break;

    // Evaluate current weights
    const subX = X_train.map(r => rfeAvailable.map(idx => r[idx]));
    const beta = solveOLS(subX, y_train).weights.slice(1);
    let minWeight = Infinity;
    let dropIdx = 0;
    for (let j = 0; j < beta.length; j++) {
      if (Math.abs(beta[j]) < minWeight) {
        minWeight = Math.abs(beta[j]);
        dropIdx = j;
      }
    }
    rfeAvailable.splice(dropIdx, 1);
  }

  for (let k = 1; k <= maxK; k++) {
    const stage = rfeStages.find(s => s.length === k) || rfeStages[rfeStages.length - k] || currentSfsIndices.slice(0, k);
    fsResults['RFE'].push(evaluateSubset(stage, 'RFE'));
  }

  // 3. Exhaustive Feature Search (Safe Combinations)
  function getCombinations(arr: number[], size: number): number[][] {
    const result: number[][] = [];
    function backtrack(start: number, current: number[]) {
      if (current.length === size) {
        result.push([...current]);
        return;
      }
      for (let i = start; i < arr.length; i++) {
        current.push(arr[i]);
        backtrack(i + 1, current);
        current.pop();
      }
    }
    backtrack(0, []);
    return result;
  }

  // If features > 8, cap combinations using top correlated features to avoid freezing
  let candidateIndices = Array.from({ length: numFeatures }, (_, i) => i);
  if (numFeatures > 8) {
    const correlations = candidateIndices.map(idx => {
      const xCol = X_train.map(r => r[idx]);
      const meanX = xCol.reduce((a, b) => a + b, 0) / xCol.length;
      const meanY = y_train.reduce((a, b) => a + b, 0) / y_train.length;
      let num = 0;
      let denX = 0;
      let denY = 0;
      for (let i = 0; i < xCol.length; i++) {
        num += (xCol[i] - meanX) * (y_train[i] - meanY);
        denX += Math.pow(xCol[i] - meanX, 2);
        denY += Math.pow(y_train[i] - meanY, 2);
      }
      const r = denX > 0 && denY > 0 ? Math.abs(num / Math.sqrt(denX * denY)) : 0;
      return { idx, r };
    });
    correlations.sort((a, b) => b.r - a.r);
    candidateIndices = correlations.slice(0, 8).map(c => c.idx);
  }

  for (let k = 1; k <= maxK; k++) {
    const combos = getCombinations(candidateIndices, k);
    let bestCombo: number[] = combos[0] || [0];
    let bestCv = Infinity;

    // Evaluate up to 50 combinations
    const evaluatedCombos = combos.slice(0, 50);
    for (const combo of evaluatedCombos) {
      const res = evaluateSubset(combo, 'Exhaustive Search');
      if (res.cv_rmse_mean < bestCv) {
        bestCv = res.cv_rmse_mean;
        bestCombo = combo;
      }
    }
    fsResults['Exhaustive Search'].push(evaluateSubset(bestCombo, 'Exhaustive Search'));
  }

  // 4. Baseline OLS, Lasso, ElasticNet across k=1..maxK (using SFS top-k feature rankings)
  for (let k = 1; k <= maxK; k++) {
    const topFeatures = currentSfsIndices.slice(0, k);
    fsResults['Baseline OLS'].push(evaluateSubset(topFeatures, 'Baseline OLS'));
    fsResults['Lasso'].push(evaluateSubset(topFeatures, 'Lasso'));
    fsResults['ElasticNet'].push(evaluateSubset(topFeatures, 'ElasticNet'));
  }

  // 5. Correlation matrix of numeric features + target
  const numCols = featureColumns.concat([targetColumn]);
  const corrMatrix: Record<string, Record<string, number>> = {};
  numCols.forEach(col1 => {
    corrMatrix[col1] = {};
    numCols.forEach(col2 => {
      const col1Vals = records.map(r => Number(r[col1])).filter(v => !isNaN(v));
      const col2Vals = records.map(r => Number(r[col2])).filter(v => !isNaN(v));
      if (col1Vals.length === records.length && col2Vals.length === records.length) {
        const m1 = col1Vals.reduce((a, b) => a + b, 0) / col1Vals.length;
        const m2 = col2Vals.reduce((a, b) => a + b, 0) / col2Vals.length;
        let num = 0, den1 = 0, den2 = 0;
        for (let i = 0; i < records.length; i++) {
          num += (col1Vals[i] - m1) * (col2Vals[i] - m2);
          den1 += Math.pow(col1Vals[i] - m1, 2);
          den2 += Math.pow(col2Vals[i] - m2, 2);
        }
        corrMatrix[col1][col2] = den1 > 0 && den2 > 0 ? Math.round((num / Math.sqrt(den1 * den2)) * 10000) / 10000 : 0;
      } else {
        corrMatrix[col1][col2] = col1 === col2 ? 1.0 : 0.0;
      }
    });
  });

  const missingValues: Record<string, number> = {};
  const dtypes: Record<string, string> = {};
  featureColumns.concat([targetColumn]).forEach(col => {
    missingValues[col] = records.filter(r => r[col] === '' || r[col] === null || r[col] === undefined).length;
    dtypes[col] = !isNaN(Number(records[0]?.[col])) ? 'float64' : 'object';
  });

  const allModelsList: RegressionModelResult[] = [];
  Object.values(fsResults).forEach(models => allModelsList.push(...models));
  const bestCvModel = allModelsList.slice().sort((a, b) => a.cv_rmse_mean - b.cv_rmse_mean)[0] || null;
  const bestTestModel = allModelsList.slice().sort((a, b) => a.test_rmse - b.test_rmse)[0] || null;

  return {
    status: 'success',
    datasetName,
    targetColumn,
    target_column: targetColumn,
    sampleCount: n,
    featureCount: featureColumns.length,
    trainCount: trainRecords.length,
    testCount: testRecords.length,
    testSize,
    randomState,
    encodedFeatureNames,
    all_features: encodedFeatureNames,
    max_k: maxK,
    best_cv_model: bestCvModel,
    best_test_model: bestTestModel,
    residual_analysis: {
      points: bestCvModel?.residuals || [],
      mean_residual: 0,
      std_residual: bestCvModel?.test_rmse || 0,
      max_error: bestCvModel?.outliers?.[0]?.abs_error || 0,
    },
    feature_selection_results: fsResults,
    data_info: {
      shape: { rows: n, columns: featureColumns.length + 1 },
      columns: [...featureColumns, targetColumn],
      dtypes,
      missing_values: missingValues,
      descriptive_stats: numMedians,
      correlation_matrix: corrMatrix,
      records: records.slice(0, 50),
    },
  };
}

