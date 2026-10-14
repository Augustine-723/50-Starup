/**
 * 50 Startups ML Regression Dashboard - Interactive Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Tab Navigation
  initTabs();

  // 2. Render Results Table
  renderResultsTable();

  // 3. Render Chart.js Visualizations (Dual side-by-side + Train/Test)
  initCharts();

  // 4. Initialize RFE Explorer
  initRfeExplorer();

  // 5. Initialize Live Profit Predictor
  initPredictor();

  // 6. Initialize Dataset Explorer
  initDatasetExplorer();
});

/* ==========================================================================
   1. Tab Navigation
   ========================================================================== */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add('active');
      }

      // Trigger chart resize if needed
      if (window.rmseChart) window.rmseChart.resize();
      if (window.r2Chart) window.r2Chart.resize();
      if (window.trainTestChart) window.trainTestChart.resize();
      if (window.coefChart) window.coefChart.resize();
    });
  });
}

/* ==========================================================================
   2. Render Evaluation Results Table
   ========================================================================== */
function renderResultsTable() {
  const tbody = document.getElementById('results-table-body');
  if (!tbody || !STARTUPS_DATA || !STARTUPS_DATA.models) return;

  tbody.innerHTML = '';

  for (let k = 1; k <= 5; k++) {
    const m = STARTUPS_DATA.models[k];
    if (!m) continue;

    const tr = document.createElement('tr');
    if (k === 3) {
      tr.className = 'highlight-best';
    }

    const featureTags = m.features.map(f => {
      const isKey = f === 'R&D Spend';
      return `<span class="feature-tag ${isKey ? 'key-feature' : ''}">${f}</span>`;
    }).join(' ');

    const statusBadge = k === 3
      ? '<span class="badge badge-success">🏆 最佳模型 (Optimal)</span>'
      : (k <= 2 
          ? '<span class="badge" style="background: rgba(244,63,94,0.15); color: #fb7185;">欠擬合</span>'
          : '<span class="badge" style="background: rgba(245,158,11,0.15); color: #fbbf24;">微幅過擬合</span>');

    tr.innerHTML = `
      <td><strong style="color: var(--accent-cyan); font-size: 1.05rem;">${m.k}</strong></td>
      <td><div class="feature-tag-list">${featureTags}</div></td>
      <td style="font-family: var(--font-mono);">${m.train_rmse.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
      <td style="font-family: var(--font-mono); font-weight: 700; color: ${k === 3 ? '#34d399' : 'var(--text-bright)'};">
        ${m.test_rmse.toLocaleString(undefined, {minimumFractionDigits: 2})}
      </td>
      <td style="font-family: var(--font-mono); color: var(--text-muted); font-size: 0.82rem;">
        ${m.test_mse.toLocaleString(undefined, {minimumFractionDigits: 2})}
      </td>
      <td style="font-family: var(--font-mono);">${m.test_mae.toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
      <td style="font-family: var(--font-mono); font-weight: 700; color: ${m.test_r2 > 0 ? '#38bdf8' : '#fb7185'};">
        ${m.test_r2.toFixed(4)}
      </td>
      <td>${statusBadge}</td>
    `;

    tbody.appendChild(tr);
  }
}

/* ==========================================================================
   3. Chart.js Visualizations
   ========================================================================== */
function initCharts() {
  const kLabels = [1, 2, 3, 4, 5];
  const rmseValues = kLabels.map(k => STARTUPS_DATA.models[k].test_rmse);
  const r2Values = kLabels.map(k => STARTUPS_DATA.models[k].test_r2);
  const trainRmseValues = kLabels.map(k => STARTUPS_DATA.models[k].train_rmse);

  const commonGrid = {
    color: 'rgba(255, 255, 255, 0.07)',
    drawBorder: false
  };

  const commonTicks = {
    color: '#94a3b8',
    font: { family: 'Outfit', size: 12 }
  };

  // 1. Left Chart: RMSE by Number of Features (Blue line with circular markers)
  const ctxRmse = document.getElementById('chart-rmse');
  if (ctxRmse) {
    window.rmseChart = new Chart(ctxRmse, {
      type: 'line',
      data: {
        labels: kLabels,
        datasets: [{
          label: 'RMSE (Root Mean Squared Error)',
          data: rmseValues,
          borderColor: '#1f77b4',
          backgroundColor: 'rgba(31, 119, 180, 0.1)',
          pointBackgroundColor: '#1f77b4',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
          borderWidth: 2.5,
          tension: 0.15,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#f8fafc',
            bodyColor: '#38bdf8',
            borderColor: 'rgba(255, 255, 255, 0.15)',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: (ctx) => `RMSE: $${ctx.raw.toLocaleString(undefined, {minimumFractionDigits: 2})}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Number of Features', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          },
          y: {
            title: { display: true, text: 'RMSE (Dollars)', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          }
        }
      }
    });
  }

  // 2. Right Chart: R-squared by Number of Features (Blue line with circular markers)
  const ctxR2 = document.getElementById('chart-r2');
  if (ctxR2) {
    window.r2Chart = new Chart(ctxR2, {
      type: 'line',
      data: {
        labels: kLabels,
        datasets: [{
          label: 'R-squared',
          data: r2Values,
          borderColor: '#1f77b4',
          backgroundColor: 'rgba(31, 119, 180, 0.1)',
          pointBackgroundColor: '#1f77b4',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 6,
          pointHoverRadius: 8,
          borderWidth: 2.5,
          tension: 0.15,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#0f172a',
            titleColor: '#f8fafc',
            bodyColor: '#38bdf8',
            borderColor: 'rgba(255, 255, 255, 0.15)',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: (ctx) => `R²: ${ctx.raw.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Number of Features', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          },
          y: {
            title: { display: true, text: 'R-squared', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          }
        }
      }
    });
  }

  // 3. Training vs Testing RMSE Chart (Overfitting / Underfitting)
  const ctxTrainTest = document.getElementById('chart-train-test');
  if (ctxTrainTest) {
    window.trainTestChart = new Chart(ctxTrainTest, {
      type: 'line',
      data: {
        labels: kLabels,
        datasets: [
          {
            label: 'Training RMSE',
            data: trainRmseValues,
            borderColor: '#38bdf8',
            backgroundColor: 'transparent',
            pointBackgroundColor: '#38bdf8',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointStyle: 'circle',
            borderWidth: 2.5
          },
          {
            label: 'Testing RMSE',
            data: rmseValues,
            borderColor: '#f97316',
            backgroundColor: 'transparent',
            pointBackgroundColor: '#f97316',
            pointBorderColor: '#ffffff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointStyle: 'rectRot',
            borderWidth: 2.5,
            borderDash: [6, 4]
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: { color: '#e2e8f0', font: { size: 13, family: 'Outfit', weight: '600' } }
          },
          tooltip: {
            backgroundColor: '#0f172a',
            padding: 12,
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: $${ctx.raw.toLocaleString(undefined, {minimumFractionDigits: 2})}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Number of Features', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          },
          y: {
            title: { display: true, text: 'RMSE (Dollars)', color: '#cbd5e1', font: { weight: 'bold', size: 13 } },
            grid: commonGrid,
            ticks: commonTicks
          }
        }
      }
    });
  }
}

/* ==========================================================================
   4. RFE Feature Selection Explorer
   ========================================================================== */
function initRfeExplorer() {
  const stepBtns = document.querySelectorAll('.step-btn');
  const formulaBox = document.getElementById('rfe-formula-display');
  const tagsContainer = document.getElementById('rfe-selected-tags');

  function updateRfeView(k) {
    const model = STARTUPS_DATA.models[k];
    if (!model) return;

    // 1. Build Equation String
    let formula = `Profit = ${model.intercept.toLocaleString()}`;
    for (const [feat, coef] of Object.entries(model.coefficients)) {
      const sign = coef >= 0 ? '+' : '-';
      const absCoef = Math.abs(coef).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 4});
      formula += ` ${sign} (${absCoef} × [${feat}])`;
    }
    if (formulaBox) {
      formulaBox.innerHTML = `<strong>迴歸方程式 (k=${k}):</strong><br>${formula}`;
    }

    // 2. Feature Tags
    if (tagsContainer) {
      tagsContainer.innerHTML = model.features.map(f => {
        const isKey = f === 'R&D Spend';
        return `<span class="feature-tag ${isKey ? 'key-feature' : ''}" style="font-size: 0.9rem; padding: 0.4rem 0.8rem;">
          ${f} (係數: ${model.coefficients[f]})
        </span>`;
      }).join('');
    }

    // 3. Coefficient Chart
    updateCoefChart(model);
  }

  stepBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      stepBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const k = parseInt(btn.getAttribute('data-k'), 10);
      updateRfeView(k);
    });
  });

  // Default view: k=3 (Optimal)
  updateRfeView(3);
}

function updateCoefChart(model) {
  const ctx = document.getElementById('chart-coefficients');
  if (!ctx) return;

  const labels = Object.keys(model.coefficients);
  const data = Object.values(model.coefficients);
  const colors = data.map(v => v >= 0 ? '#38bdf8' : '#f43f5e');

  if (window.coefChart) {
    window.coefChart.destroy();
  }

  window.coefChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: '迴歸係數 (Regression Weight)',
        data: data,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          callbacks: {
            label: (ctx) => `係數: ${ctx.raw.toLocaleString()}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.07)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f1f5f9', font: { family: 'Outfit', size: 11 } }
        }
      }
    }
  });
}

/* ==========================================================================
   5. Interactive Profit Predictor
   ========================================================================== */
function initPredictor() {
  const sliderRd = document.getElementById('slider-rd');
  const inputRd = document.getElementById('input-rd');
  const dispRd = document.getElementById('val-display-rd');

  const sliderAdmin = document.getElementById('slider-admin');
  const inputAdmin = document.getElementById('input-admin');
  const dispAdmin = document.getElementById('val-display-admin');

  const sliderMktg = document.getElementById('slider-mktg');
  const inputMktg = document.getElementById('input-mktg');
  const dispMktg = document.getElementById('val-display-mktg');

  const modelSelect = document.getElementById('predict-model-select');
  const statePills = document.querySelectorAll('.radio-pill');

  let activeState = 'New York';

  // Sync Slider <-> Input
  function syncControl(slider, input, display) {
    const update = (val) => {
      slider.value = val;
      input.value = val;
      display.textContent = `$${parseFloat(val).toLocaleString()}`;
      calculatePrediction();
    };

    slider.addEventListener('input', (e) => update(e.target.value));
    input.addEventListener('input', (e) => update(e.target.value));
  }

  syncControl(sliderRd, inputRd, dispRd);
  syncControl(sliderAdmin, inputAdmin, dispAdmin);
  syncControl(sliderMktg, inputMktg, dispMktg);

  // State selection
  statePills.forEach(pill => {
    pill.addEventListener('click', () => {
      statePills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      activeState = pill.getAttribute('data-state');
      calculatePrediction();
    });
  });

  if (modelSelect) {
    modelSelect.addEventListener('change', calculatePrediction);
  }

  function calculatePrediction() {
    const k = parseInt(modelSelect ? modelSelect.value : 3, 10);
    const model = STARTUPS_DATA.models[k];
    if (!model) return;

    const rdVal = parseFloat(sliderRd.value) || 0;
    const adminVal = parseFloat(sliderAdmin.value) || 0;
    const mktgVal = parseFloat(sliderMktg.value) || 0;

    // Dummy values
    const isFlorida = activeState === 'Florida' ? 1 : 0;
    const isNewYork = activeState === 'New York' ? 1 : 0;

    let predicted = model.intercept;
    let bdRdVal = 0;
    let bdStateVal = 0;

    if (model.coefficients['R&D Spend']) {
      bdRdVal = model.coefficients['R&D Spend'] * rdVal;
      predicted += bdRdVal;
    }
    if (model.coefficients['Administration']) {
      predicted += model.coefficients['Administration'] * adminVal;
    }
    if (model.coefficients['Marketing Spend']) {
      predicted += model.coefficients['Marketing Spend'] * mktgVal;
    }
    if (model.coefficients['State_Florida']) {
      bdStateVal += model.coefficients['State_Florida'] * isFlorida;
    }
    if (model.coefficients['State_New York']) {
      bdStateVal += model.coefficients['State_New York'] * isNewYork;
    }
    predicted += bdStateVal;

    // Update DOM
    const resultEl = document.getElementById('predict-result-amount');
    const subEl = document.getElementById('predict-result-sub');
    const bdInterceptEl = document.getElementById('bd-intercept');
    const bdRdEl = document.getElementById('bd-rd');
    const bdStateEl = document.getElementById('bd-state');
    const bdTotalEl = document.getElementById('bd-total');

    if (resultEl) {
      resultEl.textContent = `$${Math.round(predicted).toLocaleString()}`;
    }
    if (subEl) {
      subEl.textContent = `基於 ${k} Features 線性迴歸模型 (州別: ${activeState})`;
    }
    if (bdInterceptEl) bdInterceptEl.textContent = `+$${model.intercept.toLocaleString()}`;
    if (bdRdEl) bdRdEl.textContent = `${bdRdVal >= 0 ? '+' : ''}$${Math.round(bdRdVal).toLocaleString()}`;
    if (bdStateEl) bdStateEl.textContent = `${bdStateVal >= 0 ? '+' : ''}$${Math.round(bdStateVal).toLocaleString()}`;
    if (bdTotalEl) bdTotalEl.textContent = `$${Math.round(predicted).toLocaleString()}`;
  }

  // Initial calculation
  calculatePrediction();
}

/* ==========================================================================
   6. Dataset Explorer Table & Search
   ========================================================================== */
function initDatasetExplorer() {
  const tbody = document.getElementById('raw-table-body');
  const searchInput = document.getElementById('table-search-input');
  const stateFilter = document.getElementById('table-state-filter');
  const badgeCount = document.getElementById('badge-record-count');

  if (!tbody || !STARTUPS_DATA || !STARTUPS_DATA.dataset) return;

  function renderRows() {
    const query = (searchInput.value || '').trim().toLowerCase();
    const filterState = stateFilter.value;

    const filtered = STARTUPS_DATA.dataset.filter((row, idx) => {
      if (filterState !== 'ALL' && row.State !== filterState) return false;
      if (!query) return true;
      const rowStr = `${row['R&D Spend']} ${row['Administration']} ${row['Marketing Spend']} ${row['State']} ${row['Profit']}`.toLowerCase();
      return rowStr.includes(query);
    });

    if (badgeCount) {
      badgeCount.textContent = `顯示 ${filtered.length} / 50 筆資料`;
    }

    tbody.innerHTML = filtered.map((row, i) => `
      <tr>
        <td style="color: var(--text-dim); font-size: 0.8rem;">#${i + 1}</td>
        <td style="font-family: var(--font-mono); font-weight: 600; color: #38bdf8;">$${row['R&D Spend'].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
        <td style="font-family: var(--font-mono);">$${row['Administration'].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
        <td style="font-family: var(--font-mono);">$${row['Marketing Spend'].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
        <td><span class="feature-tag">${row['State']}</span></td>
        <td style="font-family: var(--font-mono); font-weight: 700; color: #34d399;">$${row['Profit'].toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
      </tr>
    `).join('');
  }

  if (searchInput) searchInput.addEventListener('input', renderRows);
  if (stateFilter) stateFilter.addEventListener('change', renderRows);

  renderRows();
}
