/* ═══════════════════════════════════════════════════════
   DASHBOARD APP — Chart.js Visualizations + Interactive Table
   ═══════════════════════════════════════════════════════ */

(async function () {
    'use strict';

    // ─── Load Data ───────────────────────────────────────
    let DATA;
    try {
        const res = await fetch('results.json');
        DATA = await res.json();
    } catch (e) {
        document.body.innerHTML = '<div style="padding:60px;text-align:center;color:#ef4444;font-family:sans-serif"><h2>Could not load results.json</h2><p>Run the Rust backtester first:<br><code>cd backtester && cargo run --release</code></p></div>';
        return;
    }

    const { config, composite, stocks, summary, execution_time_ms } = DATA;

    // ─── Helpers ─────────────────────────────────────────
    function formatINR(val) {
        if (val === null || val === undefined || isNaN(val)) return '—';
        const abs = Math.abs(val);
        const sign = val < 0 ? '-' : '';
        if (abs >= 1e7) return sign + '₹' + (abs / 1e7).toFixed(2) + ' Cr';
        if (abs >= 1e5) return sign + '₹' + (abs / 1e5).toFixed(2) + ' L';
        return sign + '₹' + abs.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    }

    function formatINRFull(val) {
        if (val === null || val === undefined || isNaN(val)) return '—';
        return '₹' + Math.round(val).toLocaleString('en-IN');
    }

    function pct(val) {
        if (val === null || val === undefined || isNaN(val)) return 'N/A';
        return val.toFixed(2) + '%';
    }

    // ─── KPI Cards ───────────────────────────────────────
    document.getElementById('execTime').textContent = execution_time_ms + 'ms';

    if (composite) {
        document.getElementById('compositeXirr').textContent = pct(composite.xirr_pct);
        document.getElementById('compositePortfolio').textContent = 'Portfolio: ' + formatINR(composite.final_portfolio);
    }

    document.getElementById('stocksAnalyzed').textContent = summary.total_stocks_analyzed;
    document.getElementById('validResults').textContent = summary.valid_results + ' valid results';
    document.getElementById('avgXirr').textContent = pct(summary.avg_xirr);
    document.getElementById('xirrRange').textContent = pct(summary.worst_xirr) + ' → ' + pct(summary.best_xirr);
    document.getElementById('bestSymbol').textContent = summary.best_symbol;
    document.getElementById('bestXirr').textContent = pct(summary.best_xirr) + ' XIRR';
    document.getElementById('profitableCount').textContent = summary.positive_xirr_count;
    document.getElementById('profitablePct').textContent = ((summary.positive_xirr_count / summary.valid_results) * 100).toFixed(1) + '% of all stocks';
    document.getElementById('above10').textContent = summary.above_10_count;
    document.getElementById('above15').textContent = summary.above_15_count + ' above 15% · ' + summary.above_20_count + ' above 20%';

    // ─── Config Bar ──────────────────────────────────────
    document.getElementById('cfgCapital').textContent = formatINR(config.initial_capital);
    document.getElementById('cfgLookback').textContent = config.lookback_days + ' days';
    document.getElementById('cfgDipRange').textContent = formatINR(config.min_trade_size) + ' → ' + formatINR(config.max_trade_size);
    document.getElementById('cfgSweep').textContent = config.profit_sweep_fraction_pct + '% @ +' + config.profit_sweep_target_pct + '%';
    document.getElementById('cfgLiquid').textContent = config.liquid_bees_rate_pct + '% p.a.';
    document.getElementById('cfgCooldown').textContent = config.cooldown_days + ' days';

    // ─── Composite Detail ────────────────────────────────
    if (composite) {
        const grid = document.getElementById('compositeGrid');
        const metrics = [
            { label: 'True XIRR', value: pct(composite.xirr_pct), cls: composite.xirr_pct > 0 ? 'positive' : 'negative' },
            { label: 'Final Portfolio', value: formatINRFull(composite.final_portfolio), cls: 'neutral' },
            { label: 'Equity Value', value: formatINRFull(composite.equity_value), cls: 'neutral' },
            { label: 'Liquid BeES', value: formatINRFull(composite.liquid_cash), cls: 'neutral' },
            { label: 'Dip Buys', value: composite.total_dip_buys, cls: 'neutral' },
            { label: 'Avg Trade Size', value: formatINRFull(composite.avg_trade_size), cls: 'neutral' },
            { label: 'Profit Sweeps', value: composite.profit_sweeps, cls: composite.profit_sweeps > 0 ? 'positive' : 'negative' },
            { label: 'Data Range', value: composite.data_start + ' → ' + composite.data_end, cls: 'neutral' },
        ];
        grid.innerHTML = metrics.map(m =>
            `<div class="comp-metric"><div class="comp-metric-label">${m.label}</div><div class="comp-metric-value ${m.cls}">${m.value}</div></div>`
        ).join('');
    }

    // ─── Data Range Footer ───────────────────────────────
    if (stocks.length > 0) {
        const validStocks = stocks.filter(s => !isNaN(s.xirr_pct));
        const starts = validStocks.map(s => s.data_start).sort();
        const ends = validStocks.map(s => s.data_end).sort();
        document.getElementById('dataRange').textContent =
            'Coverage: ' + starts[0] + ' → ' + ends[ends.length - 1] + ' · ' + stocks.length + ' stocks';
    }

    // ─── Chart.js Global Config ──────────────────────────
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 12;

    const validStocks = stocks.filter(s => !isNaN(s.xirr_pct) && s.xirr_pct !== null);

    // ─── Histogram Chart ─────────────────────────────────
    const bins = {};
    const binSize = 2;
    const minBin = Math.floor(Math.min(...validStocks.map(s => s.xirr_pct)) / binSize) * binSize;
    const maxBin = Math.ceil(Math.max(...validStocks.map(s => s.xirr_pct)) / binSize) * binSize;

    for (let b = minBin; b <= maxBin; b += binSize) bins[b] = 0;
    validStocks.forEach(s => {
        const b = Math.floor(s.xirr_pct / binSize) * binSize;
        bins[b] = (bins[b] || 0) + 1;
    });

    const histLabels = Object.keys(bins).map(Number).sort((a, b) => a - b);
    const histData = histLabels.map(b => bins[b]);
    const histColors = histLabels.map(b =>
        b >= 10 ? '#10b981' : b >= 5 ? '#6366f1' : b >= 0 ? '#8b5cf6' : '#ef4444'
    );

    new Chart(document.getElementById('histogramChart'), {
        type: 'bar',
        data: {
            labels: histLabels.map(b => b + '–' + (b + binSize) + '%'),
            datasets: [{
                data: histData,
                backgroundColor: histColors.map(c => c + '99'),
                borderColor: histColors,
                borderWidth: 1.5,
                borderRadius: 6,
                barPercentage: 0.85,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleFont: { weight: 600 },
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => ctx.raw + ' stocks'
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 11 } } },
                y: { beginAtZero: true, ticks: { stepSize: 5 }, grid: { color: 'rgba(255,255,255,0.04)' } }
            }
        }
    });

    // ─── Top 20 Bar Chart ────────────────────────────────
    const top20 = validStocks.slice(0, 20);
    new Chart(document.getElementById('top20Chart'), {
        type: 'bar',
        data: {
            labels: top20.map(s => s.symbol),
            datasets: [{
                data: top20.map(s => s.xirr_pct),
                backgroundColor: top20.map((s, i) => {
                    const alpha = 1 - (i * 0.035);
                    return i < 3 ? `rgba(16, 185, 129, ${alpha})` : `rgba(99, 102, 241, ${alpha})`;
                }),
                borderRadius: 5,
                barPercentage: 0.7,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => ctx.raw.toFixed(2) + '% XIRR'
                    }
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } },
                y: { grid: { display: false }, ticks: { font: { size: 11, weight: 600 } } }
            }
        }
    });

    // ─── Scatter Chart ───────────────────────────────────
    new Chart(document.getElementById('scatterChart'), {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Stocks',
                data: validStocks.map(s => ({
                    x: s.total_dip_buys,
                    y: s.profit_sweeps,
                    r: Math.max(4, Math.min(16, s.xirr_pct * 0.8)),
                    symbol: s.symbol,
                    xirr: s.xirr_pct,
                })),
                backgroundColor: validStocks.map(s =>
                    s.xirr_pct >= 10 ? 'rgba(16,185,129,0.5)' :
                    s.xirr_pct >= 5 ? 'rgba(99,102,241,0.5)' :
                    s.xirr_pct >= 0 ? 'rgba(139,92,246,0.4)' :
                    'rgba(239,68,68,0.5)'
                ),
                borderColor: validStocks.map(s =>
                    s.xirr_pct >= 10 ? '#10b981' :
                    s.xirr_pct >= 5 ? '#6366f1' :
                    s.xirr_pct >= 0 ? '#8b5cf6' :
                    '#ef4444'
                ),
                borderWidth: 1.5,
                pointRadius: validStocks.map(s => Math.max(4, Math.min(14, Math.abs(s.xirr_pct) * 0.6))),
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => {
                            const d = ctx.raw;
                            return `${d.symbol}: ${d.xirr.toFixed(2)}% XIRR · ${d.x} buys · ${d.y} sweeps`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Total Dip Buys', font: { weight: 600 } },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    title: { display: true, text: 'Profit Sweeps', font: { weight: 600 } },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            }
        }
    });

    // ─── Tier Doughnut Chart ─────────────────────────────
    const tiers = {
        'Negative': validStocks.filter(s => s.xirr_pct < 0).length,
        '0–5%': validStocks.filter(s => s.xirr_pct >= 0 && s.xirr_pct < 5).length,
        '5–8%': validStocks.filter(s => s.xirr_pct >= 5 && s.xirr_pct < 8).length,
        '8–10%': validStocks.filter(s => s.xirr_pct >= 8 && s.xirr_pct < 10).length,
        '10–15%': validStocks.filter(s => s.xirr_pct >= 10 && s.xirr_pct < 15).length,
        '15%+': validStocks.filter(s => s.xirr_pct >= 15).length,
    };

    new Chart(document.getElementById('tierChart'), {
        type: 'doughnut',
        data: {
            labels: Object.keys(tiers),
            datasets: [{
                data: Object.values(tiers),
                backgroundColor: [
                    'rgba(239,68,68,0.7)',
                    'rgba(245,158,11,0.7)',
                    'rgba(139,92,246,0.7)',
                    'rgba(99,102,241,0.7)',
                    'rgba(16,185,129,0.7)',
                    'rgba(52,211,153,0.8)',
                ],
                borderColor: '#0a0e1a',
                borderWidth: 3,
                hoverOffset: 8,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 16, usePointStyle: true, pointStyleWidth: 10, font: { size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    cornerRadius: 8,
                    callbacks: {
                        label: ctx => ctx.label + ': ' + ctx.raw + ' stocks'
                    }
                }
            }
        }
    });

    // ─── Leaderboard Table ───────────────────────────────
    const tbody = document.getElementById('leaderboardBody');
    let currentSort = { key: null, dir: 'desc' };

    function renderTable(data) {
        tbody.innerHTML = data.map((s, i) => {
            const rank = i + 1;
            const xirrClass = isNaN(s.xirr_pct) ? '' : (s.xirr_pct >= 0 ? 'xirr-positive' : 'xirr-negative');
            const topClass = rank <= 3 ? 'top-3' : '';
            return `<tr class="${topClass}">
                <td>${rank}</td>
                <td>${s.symbol}</td>
                <td class="${xirrClass}">${pct(s.xirr_pct)}</td>
                <td>${formatINRFull(s.final_portfolio)}</td>
                <td>${formatINRFull(s.equity_value)}</td>
                <td>${formatINRFull(s.liquid_cash)}</td>
                <td>${s.total_dip_buys}</td>
                <td>${s.profit_sweeps}</td>
                <td>${formatINRFull(s.avg_trade_size)}</td>
                <td style="font-size:11px;color:var(--text-muted)">${s.data_start} → ${s.data_end}</td>
            </tr>`;
        }).join('');
    }

    renderTable(stocks);

    // Sorting
    document.querySelectorAll('.leaderboard-table thead th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sort;
            const dir = (currentSort.key === key && currentSort.dir === 'desc') ? 'asc' : 'desc';
            currentSort = { key, dir };

            document.querySelectorAll('th').forEach(t => t.classList.remove('sorted-asc', 'sorted-desc'));
            th.classList.add('sorted-' + dir);

            const sorted = [...stocks].sort((a, b) => {
                let va = key === 'rank' ? stocks.indexOf(a) : a[key];
                let vb = key === 'rank' ? stocks.indexOf(b) : b[key];
                if (typeof va === 'string') return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
                if (isNaN(va)) va = -Infinity;
                if (isNaN(vb)) vb = -Infinity;
                return dir === 'asc' ? va - vb : vb - va;
            });
            renderTable(sorted);
        });
    });

    // Search
    document.getElementById('searchInput').addEventListener('input', e => {
        const q = e.target.value.toLowerCase().trim();
        if (!q) {
            renderTable(stocks);
            return;
        }
        const filtered = stocks.filter(s => s.symbol.toLowerCase().includes(q));
        renderTable(filtered);
    });

})();
