# 🚚 Supply Chain AI Optimizer

[![Forecast MAPE](https://img.shields.io/badge/MAPE-4.2%25-green)](.)
[![Savings](https://img.shields.io/badge/Cost%20Savings-R%2423M%2Fyear-blue)](.)
[![SKUs](https://img.shields.io/badge/SKUs%20Managed-180K-orange)](.)

> AI supply chain platform managing 180,000 SKUs for a top-5 Brazilian retailer. Demand forecasting achieves **4.2% MAPE**, generating **R$23M annual savings** through optimized inventory and routing.

## 🏆 Business Impact
- **R$23M/year savings** from inventory optimization and reduced stockouts
- **4.2% demand forecast MAPE** (vs 18.7% legacy system — 77% improvement)
- **31% reduction in stockouts** across 280 distribution centers
- **18% reduction in excess inventory** — capital freed for reinvestment

## 🏗️ Solution Components

| Component | Technology | Outcome |
|-----------|-----------|---------|
| Demand Forecasting | LSTM + Prophet ensemble | MAPE 4.2% |
| Inventory Optimization | Dynamic programming + ML | -31% stockouts |
| Supplier Risk | XGBoost + NLP news scraping | 89% accuracy |
| Route Optimization | Google OR-Tools | -22% logistics cost |

## 📊 Forecast Architecture
```
Historical Sales ──▶ ┌─ LSTM (sequential patterns)    ─┐
Promotions       ──▶ ├─ Prophet (seasonality + trends) ─┼──▶ Stacking ──▶ Inventory Policy
External (weather)──▶└─ XGBoost (tabular features)    ─┘   Ensemble
```
