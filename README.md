# Amazon FBA Dynamic Reorder Point A/B Testing

## 🎯 Executive Summary

This project implements a rigorous A/B test comparing **Fixed Reorder Points** (traditional approach) vs **Dynamic Reorder Points** (adaptive approach) for Amazon FBA inventory management.

### Key Findings

✅ **Fill Rate: +11.8% improvement** (p=0.0199, statistically significant)
✅ **Stockouts: -65.9% reduction** (p=0.0204, statistically significant)  
⚠️ **Average Inventory: +20.6% increase** (p=0.0815, not significant)

**Recommendation**: Implement Dynamic ROP for improved service levels at the cost of slightly higher safety stock.

---

## 📊 Methodology

### Test Design
- **Duration**: 90 days
- **Sample Size**: 20 SKUs (9 control, 11 treatment)
- **Randomization**: Stratified by ABC classification
- **Significance Level**: α = 0.05

### Control Group: Fixed ROP
```
ROP = (Average Daily Demand × Average Lead Time) + Safety Stock
Safety Stock = Z-score × σ_demand × √Lead_Time
```

### Treatment Group: Dynamic ROP
```
ROP = (Weighted Moving Avg Demand × Forecasted Lead Time) + Dynamic Safety Stock
WMA = 50% (30-day) + 30% (60-day) + 20% (90-day)
```

---

## 📈 Results

### Primary Metrics

| Metric | Fixed ROP | Dynamic ROP | Change | p-value | Significant? |
|--------|-----------|-------------|--------|---------|--------------|
| **Fill Rate (%)** | 85.2% | 95.2% | **+11.8%** | 0.0199 | ✅ YES |
| **Avg Inventory** | 11,077 | 13,358 | +20.6% | 0.0815 | ❌ NO |
| **Stockouts** | 14.1 | 4.8 | **-65.9%** | 0.0204 | ✅ YES |

### Statistical Significance
- **Cohen's d (Fill Rate)**: 1.114 (Large effect size)
- **95% Confidence Interval**: [1.89, 18.22]
- **Power Analysis**: Sufficient power to detect 10% difference

---

## 💰 Business Impact

While ROI is negative in this small sample due to increased inventory:
- **Stockout Cost Savings**: $11,100/year (fewer emergency orders)
- **Service Level Improvement**: Critical for Prime customers
- **Trade-off**: Higher safety stock for better reliability

**Note**: With larger scale (1,000+ SKUs), inventory optimization algorithms would reduce excess stock while maintaining service levels.

---

## 🚀 Technologies Used

- **Python 3.10**: Core programming language
- **pandas, numpy**: Data manipulation and analysis
- **scipy, statsmodels**: Statistical testing
- **matplotlib, seaborn**: Data visualization
- **PyYAML**: Configuration management

---

## 📁 Project Structure
```
amazon-fba-reorder-point-ab-test/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned data
│   └── results/                # Test outputs
├── src/
│   ├── data_preprocessing.py   # Data cleaning
│   ├── rop_calculations.py     # ROP formulas
│   ├── simulation_engine.py    # A/B test simulation
│   ├── statistical_analysis.py # Hypothesis testing
│   └── visualization.py        # Charts
├── results/figures/            # Generated visualizations
└── README.md
```

---

## 🏃 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python -m src.data_preprocessing

# 3. Run simulation
python -m src.simulation_engine

# 4. Analyze results
python -m src.statistical_analysis

# 5. Generate visualizations
python -m src.visualization
```

---

## 📊 Visualizations

See `results/figures/` for:
1. **Comparison Dashboard**: Side-by-side metric comparison
2. **Distribution Plots**: Statistical distributions
3. **Scatter Analysis**: Fill rate vs inventory relationship
4. **Metrics Summary Table**: Complete statistical results

---

## 🔬 Key Learnings

1. **Trade-offs are Real**: Higher service levels often require more inventory
2. **Small Sample Limitations**: 20 SKUs limits statistical power
3. **Context Matters**: Amazon FBA prioritizes fill rate over inventory costs
4. **Statistical Rigor**: Proper hypothesis testing validates business decisions

---

## 📚 Data Source

- **Dataset**: Retail Store Inventory Forecasting Dataset (Kaggle)
- **Records**: 73,100 transactions
- **Transformation**: Rebranded with Amazon FBA context (fulfillment centers, Prime eligibility)

---

## 👤 Author

Hetvi Soni 
Data Analyst