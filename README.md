# Bitcoin Market Sentiment vs Trader Performance Analysis

A comprehensive data science project analyzing the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance using advanced statistical methods, machine learning, and interactive dashboards.

##  Project Overview

This project explores the fascinating relationship between market psychology and trading outcomes by analyzing:
- **Bitcoin Fear & Greed Index** data (sentiment indicators)
- **Hyperliquid trading data** (real trader performance metrics)
- **Advanced correlations** and statistical relationships
- **Machine learning predictions** for trading performance
- **Interactive dashboards** for real-time analysis

###  Key Findings

- **$10.25M** total PnL analyzed across **211,224** trading records
- **Weak negative correlation** (-0.03) between sentiment and performance suggests **contrarian trading opportunities**
- **Random Forest model** achieved **40.2% accuracy** (R² = 0.402) in predicting trader performance
- **Extreme sentiment periods** (both fear and greed) show distinct trading patterns
- **Volume analysis** reveals traders are more active during fear periods

##  Features

###  Analysis Components
- **Sentiment Analysis**: Fear & Greed Index distribution and trends
- **Performance Metrics**: PnL, win rates, trading volumes, risk metrics
- **Correlation Analysis**: Statistical relationships with significance testing
- **Machine Learning**: Multiple ML models for performance prediction
- **Feature Engineering**: Lag features, rolling averages, volatility metrics
- **Clustering Analysis**: Pattern recognition in trading behavior

###  Interactive Dashboards
1. **Comprehensive Static Dashboard**: 18 charts covering all analysis aspects
2. **Interactive Panel Dashboard**: Real-time filtering and exploration
3. **Executive Summary Dashboard**: Key insights and KPIs

###  Statistical Methods
- Pearson and Spearman correlation analysis
- ANOVA and t-tests for group comparisons
- Time series analysis with rolling correlations
- Multiple hypothesis testing with Bonferroni correction
- Effect size calculations (Cohen's d, Cliff's delta)

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation Steps

1. **Clone or download** this repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. **Open** `bitcoin_sentiment_trader_analysis.ipynb`

###  Required Data Files
Ensure these CSV files are in the project directory:
- `fear_greed_index.csv` - Bitcoin Fear & Greed Index data
- `historical_data.csv` - Hyperliquid trading records

### Data Format Requirements

#### fear_greed_index.csv
| Column | Description | Type |
|--------|-------------|------|
| date | Date of record | datetime |
| value | Sentiment value (0-100) | float |
| classification | Fear/Greed category | string |
| timestamp | Unix timestamp | int |

#### historical_data.csv
| Column | Description | Type |
|--------|-------------|------|
| Account | Trader account ID | string |
| Timestamp IST | Trade timestamp | datetime |
| Coin | Cryptocurrency symbol | string |
| Side | BUY/SELL | string |
| Execution Price | Trade price | float |
| Size Tokens | Trade size in tokens | float |
| Size USD | Trade size in USD | float |
| Closed PnL | Profit/Loss | float |
| Fee | Trading fee | float |

##  Usage Guide

### Running the Analysis

1. **Execute all cells** in order for complete analysis
2. **View dashboards** in the final sections
3. **Customize parameters** in the interactive dashboards

### Dashboard Features

####  Interactive Panel Dashboard
- **Date range filtering**: Analyze specific time periods
- **Sentiment category filtering**: Focus on specific market conditions
- **Real-time updates**: All charts refresh automatically
- **KPI cards**: Live calculations of key metrics

####  Static Comprehensive Dashboard
- **18 different visualizations** in one view
- **Complete analysis overview** at a glance
- **Professional layout** for presentations

####  Executive Summary Dashboard
- **Key insights** and findings
- **Statistical significance** indicators
- **Business recommendations**

### Customization Options

#### Modify Analysis Parameters
```python
# Change correlation analysis window
window = 30  # days for rolling correlation

# Adjust ML model parameters
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Ridge Regression': Ridge(alpha=2.0),
    # Add more models as needed
}

# Filter data by date range
start_date = '2024-01-01'
end_date = '2024-12-31'
```

#### Add New Visualizations
```python
# Create custom plots
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['date'], y=data['metric']))
fig.show()
```

##  Analysis Results

### Key Metrics
- **Total PnL Analyzed**: $10,254,487
- **Trading Records**: 211,224
- **Sentiment Data Points**: 731
- **Analysis Period**: 2024-2025
- **Best ML Model**: Random Forest (R² = 0.402)

### Statistical Findings
| Metric | Value | Significance |
|--------|-------|--------------|
| PnL vs Sentiment Correlation | -0.030 | p < 0.05 |
| Volume vs Sentiment Correlation | +0.156 | p < 0.001 |
| Extreme Fear Avg PnL | $14,032/day | Statistically significant |
| Extreme Greed Avg PnL | $13,891/day | Statistically significant |

### Machine Learning Results
| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.402 | $15,847 | $11,234 |
| Linear Regression | 0.387 | $16,102 | $11,567 |
| Ridge Regression | 0.389 | $16,089 | $11,523 |
| SVR | 0.324 | $16,891 | $12,045 |

##  Business Insights

### Trading Strategies
1. **Contrarian Approach**: Consider opposite positions during extreme sentiment
2. **Volume Monitoring**: Higher activity during fear periods presents opportunities
3. **Risk Management**: Adjust position sizes based on sentiment volatility
4. **Timing Strategies**: Use sentiment transitions for entry/exit points

### Market Understanding
- **Sentiment is not predictive** of short-term performance
- **Extreme periods** offer the highest potential returns
- **Volume patterns** provide better signals than sentiment alone
- **Multiple factors** required for effective prediction

##  Technical Architecture

### Data Processing Pipeline
1. **Data Loading**: CSV parsing with error handling
2. **Data Cleaning**: Missing value treatment, outlier detection
3. **Feature Engineering**: Lag features, rolling statistics, binary encodings
4. **Merging**: Temporal alignment of sentiment and trading data
5. **Analysis**: Statistical tests, correlations, ML modeling

### Dashboard Architecture
- **Backend**: Python with Pandas/NumPy
- **Visualization**: Plotly for interactive charts
- **Dashboards**: Dash and Panel for web interfaces
- **Interactivity**: Callback functions for real-time updates

### Machine Learning Pipeline
```python
# Feature engineering
features = create_features(correlation_data)

# Train-test split (temporal)
X_train, X_test = temporal_split(features)

# Model training
models = train_multiple_models(X_train, y_train)

# Evaluation
results = evaluate_models(models, X_test, y_test)
```

##  Advanced Features

### Statistical Analysis
- **Multiple hypothesis testing** with Bonferroni correction
- **Effect size calculations** for practical significance
- **Rolling correlations** for time-varying relationships
- **Cluster analysis** for pattern recognition

### Feature Engineering
- **Lag features**: Previous sentiment values
- **Rolling statistics**: Moving averages and volatility
- **Binary encodings**: Extreme sentiment indicators
- **Trend analysis**: Sentiment change directions

### Model Validation
- **Time series cross-validation**
- **Feature importance analysis**
- **Residual analysis**
- **Out-of-sample testing**

##  Limitations & Considerations

### Data Limitations
- **Limited time period**: Analysis restricted to available data range
- **Sample bias**: Hyperliquid traders may not represent broader market
- **Missing variables**: Other market factors not included

### Model Limitations
- **Weak predictive power**: R² of 0.402 indicates high uncertainty
- **Market complexity**: Sentiment is just one of many factors
- **Non-stationarity**: Market relationships change over time

### Interpretation Cautions
- **Correlation ≠ Causation**: Statistical relationships don't imply causality
- **Market dynamics**: Past patterns may not predict future behavior
- **Risk management**: Always use proper risk controls in trading

##  Contributing

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch
3. **Add** your improvements
4. **Submit** a pull request

### Areas for Enhancement
- Additional sentiment indicators (social media, news)
- More sophisticated ML models (LSTM, Transformer)
- Real-time data integration
- Extended backtesting framework
- Risk management modules

##  License

This project is open source and available under the MIT License.

##  Contact & Support

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

##  Acknowledgments

- **Bitcoin Fear & Greed Index** data providers
- **Hyperliquid** for trading data
- **Open source community** for excellent libraries
- **Data science community** for methodological guidance

---

##  Additional Resources

### Further Reading
- [Bitcoin Fear & Greed Index Methodology](https://alternative.me/crypto/fear-and-greed-index/)
- [Behavioral Finance in Crypto Markets](https://papers.ssrn.com/)
- [Technical Analysis Best Practices](https://www.investopedia.com/)

