# 📈 EMAScalpBackTest

> A modular Python framework for quantitative trading research, strategy development, and backtesting using Binance historical market data.

![Status](https://img.shields.io/badge/Status-Active%20Development-blue)
![Python](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

---

## 📖 Overview

**EMAScalpBackTest** is an open-source Python project focused on building a **professional-grade quantitative trading framework**.

The project started with an EMA-based scalping strategy but is being designed as a reusable trading research platform capable of supporting:

- Multiple trading strategies
- Historical backtesting
- Risk management
- Performance analytics
- Strategy optimization
- Machine Learning models
- Future live trading integrations

The emphasis is on **clean architecture**, **modularity**, and **realistic trading simulations**.

---

## ✨ Current Features

- ✅ Binance historical market data downloader
- ✅ Incremental data caching
- ✅ Modular project architecture
- ✅ Configuration-based project setup
- ✅ Unit test structure
- 🚧 EMA crossover strategy (In Progress)
- 🚧 Event-driven backtesting engine (In Progress)

---

## 🚀 Planned Features

### 📊 Technical Indicators

- EMA
- SMA
- RSI
- ATR
- MACD
- Bollinger Bands
- VWAP
- SuperTrend
- ADX
- Custom Indicators

---

### 📈 Strategy Engine

- EMA Crossover
- Candlestick Pattern Recognition
- Breakout Strategies
- Mean Reversion
- Trend Following
- Multi-Timeframe Strategies
- Custom Strategy Interface

---

### 💰 Backtesting Engine

- Event-driven execution
- Long & Short positions
- Trading fees
- Slippage simulation
- Position sizing
- Equity curve generation
- Portfolio tracking
- Drawdown analysis

---

### 📉 Performance Metrics

- Net Profit
- Win Rate
- Profit Factor
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- CAGR
- Average Trade
- Risk / Reward
- Expectancy

---

### 🤖 Machine Learning (Future)

- Feature Engineering
- Random Forest
- XGBoost
- LightGBM
- LSTM Networks
- Transformer Models
- Reinforcement Learning

---

### 🌐 Live Trading (Future)

- Binance API Integration
- Paper Trading
- Live Order Execution
- Risk Controls
- Portfolio Monitoring

---

# 🏗 Project Architecture

```text
                   Historical Data
                          │
                          ▼
                 Data Download & Cache
                          │
                          ▼
                  Indicator Engine
                          │
                          ▼
                  Strategy Engine
                          │
                          ▼
                 Event Backtester
                          │
                          ▼
                 Performance Report
                          │
                          ▼
                 Strategy Optimization
                          │
                          ▼
                  Machine Learning
                          │
                          ▼
                    Live Trading
```

---

# 📂 Project Structure

```text
EMAScalpBackTest
│
├── data/                  # Historical market data
├── reports/               # Generated reports
├── notebooks/             # Research notebooks
├── src/
│   ├── data_loader.py
│   ├── indicators.py
│   ├── strategy.py
│   ├── backtester.py
│   ├── report.py
│   ├── patterns.py
│   ├── optimizer.py
│   ├── plotting.py
│   └── utils.py
│
├── tests/
│
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/abdulwaheedandroid/EMAScalpBackTest.git
```

Move into the project

```bash
cd EMAScalpBackTest
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```

---

# 🧪 Running Tests

```bash
pytest
```

---

# 🛣 Development Roadmap

## Phase 1 — Foundation

- [x] Project setup
- [x] GitHub repository
- [x] Historical data downloader
- [x] Incremental caching
- [ ] Technical indicator engine
- [ ] EMA crossover strategy
- [ ] Event-driven backtester

---

## Phase 2 — Trading Framework

- [ ] Performance reporting
- [ ] Risk management
- [ ] Position sizing
- [ ] Candlestick recognition
- [ ] Strategy optimization

---

## Phase 3 — AI & Quant Research

- [ ] Feature engineering
- [ ] Machine Learning models
- [ ] Walk-forward optimization
- [ ] Portfolio optimization

---

## Phase 4 — Production

- [ ] Paper trading
- [ ] Binance live trading
- [ ] Dashboard
- [ ] Docker support
- [ ] CI/CD pipeline

---

# 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Binance API
- Pytest
- Git
- GitHub

---

# 🤝 Contributing

Contributions, ideas, bug reports, and feature requests are welcome.

If you'd like to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

# ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.

It does **not** provide financial advice.

Trading cryptocurrencies involves substantial financial risk. Historical backtesting results **do not guarantee future performance**.

Always perform your own research before trading.

---

# 👨‍💻 Author

**Abdul Waheed**

Senior Android Developer • Python Learner • Algorithmic Trading Enthusiast

GitHub: https://github.com/abdulwaheedandroid

---

# ⭐ Support the Project

If you find this repository useful or interesting, consider giving it a **Star** ⭐ on GitHub.

Your support helps motivate future development and improvements.