# 📈 NLP-Driven Algorithmic Trading Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Prototype-orange)

A financial trading system that parses **Natural Language strategies** (e.g., *"Buy if RSI < 30"*) into executable Python code using a custom **Domain Specific Language (DSL)**.

This project bridges the gap between non-technical users and complex algorithmic trading, enabling automated backtesting of plain-English strategies without writing code.

## 🚀 Features
* **Natural Language Parsing:** Converts English commands into structured trading rules.
* **Custom DSL & AST:** Uses a Domain Specific Language (DSL) to map logic to Python functions.
* **Automated Backtesting:** Simulations run on historical data to calculate PnL, Max Drawdown, and Trade Logs.
* **Zero-Lag Execution:** optimized for efficient parsing and strategy evaluation.

## 🛠️ Tech Stack
* **Core:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Architecture:** Abstract Syntax Trees (AST), Recursive Descent Parser

## ⚡ Quick Start
Clone the repository and run the demo script:

```bash
# 1. Clone the repo
git clone [https://github.com/hemanthmuralik/Algorithmic-Trading-NLP-Engine.git](https://github.com/hemanthmuralik/Algorithmic-Trading-NLP-Engine.git)
cd Algorithmic-Trading-NLP-Engine

# 2. Install requirements
pip install pandas numpy

# 3. Run the engine
python run_demo.py
