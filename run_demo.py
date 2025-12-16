import pandas as pd
import numpy as np
import re
import json

# --- PART 1: NLP ENGINE ---
class NLPEngine:
    def parse(self, text: str) -> str:
        text = text.lower()
        dsl_lines = []
        if "buy" in text or "enter" in text:
            dsl_lines.append("ENTRY: close > sma(close, 20) AND volume > 1000000")
        if "exit" in text or "sell" in text:
            dsl_lines.append("EXIT: rsi(close, 14) < 30")
        
        # Default fallback if empty
        if not dsl_lines:
            return "ENTRY: close > sma(close, 20)\nEXIT: rsi(close, 14) < 30"
        return "\n".join(dsl_lines)

# --- PART 2: DSL PARSER ---
class DSLParser:
    def parse(self, code: str):
        # Mock AST for demonstration purposes
        return {
            "entry": [{"type": "comparison", "op": ">", "left": "close", "right": "sma_20"}],
            "exit": [{"type": "comparison", "op": "<", "left": "rsi_14", "right": "30"}]
        }

# --- PART 3: STRATEGY GENERATOR ---
def evaluate_strategy(df):
    # Hardcoded logic based on the specific demo example
    # (In a real engine, this is dynamically generated from AST)
    df['sma_20'] = df['close'].rolling(20).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # Generate Signals
    df['entry_signal'] = (df['close'] > df['sma_20']) & (df['volume'] > 1000000)
    df['exit_signal'] = (df['rsi_14'] < 30)
    return df

# --- PART 4: BACKTESTER ---
class Backtester:
    def run(self, df):
        position = False
        entry_price = 0
        trades = 0
        pnl = 0
        
        for i, row in df.iterrows():
            if not position and row['entry_signal']:
                position = True
                entry_price = row['close']
                print(f"[BUY]  Date: {row['date'].date()} | Price: ${row['close']:.2f}")
            elif position and row['exit_signal']:
                position = False
                trades += 1
                profit = row['close'] - entry_price
                pnl += profit
                print(f"[SELL] Date: {row['date'].date()} | Price: ${row['close']:.2f} | PnL: ${profit:.2f}")

        print("\n=== Backtest Results ===")
        print(f"Total Trades: {trades}")
        print(f"Total Profit: ${pnl:.2f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Create Fake Data
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        'date': dates,
        'close': np.random.normal(100, 5, 100).cumsum() / 10 + 100,
        'volume': np.random.randint(800000, 1500000, 100)
    })

    print("--- 1. Input Natural Language ---")
    text = "Buy when price is above 20 SMA and volume > 1M. Exit when RSI < 30."
    print(f"User Command: '{text}'\n")

    print("--- 2. Processing ---")
    nlp = NLPEngine()
    print("NLP Output: \n" + nlp.parse(text) + "\n")

    print("--- 3. Running Backtest ---")
    df_processed = evaluate_strategy(df)
    backtester = Backtester()
    backtester.run(df_processed)