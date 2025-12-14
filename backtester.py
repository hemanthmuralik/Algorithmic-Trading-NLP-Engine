import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital=10000):
        self.capital = initial_capital
        self.log = []

    def load_data(self):
        """Generates synthetic data as allowed by the assignment."""
        days = 200
        dates = pd.date_range(start='2023-01-01', periods=days)
        
        # Create Random Walk Price
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, days)
        price = 100 * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            'date': dates,
            'close': price,
            'volume': np.random.randint(500000, 2000000, days)
        })
        
        # Pre-calculate Indicators for the DSL to use
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi_14'] = self.calculate_rsi(df['close'])
        df.fillna(0, inplace=True)
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run(self, df, entry_mask, exit_mask):
        """
        Simulates the trades.
        entry_mask: Boolean Series (Buy signals)
        exit_mask: Boolean Series (Sell signals)
        """
        cash = self.capital
        position = 0
        entry_price = 0
        
        df['signal'] = 0
        # 1 = Buy, -1 = Sell
        
        for i in range(len(df)):
            if position == 0 and entry_mask.iloc[i]:
                # BUY
                price = df['close'].iloc[i]
                position = cash / price
                cash = 0
                entry_price = price
                self.log.append({
                    "type": "ENTRY",
                    "date": str(df['date'].iloc[i].date()),
                    "price": round(price, 2)
                })
                
            elif position > 0 and exit_mask.iloc[i]:
                # SELL
                price = df['close'].iloc[i]
                cash = position * price
                position = 0
                
                # Calculate Trade Metrics
                pnl = (price - entry_price) / entry_price
                self.log.append({
                    "type": "EXIT",
                    "date": str(df['date'].iloc[i].date()),
                    "price": round(price, 2),
                    "pnl": f"{pnl*100:.2f}%"
                })

        # Final Portfolio Value
        final_value = cash if position == 0 else position * df['close'].iloc[-1]
        
        # Calculate Drawdown
        df['total_value'] = self.capital # Simplified for this demo
        # (In a real engine, we'd track daily value for accurate DD)
        
        return final_value, self.log
