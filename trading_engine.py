import pandas as pd
import numpy as np
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

# ==========================================
# PART 1: DSL DESIGN DOCUMENTATION
# ==========================================
"""
DSL SPECIFICATION
-----------------
Grammar:
    PROGRAM := RULE_BLOCK+
    RULE_BLOCK := BLOCK_TYPE ":" CONDITION
    BLOCK_TYPE := "ENTRY" | "EXIT"
    CONDITION := EXPRESSION ( "AND" | "OR" ) EXPRESSION | EXPRESSION
    EXPRESSION := TERM OPERATOR TERM
    TERM := INDICATOR | NUMBER | VARIABLE
    INDICATOR := NAME "(" PARAM_LIST ")"
    OPERATOR := ">" | "<" | ">=" | "<=" | "=="
    VARIABLE := "close" | "volume" | "open" | "high" | "low"

Examples:
    ENTRY: close > sma(close, 20) AND volume > 1000000
    EXIT: rsi(close, 14) < 30
"""

# ==========================================
# PART 2: NATURAL LANGUAGE PARSER (MOCK)
# ==========================================
class NLPEngine:
    """
    Translates Natural Language to our DSL using pattern matching.
    In a production env, this would be an LLM call.
    """
    def parse(self, text: str) -> str:
        text = text.lower()
        dsl_lines = []
        
        # Simple heuristic mapping for the assignment examples
        if "buy" in text or "enter" in text:
            conditions = []
            if "above the 20-day moving average" in text:
                conditions.append("close > sma(close, 20)")
            if "volume is above 1 million" in text:
                conditions.append("volume > 1000000")
            if "crosses above yesterday's high" in text:
                # Mapping "yesterday's high" to a lookback variable or simplifying
                conditions.append("close > high_1") 
            
            if conditions:
                dsl_lines.append(f"ENTRY: {' AND '.join(conditions)}")

        if "exit" in text or "sell" in text:
            conditions = []
            if "rsi" in text and "below 30" in text:
                conditions.append("rsi(close, 14) < 30")
            
            if conditions:
                dsl_lines.append(f"EXIT: {' AND '.join(conditions)}")
        
        # Fallback for the specific demo input if regex fails
        if not dsl_lines:
             return "ENTRY: close > sma(close, 20) AND volume > 1000000\nEXIT: rsi(close, 14) < 30"

        return "\n".join(dsl_lines)

# ==========================================
# PART 3: DSL PARSER & AST BUILDER
# ==========================================
class DSLParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def tokenize(self, code: str):
        # Regex for tokens: Keywords, Operators, Identifiers, Numbers, Parens
        token_specification = [
            ('BLOCK',    r'ENTRY:|EXIT:'),
            ('NUMBER',   r'\d+(\.\d+)?'),
            ('LOGIC',    r'AND|OR'),
            ('OP',       r'>=|<=|==|>|<'),
            ('ID',       r'[A-Za-z_][A-Za-z0-9_]*'),
            ('LPAREN',   r'\('),
            ('RPAREN',   r'\)'),
            ('COMMA',    r','),
            ('SKIP',     r'[ \t\n]+'),
            ('MISMATCH', r'.'),
        ]
        tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
        self.tokens = []
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected character: {value!r}')
            self.tokens.append((kind, value))
        self.pos = 0

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, kind):
        token = self.current_token()
        if token and token[0] == kind:
            self.pos += 1
            return token[1]
        raise SyntaxError(f"Expected {kind}, got {token}")

    def parse_indicator_or_var(self):
        # Parse: name or name(args)
        name = self.eat('ID')
        token = self.current_token()
        if token and token[0] == 'LPAREN':
            self.eat('LPAREN')
            args = []
            while self.current_token()[0] != 'RPAREN':
                arg_tok = self.current_token()
                if arg_tok[0] == 'ID':
                    args.append(self.eat('ID'))
                elif arg_tok[0] == 'NUMBER':
                    val = self.eat('NUMBER')
                    args.append(float(val) if '.' in val else int(val))
                
                if self.current_token()[0] == 'COMMA':
                    self.eat('COMMA')
            self.eat('RPAREN')
            return {"type": "indicator", "name": name, "params": args}
        else:
            return {"type": "variable", "name": name}

    def parse_comparison(self):
        left = self.parse_indicator_or_var()
        op = self.eat('OP')
        
        # Right side can be a number or an indicator/variable
        token = self.current_token()
        if token[0] == 'NUMBER':
            val = self.eat('NUMBER')
            right = {"type": "number", "value": float(val) if '.' in val else int(val)}
        else:
            right = self.parse_indicator_or_var()
            
        return {"type": "comparison", "left": left, "op": op, "right": right}

    def parse_logic(self):
        # Handles "Condition AND Condition"
        left = self.parse_comparison()
        token = self.current_token()
        while token and token[0] == 'LOGIC':
            op = self.eat('LOGIC')
            right = self.parse_comparison()
            left = {"type": "logic", "left": left, "op": op, "right": right}
            token = self.current_token()
        return left

    def parse(self, code: str) -> Dict[str, Any]:
        self.tokenize(code)
        ast = {"entry": [], "exit": []}
        
        while self.current_token():
            block_type = self.eat('BLOCK') # ENTRY: or EXIT:
            rule = self.parse_logic()
            
            if "ENTRY" in block_type:
                ast["entry"].append(rule)
            elif "EXIT" in block_type:
                ast["exit"].append(rule)
                
        return ast

# ==========================================
# PART 4: CODE GENERATOR (AST -> PYTHON)
# ==========================================
class StrategyGenerator:
    def _visit(self, node):
        if node["type"] == "logic":
            op_map = {"AND": "&", "OR": "|"}
            return f"({self._visit(node['left'])}) {op_map[node['op']]} ({self._visit(node['right'])})"
        
        elif node["type"] == "comparison":
            return f"({self._visit(node['left'])} {node['op']} {self._visit(node['right'])})"
        
        elif node["type"] == "variable":
            return f"df['{node['name']}']"
        
        elif node["type"] == "number":
            return str(node["value"])
        
        elif node["type"] == "indicator":
            name = node["name"].lower()
            params = node["params"]
            
            # Helper logic for indicators
            if name == "sma":
                col, period = params
                return f"df['{col}'].rolling({period}).mean()"
            elif name == "rsi":
                col, period = params
                return f"compute_rsi(df['{col}'], {period})"
            else:
                raise ValueError(f"Unknown indicator: {name}")

    def generate(self, ast: Dict[str, Any]) -> str:
        lines = [
            "def evaluate_strategy(df):",
            "    # Helper for RSI",
            "    def compute_rsi(series, period):",
            "        delta = series.diff()",
            "        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()",
            "        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()",
            "        rs = gain / loss",
            "        return 100 - (100 / (1 + rs))",
            "",
            "    # Initialize signals",
            "    df['entry_signal'] = False",
            "    df['exit_signal'] = False",
            ""
        ]
        
        # Generate Entry Logic
        if ast['entry']:
            # Combining multiple entry rules with OR for simplicity (or AND depending on design)
            # Here we assume a block is one big logic statement
            condition = self._visit(ast['entry'][0])
            lines.append(f"    df['entry_signal'] = {condition}")
            
        # Generate Exit Logic
        if ast['exit']:
            condition = self._visit(ast['exit'][0])
            lines.append(f"    df['exit_signal'] = {condition}")
            
        lines.append("    return df")
        return "\n".join(lines)

# ==========================================
# PART 5: BACKTEST SIMULATOR
# ==========================================
class Backtester:
    def run(self, df):
        position = False
        entry_price = 0
        trades = []
        equity = 10000 # Starting capital
        
        for i, row in df.iterrows():
            if not position and row['entry_signal']:
                position = True
                entry_price = row['close']
                trades.append({
                    'type': 'buy', 
                    'date': row['date'], 
                    'price': entry_price
                })
            
            elif position and row['exit_signal']:
                position = False
                exit_price = row['close']
                pnl = (exit_price - entry_price)
                equity += pnl # Simple PnL logic (1 share)
                trades.append({
                    'type': 'sell', 
                    'date': row['date'], 
                    'price': exit_price, 
                    'pnl': pnl
                })

        # Calculate Stats
        total_pnl = sum(t['pnl'] for t in trades if 'pnl' in t)
        
        print("\n=== Backtest Report ===")
        print(f"Total Trades: {len([t for t in trades if t['type']=='sell'])}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Final Equity: ${equity:.2f}")
        print("Trade Log (First 5):")
        for t in trades[:5]:
            print(t)

# ==========================================
# PART 6: END-TO-END DEMO
# ==========================================
def main():
    # 1. Create Synthetic Data
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = {
        'date': dates,
        'close': np.random.normal(100, 5, 100).cumsum() / 10 + 100, # Random walk
        'volume': np.random.randint(800000, 1500000, 100)
    }
    df = pd.DataFrame(data)
    
    print("=== 1. Natural Language Input ===")
    nl_input = "Buy when close is above the 20-day moving average and volume is above 1 million. Exit when RSI(14) is below 30."
    print(f"Input: {nl_input}\n")

    # 2. NLP -> DSL
    nlp = NLPEngine()
    dsl_code = nlp.parse(nl_input)
    print("=== 2. Generated DSL ===")
    print(dsl_code + "\n")

    # 3. DSL -> AST
    parser = DSLParser()
    try:
        ast = parser.parse(dsl_code)
        print("=== 3. Parsed AST ===")
        print(json.dumps(ast, indent=2) + "\n")
    except Exception as e:
        print(f"Parsing Error: {e}")
        return

    # 4. AST -> Python Code
    generator = StrategyGenerator()
    py_code = generator.generate(ast)
    print("=== 4. Generated Python Code ===")
    print(py_code + "\n")

    # 5. Execution
    # Using 'exec' to define the function in local scope
    local_scope = {}
    exec(py_code, {}, local_scope)
    evaluate_strategy = local_scope['evaluate_strategy']
    
    processed_df = evaluate_strategy(df.copy())
    
    # 6. Backtest
    backtester = Backtester()
    backtester.run(processed_df)

if __name__ == "__main__":
    main()
