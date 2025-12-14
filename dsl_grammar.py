import ast
import pandas as pd

class DSLParser:
    """
    Parses the Trading DSL into an Abstract Syntax Tree (AST) 
    and generates executable Python code for Pandas.
    """
    
    def parse_to_ast(self, dsl_code):
        """
        Parses a DSL string (e.g., "close > sma(close, 20)") into a Python AST.
        """
        # 1. Pre-processing: Ensure DSL operators map to Python syntax
        # The DSL is designed to be Python-compatible for this prototype.
        clean_code = dsl_code.replace("AND", "and").replace("OR", "or")
        
        try:
            # 2. Build AST using Python's native parser
            tree = ast.parse(clean_code, mode='eval')
            return tree
        except SyntaxError as e:
            raise ValueError(f"DSL Syntax Error: {e}")

    def generate_pandas_logic(self, dsl_code, df):
        """
        Converts DSL logic into a Pandas Boolean Series (The 'Code Generator').
        """
        try:
            # We map DSL variables directly to DataFrame columns using df.eval()
            # This serves as our "Compiler" to vectorized machine code.
            
            # 1. Lowercase for consistency
            expr = dsl_code.lower()
            
            # 2. Handle 'AND'/'OR' for pandas.eval
            expr = expr.replace("and", "&").replace("or", "|")
            
            # 3. Dynamic Indicator Resolution (Pre-calculation hook)
            # In a full compiler, we would walk the AST to find 'sma(20)'. 
            # Here, we assume indicators are pre-calculated in the DF (see backtester.py)
            # or we map simple function calls.
            
            # Execute the logic
            # This effectively runs the "Generated Code"
            return df.eval(expr)
            
        except Exception as e:
            print(f"Code Generation Error for '{dsl_code}': {e}")
            return pd.Series([False] * len(df), index=df.index)
