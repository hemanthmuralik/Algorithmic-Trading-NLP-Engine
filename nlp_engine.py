import re
import json

class NLPEngine:
    """
    Translates Natural Language trading rules into structured DSL.
    """
    
    def parse_nl(self, text):
        """
        Main pipeline: NL -> Structured JSON -> DSL String
        """
        text = text.lower()
        
        # 1. Identify Action
        action = "BUY"
        if "exit" in text or "sell" in text:
            action = "EXIT"
            
        # 2. Extract Logic (Heuristic / Regex approach)
        conditions = []
        
        # Pattern: "price above sma 20"
        if "above" in text or "greater" in text:
            operator = ">"
        elif "below" in text or "less" in text:
            operator = "<"
        else:
            operator = ">" # Default assumption
            
        # Logic for SMA (Simple Moving Average)
        # Matches: "20 day moving average", "sma 20", "20-day sma"
        sma_match = re.search(r'(\d+)[ -]?(day|period)?\s?(moving average|sma)', text)
        if sma_match and ("price" in text or "close" in text):
            period = sma_match.group(1)
            conditions.append(f"close {operator} sma_{period}")
            
        # Logic for Volume
        # Matches: "volume above 1000000"
        vol_match = re.search(r'volume\s?(is)?\s?(above|greater than)?\s?(\d+)[km]?', text)
        if vol_match:
            # Simple extractor for the number
            numbers = re.findall(r'\d+', text.replace(',', ''))
            # Heuristic: the largest number is likely the volume threshold
            vol_threshold = max([int(n) for n in numbers if int(n) > 200]) 
            conditions.append(f"volume > {vol_threshold}")

        # Logic for RSI
        # Matches: "RSI(14) is below 30"
        rsi_match = re.search(r'rsi', text)
        if rsi_match:
             # Find numbers near RSI
             val_match = re.search(r'(below|less than)\s?(\d+)', text)
             if val_match:
                 threshold = val_match.group(2)
                 conditions.append(f"rsi_14 < {threshold}")

        # 3. Construct DSL
        dsl_rule = " AND ".join(conditions)
        if not dsl_rule:
            dsl_rule = "close > 0" # Fallback safe rule
            
        return action, dsl_rule

    def get_structured_json(self, action, dsl_rule):
        """
        Returns the JSON representation required by Part 1 of the assignment.
        """
        # Simple parser to split back into JSON for the 'deliverable' requirement
        rules = []
        parts = dsl_rule.split(" AND ")
        for part in parts:
            if ">" in part:
                left, right = part.split(">")
                op = ">"
            elif "<" in part:
                left, right = part.split("<")
                op = "<"
            else:
                continue
            
            rules.append({
                "left": left.strip(),
                "operator": op,
                "right": right.strip()
            })
            
        return json.dumps({action.lower(): rules}, indent=2)
