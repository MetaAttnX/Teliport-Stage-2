# methods/baseline.py
"""
Baseline Method: Single LLM does everything alone
This is like having one engineer who knows mechanical, electronics, firmware, and bio
(Not realistic - no one can be expert in everything!)
"""

class BaselineMethod:
    """
    One model handles the entire medical device design
    Pros: Simple, easy to implement
    Cons: Model gets confused switching between domains, limited context
    """
    
    def __init__(self, model):
        self.model = model  # Single model
        
    def run(self, task):
        # One model does everything sequentially
        prompt = f"""
        Design a complete medical device:
        {task}
        
        Please provide:
        1. Mechanical design (CAD, materials)
        2. Electronics design (schematics, VHDL)
        3. Firmware code (C++)
        4. Bio-interface specs (safety standards)
        """
        
        # Model generates everything at once
        output = self.model.generate(prompt)
        return output