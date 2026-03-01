# methods/text_mas.py
"""
Text-Based Multi-Agent System (TextMAS)
Agents talk to each other using NATURAL LANGUAGE
This is like having 4 engineers who email each other
"""

class TextMASMethod:
    """
    Multiple specialized models communicating via TEXT
    Pros: Each model is expert in its domain
    Cons: Information lost in translation, tokenizer mismatches, verbose
    
    THIS IS WHAT YOUR COLLEAGUE WAS WORRIED ABOUT!
    """
    
    def __init__(self, agents):
        self.agents = agents  # Dict of 4 specialized models
        
    def run(self, task):
        # Agent 1: Mechanical thinks
        mech_output = self.agents["mechanical"].generate(
            f"Design mechanical parts for: {task}"
        )
        
        # Convert to text (LOSSY!)
        mech_text = mech_output.text
        
        # Agent 2: Electronics reads mechanical's text
        elec_output = self.agents["electronics"].generate(
            f"Design electronics for this mechanical design: {mech_text}\n\nTask: {task}"
        )
        
        # More text conversion...
        elec_text = elec_output.text
        
        # Agent 3: Firmware reads electronics' text
        fw_output = self.agents["firmware"].generate(
            f"Write firmware for these electronics: {elec_text}"
        )
        
        # Agent 4: Bio reads everything
        bio_output = self.agents["bio"].generate(
            f"Add safety specs for: {fw_output.text}"
        )
        
        # PROBLEMS:
        # 1. Each text conversion loses information
        # 2. Different tokenizers mis-interpret text
        # 3. Very verbose (tons of tokens)
        # 4. Errors compound