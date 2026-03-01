# agents/bio_agent.py
class BioAgent:
    """Bio-interface expert agent"""
    
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.agent_type = "bio"
        
    def generate_latent_thoughts(self, context, latent_steps=15):
        prompt = f"""You are a biomedical engineer specializing in medical device safety.
Based on the design context:
{context}

Specify bio-interface requirements:
1. Patient connection safety (IEC 60601-1)
2. Biocompatible materials (ISO 10993)
3. Sterilization methods
4. Clinical workflow integration

Think step by step about safety and compliance."""
        
        with torch.no_grad():
            thoughts = self.model.generate_latent(
                prompt=prompt,
                latent_steps=latent_steps,
                return_kv_cache=True
            )
        return thoughts
    
    def decode_to_files(self, kv_cache):
        """Decode to safety specifications"""
        return {
            "safety_spec.json": json.dumps({
                "standards": ["IEC60601-1", "ISO10993"],
                "leakage_current": "<10μA",
                "isolation": "4000V"
            }),
            "biocompatibility.txt": "Material certifications...",
            "clinical_workflow.md": "# Clinical use guidelines..."
        }