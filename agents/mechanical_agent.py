# agents/mechanical_agent.py
import torch
import json

class MechanicalAgent:
    """Mechanical/CAD expert agent with fine-tuned weights"""
    
    def __init__(self, model_wrapper):
        self.model = model_wrapper
        self.agent_type = "mechanical"
        
    def generate_latent_thoughts(self, context, latent_steps=20):
        """Generate latent thoughts about mechanical design"""
        
        prompt = f"""You are a mechanical engineer designing medical devices.
Based on the design context:
{context}

Generate mechanical design specifications including:
1. Enclosure dimensions and materials
2. Component placement and mounting
3. Ergonomics and user interface
4. Manufacturing considerations

Think step by step about the mechanical requirements."""
        
        # Generate latent thoughts (no text output yet)
        with torch.no_grad():
            thoughts = self.model.generate_latent(
                prompt=prompt,
                latent_steps=latent_steps,
                return_kv_cache=True
            )
        return thoughts
    
    def decode_to_files(self, kv_cache):
        """Decode latent thoughts to actual CAD files and specs"""
        
        prompt = """Based on the mechanical design thoughts, generate:
1. OpenSCAD code for the enclosure
2. Material specifications in JSON format
3. STEP file description
4. Assembly instructions

Output format: JSON with files as base64 strings."""
        
        output = self.model.decode_latent(
            kv_cache=kv_cache,
            prompt=prompt,
            max_tokens=2000
        )
        
        # Parse and structure the output
        return self._structure_output(output)
    
    def _structure_output(self, raw_output):
        """Structure the output into files"""
        # Implementation depends on your training format
        return {
            "enclosure.scad": "// OpenSCAD code...",
            "material_spec.json": json.dumps({
                "material": "ABS-M30i",
                "properties": {...}
            }),
            "assembly.step": "STEP file data..."
        }