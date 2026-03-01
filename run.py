# run.py - Modified for medical device design
import argparse
import json
import os
import torch
from methods.latent_mas import LatentMASMethod
from models import ModelWrapper
from agents import (
    MechanicalAgent, 
    ElectronicsAgent, 
    FirmwareAgent, 
    BioAgent
)

def load_trained_agent(agent_type, checkpoint_path, device):
    """Load your fine-tuned weights"""
    print(f"Loading {agent_type} expert from {checkpoint_path}")
    
    # Load base model with your fine-tuned weights
    model = ModelWrapper(
        model_name=checkpoint_path,  # Path to your saved weights
        device=device,
        use_vllm=False  # Use HF for latent space
    )
    
    # Wrap with agent-specific logic
    if agent_type == "mechanical":
        return MechanicalAgent(model)
    elif agent_type == "electronics":
        return ElectronicsAgent(model)
    elif agent_type == "firmware":
        return FirmwareAgent(model)
    elif agent_type == "bio":
        return BioAgent(model)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, 
                       help="Medical device design task")
    parser.add_argument("--mechanical_ckpt", type=str, 
                       default="./checkpoints/mechanical_expert/final_model")
    parser.add_argument("--electronics_ckpt", type=str, 
                       default="./checkpoints/electronics_expert/final_model")
    parser.add_argument("--firmware_ckpt", type=str, 
                       default="./checkpoints/firmware_expert/final_model")
    parser.add_argument("--bio_ckpt", type=str, 
                       default="./checkpoints/bio_expert/final_model")
    parser.add_argument("--latent_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all 4 trained agents
    devices = [args.device] * 4  # Same device or distribute
    agents = {
        "mechanical": load_trained_agent("mechanical", args.mechanical_ckpt, devices[0]),
        "electronics": load_trained_agent("electronics", args.electronics_ckpt, devices[1]),
        "firmware": load_trained_agent("firmware", args.firmware_ckpt, devices[2]),
        "bio": load_trained_agent("bio", args.bio_ckpt, devices[3])
    }
    
    # Initialize LatentMAS
    latent_mas = LatentMASMethod(
        agents=agents,
        latent_steps=args.latent_steps,
        latent_space_realign=True,
        sequential=True
    )
    
    # Run the design process
    print(f"\n🚀 Starting medical device design for: {args.task}")
    result = latent_mas.run(args.task)
    
    # Save the complete design
    output_file = os.path.join(args.output_dir, "design_output.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    # Extract and save individual files
    if "generated_files" in result:
        for category, files in result["generated_files"].items():
            cat_dir = os.path.join(args.output_dir, category)
            os.makedirs(cat_dir, exist_ok=True)
            
            for filename, content in files.items():
                filepath = os.path.join(cat_dir, filename)
                with open(filepath, "w") as f:
                    f.write(content)
                print(f"  📄 Saved: {filename}")
    
    print(f"\n✅ Design complete! Output saved to {args.output_dir}")
    return result

if __name__ == "__main__":
    main()