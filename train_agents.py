# train_agents.py
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch
import os

def train_agent(agent_type, data_path, output_dir, base_model="Qwen/Qwen2.5-7B-Instruct"):
    """Fine-tune a specialized agent"""
    
    print(f"\n🚀 Training {agent_type} agent...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load your cleaned data
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    # Tokenize
    def tokenize_function(examples):
        texts = []
        for inst, inp, out in zip(examples['instruction'], 
                                   examples.get('input', ['']*len(examples)), 
                                   examples['output']):
            if inp:
                text = f"### Instruction: {inst}\n### Input: {inp}\n### Output: {out}"
            else:
                text = f"### Instruction: {inst}\n### Output: {out}"
            texts.append(text)
        
        return tokenizer(texts, truncation=True, padding=True, max_length=2048)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-5,
        fp16=True,
        save_total_limit=2,
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Train!
    trainer.train()
    
    # Save final model
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"✅ {agent_type} agent saved to {final_path}")
    return final_path

if __name__ == "__main__":
    # Create checkpoint directories
    os.makedirs("checkpoints/mechanical_expert", exist_ok=True)
    os.makedirs("checkpoints/electronics_expert", exist_ok=True)
    os.makedirs("checkpoints/firmware_expert", exist_ok=True)
    os.makedirs("checkpoints/bio_expert", exist_ok=True)
    
    # Train all 4 agents
    train_agent(
        "mechanical",
        "data/mechanical/cad_instructions.jsonl",
        "checkpoints/mechanical_expert"
    )
    
    train_agent(
        "electronics",
        "data/electronics/vhdl_examples.jsonl",
        "checkpoints/electronics_expert"
    )
    
    train_agent(
        "firmware",
        "data/firmware/embedded_cpp.jsonl",
        "checkpoints/firmware_expert"
    )
    
    train_agent(
        "bio",
        "data/bio/fda_guidelines.jsonl",
        "checkpoints/bio_expert"
    )
    
    print("\n✅ All 4 agents trained successfully!")