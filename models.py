# models.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import warnings

class ModelWrapper:
    """
    Wrapper for HuggingFace models that enables latent space operations
    This is the key component that allows latent collaboration
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        use_vllm: bool = False,
        torch_dtype: torch.dtype = torch.float16,
        max_length: int = 2048,
        use_flash_attention: bool = False
    ):
        """
        Initialize a model wrapper for latent space operations
        
        Args:
            model_name: Path to your fine-tuned weights or HF model name
            device: Device to load model on
            use_vllm: Whether to use vLLM (if available, for faster inference)
            torch_dtype: Precision for model weights
            max_length: Maximum sequence length
            use_flash_attention: Whether to use flash attention (if available)
        """
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        
        print(f"Loading model from {model_name} on {device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device if "cuda" in device else None,
            "trust_remote_code": True,
        }
        
        if use_flash_attention and "cuda" in device:
            model_kwargs["use_flash_attention_2"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move model to device if not using device_map
        if "cuda" in device and "device_map" not in model_kwargs:
            self.model = self.model.to(device)
        
        # Set to eval mode
        self.model.eval()
        
        # Cache for KV values during latent generation
        self.past_key_values = None
        self.current_input_ids = None
        
        print(f"✅ Model loaded successfully! {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B parameters")
    
    def generate_latent(
        self,
        prompt: str,
        latent_steps: int = 20,
        return_kv_cache: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate latent thoughts WITHOUT decoding to text
        This is the key LatentMAS function!
        
        Args:
            prompt: Input prompt
            latent_steps: Number of latent reasoning steps
            return_kv_cache: Whether to return KV cache for continuation
            temperature: Sampling temperature
        
        Returns:
            Dictionary containing latent states and KV cache
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                max_length=self.max_length).to(self.device)
        
        self.current_input_ids = inputs.input_ids
        
        # Store all hidden states
        all_hidden_states = []
        all_attention_maps = []
        
        with torch.no_grad():
            # Forward pass through the model to get initial hidden states
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Store initial hidden states
            all_hidden_states.append(outputs.hidden_states[-1].cpu().numpy())
            all_attention_maps.append(outputs.attentions[-1].cpu().numpy() if outputs.attentions else None)
            
            # Get KV cache for continuation
            past_key_values = outputs.past_key_values
            
            # Generate latent steps (generate tokens but DON'T decode to text)
            generated_tokens = []
            
            for step in range(latent_steps):
                # Sample next token logits
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = -float('Inf')
                
                # Sample token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token)
                
                # Forward pass with KV cache (efficient!)
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    attention_mask=torch.cat([inputs.attention_mask, 
                                              torch.ones(inputs.attention_mask.shape[0], 1).to(self.device)], dim=1),
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=True,
                    return_dict=True
                )
                
                # Store hidden states
                all_hidden_states.append(outputs.hidden_states[-1].cpu().numpy())
                if outputs.attentions:
                    all_attention_maps.append(outputs.attentions[-1].cpu().numpy())
                
                # Update for next iteration
                past_key_values = outputs.past_key_values
                inputs.attention_mask = torch.cat([inputs.attention_mask, 
                                                   torch.ones(inputs.attention_mask.shape[0], 1).to(self.device)], dim=1)
        
        # Store final KV cache for continuation
        self.past_key_values = past_key_values
        
        result = {
            "hidden_states": all_hidden_states,  # List of numpy arrays
            "attention_maps": all_attention_maps,
            "generated_token_ids": [t.cpu().numpy() for t in generated_tokens],
            "kv_cache": past_key_values if return_kv_cache else None,
            "latent_steps": latent_steps
        }
        
        return result
    
    def continue_from_latent(
        self,
        kv_cache: Tuple,
        prompt: Optional[str] = None,
        additional_latent_steps: int = 10,
        return_kv_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Continue latent reasoning from another agent's KV cache
        This is how agents share thoughts!
        
        Args:
            kv_cache: KV cache from previous agent
            prompt: Additional prompt to condition on
            additional_latent_steps: More latent steps
            return_kv_cache: Whether to return updated KV cache
        
        Returns:
            Updated latent states
        """
        if prompt:
            # Add new prompt to the context
            new_inputs = self.tokenizer(prompt, return_tensors="pt", 
                                       truncation=True).to(self.device)
            
            # Process with existing KV cache
            with torch.no_grad():
                outputs = self.model(
                    input_ids=new_inputs.input_ids,
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True
                )
                kv_cache = outputs.past_key_values
        else:
            outputs = None
        
        # Continue latent generation
        return self.generate_latent_from_kv(
            kv_cache=kv_cache,
            latent_steps=additional_latent_steps,
            return_kv_cache=return_kv_cache
        )
    
    def generate_latent_from_kv(
        self,
        kv_cache: Tuple,
        latent_steps: int = 10,
        return_kv_cache: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate latent thoughts starting from existing KV cache"""
        
        all_hidden_states = []
        
        with torch.no_grad():
            past_key_values = kv_cache
            
            for step in range(latent_steps):
                # Get logits from last token
                # We need to create dummy input to use past_key_values
                dummy_input = torch.ones((1, 1), dtype=torch.long).to(self.device) * self.tokenizer.pad_token_id
                
                outputs = self.model(
                    input_ids=dummy_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Store hidden states
                all_hidden_states.append(outputs.hidden_states[-1].cpu().numpy())
                
                # Sample next token (but we don't need it for latent thoughts)
                logits = outputs.logits[:, -1, :] / temperature
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update KV cache
                past_key_values = outputs.past_key_values
        
        return {
            "hidden_states": all_hidden_states,
            "kv_cache": past_key_values if return_kv_cache else None,
            "latent_steps": latent_steps
        }
    
    def decode_latent(
        self,
        kv_cache: Optional[Tuple] = None,
        prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Decode latent thoughts to actual text
        This is called at the end of collaboration
        """
        if kv_cache is None:
            kv_cache = self.past_key_values
        
        if prompt:
            # Add final prompt
            new_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=new_inputs.input_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                    return_dict=True
                )
                kv_cache = outputs.past_key_values
        
        # Now generate actual tokens
        generated = []
        past_key_values = kv_cache
        
        for _ in range(max_tokens):
            dummy_input = torch.ones((1, 1), dtype=torch.long).to(self.device) * self.tokenizer.pad_token_id
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=dummy_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[0, indices_to_remove] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                past_key_values = outputs.past_key_values
                
                # Stop if EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode to text
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text
    
    def get_hidden_state_dim(self) -> int:
        """Get dimension of hidden states"""
        return self.model.config.hidden_size


class LatentSpaceAligner:
    """
    Aligns latent spaces between different models
    This is used when mixing different model families
    """
    
    def __init__(self, source_dim: int, target_dim: int):
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        # Learnable alignment matrix
        self.alignment_matrix = torch.nn.Linear(source_dim, target_dim, bias=False)
        
    def align(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Align hidden states from source to target space"""
        return self.alignment_matrix(hidden_states)
    
    def train_alignment(
        self,
        source_states: List[torch.Tensor],
        target_states: List[torch.Tensor],
        epochs: int = 100
    ):
        """Train alignment matrix using paired hidden states"""
        optimizer = torch.optim.Adam(self.alignment_matrix.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for src, tgt in zip(source_states, target_states):
                aligned = self.align(src)
                loss = loss_fn(aligned, tgt)
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(source_states):.4f}")


# Utility function to load multiple models
def load_model_ensemble(
    model_paths: Dict[str, str],
    device_map: Optional[Dict[str, str]] = None,
    use_flash_attention: bool = False
) -> Dict[str, ModelWrapper]:
    """
    Load multiple models for multi-agent system
    
    Args:
        model_paths: Dict mapping agent names to model paths
        device_map: Optional mapping of agents to devices
    
    Returns:
        Dict of loaded ModelWrappers
    """
    models = {}
    
    for agent_name, model_path in model_paths.items():
        device = device_map.get(agent_name, "cuda:0") if device_map else "cuda:0"
        
        print(f"\n🔄 Loading {agent_name} agent...")
        models[agent_name] = ModelWrapper(
            model_name=model_path,
            device=device,
            use_flash_attention=use_flash_attention
        )
    
    return models


# If you want to use vLLM (faster inference, Linux only)
class VLLMWrapper(ModelWrapper):
    """Wrapper for vLLM - faster inference but only on Linux"""
    
    def __init__(self, model_name: str, device: str = "cuda:0", **kwargs):
        try:
            from vllm import LLM, SamplingParams
            self.vllm_available = True
        except ImportError:
            warnings.warn("vLLM not installed, falling back to HF")
            super().__init__(model_name, device, **kwargs)
            return
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            dtype="float16"
        )
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048
        )
        self.vllm_mode = True
        
    def generate_latent(self, prompt, latent_steps=20, **kwargs):
        """vLLM doesn't support latent generation, so we approximate"""
        if not hasattr(self, 'vllm_mode'):
            return super().generate_latent(prompt, latent_steps, **kwargs)
        
        # Approximate latent generation with multiple samples
        outputs = self.llm.generate([prompt], self.sampling_params)
        text = outputs[0].outputs[0].text
        
        # Simulate latent states (simplified)
        return {
            "hidden_states": [np.random.randn(1, 10, 4096)],  # Dummy
            "kv_cache": None,
            "latent_steps": latent_steps,
            "approximated": True
        }