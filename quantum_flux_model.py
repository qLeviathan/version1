"""
Quantum Flux Neural Network model.

Implements the complete model with state encoding, evolution,
and tunneling mechanisms in a unified framework.

Physics foundations:
- Complete quantum system evolution
- Quantum tunneling for information propagation
- Hebbian-inspired updates via quantum tunneling channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from .core import encode_quantum_states, simplified_negative_distance, get_causal_mask
from .evolution import evolve_with_tunneling, sample_timestep, heun_euler_step

class QuantumFluxConfig:
    """Configuration for Quantum Flux Neural Network."""
    
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=64,
        num_layers=6,
        max_seq_length=1024,
        dropout=0.1,
        dt_min=0.01,
        dt_max=0.05,
        min_radius=0.3,
        max_radius=1.0,
        use_tunneling=True,
        apply_sparsity=True
    ):
        """
        Initialize configuration.
        
        Args:
            vocab_size (int): Size of the vocabulary
            embed_dim (int): Dimension of embeddings
            num_layers (int): Number of quantum layers
            max_seq_length (int): Maximum sequence length
            dropout (float): Dropout probability
            dt_min (float): Minimum timestep for integration
            dt_max (float): Maximum timestep for integration
            min_radius (float): Minimum radius for quantum states
            max_radius (float): Maximum radius for quantum states
            use_tunneling (bool): Whether to use quantum tunneling
            apply_sparsity (bool): Whether to apply sparsity to attention
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.use_tunneling = use_tunneling
        self.apply_sparsity = apply_sparsity

class QuantumProjection(nn.Module):
    """
    Output projection that maintains quantum phase space interpretation.
    """
    
    def __init__(self, phase_dim, vocab_size):
        """
        Initialize projection.
        
        Args:
            phase_dim (int): Phase space dimension
            vocab_size (int): Vocabulary size
        """
        super().__init__()
        self.output_weights = nn.Parameter(torch.randn(phase_dim, vocab_size) * 0.02)
    
    def forward(self, states):
        """
        Forward pass.
        
        Args:
            states (torch.Tensor): Quantum states of shape [batch, seq, phase_dim]
            
        Returns:
            torch.Tensor: Logits of shape [batch, seq, vocab_size]
        """
        # Calculate energy based on phase space position
        energy = torch.norm(states, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # Normalize phase vectors
        normalized_states = states / (energy + 1e-12)  # Unit vectors
        
        # Project to vocabulary space using quantum interference patterns
        logits = torch.matmul(normalized_states, self.output_weights)  # [batch, seq, vocab_size]
        
        return logits

class QuantumFluxLayer(nn.Module):
    """
    Quantum Flux Layer implementing evolution and tunneling.
    
    This layer processes quantum states through Heun-Euler integration
    and quantum tunneling, then projects to higher dimensions.
    """
    
    def __init__(self, config, layer_idx):
        """
        Initialize layer.
        
        Args:
            config (QuantumFluxConfig): Model configuration
            layer_idx (int): Layer index
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Projection for higher dimensions (if needed)
        self.projection = nn.Linear(config.embed_dim, config.embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Flag for tunneling
        self.use_tunneling = config.use_tunneling
        
        # Whether to apply sparsity
        self.apply_sparsity = config.apply_sparsity
    
    def forward(self, states, causal_mask=None):
        """
        Forward pass.
        
        Args:
            states (torch.Tensor): Quantum states of shape [batch, seq, dim]
            causal_mask (torch.Tensor, optional): Causal mask
            
        Returns:
            tuple: (evolved_states, attention_score, tunneling_ratio)
        """
        batch_size = states.shape[0]
        device = states.device
        
        # Sample timestep for this layer
        dt = sample_timestep(
            batch_size, 
            dt_min=self.config.dt_min, 
            dt_max=self.config.dt_max,
            device=device
        )
        
        # Evolve states with tunneling
        if self.use_tunneling:
            evolved_states, score, tunneling_ratio = evolve_with_tunneling(
                states, 
                dt, 
                dt_max=self.config.dt_max,
                causal_mask=causal_mask
            )
        else:
            # Perform Heun-Euler integration without tunneling
            evolved_states, score = heun_euler_step(states, dt, causal_mask)
            tunneling_ratio = torch.tensor(0.0, device=device)
        
        # Apply projection to higher dimensions
        projected_states = self.projection(evolved_states)
        
        # Apply layer normalization and dropout
        normalized_states = self.layer_norm(projected_states)
        output_states = self.dropout(normalized_states)
        
        return output_states, score, tunneling_ratio

class QuantumFluxModel(nn.Module):
    """
    Complete Quantum Flux Neural Network model.
    
    This model implements a sequence of quantum layers that process
    token embeddings as quantum states through evolution and tunneling.
    """
    
    def __init__(self, config):
        """
        Initialize model.
        
        Args:
            config (QuantumFluxConfig): Model configuration
        """
        super().__init__()
        self.config = config
        
        # Quantum layers
        self.layers = nn.ModuleList(
            [QuantumFluxLayer(config, i) for i in range(config.num_layers)]
        )
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        
        # Output projection using quantum phase space
        self.output_projection = QuantumProjection(config.embed_dim, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tunneling histories for analysis
        self.tunneling_ratios = []
    
    def _init_weights(self, module):
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch, seq]
            attention_mask (torch.Tensor, optional): Attention mask
            labels (torch.Tensor, optional): Target labels
            
        Returns:
            dict: Model outputs including logits and loss if labels provided
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Reset tunneling histories
        self.tunneling_ratios = []
        
        # Create causal mask
        causal_mask = get_causal_mask(seq_length, device)
        
        # Encode token IDs to quantum states
        states = encode_quantum_states(input_ids, self.config.vocab_size, self.config.embed_dim)
        
        # Process through quantum layers
        all_states = [states]
        all_scores = []
        
        for layer in self.layers:
            states, score, tunneling_ratio = layer(states, causal_mask)
            all_states.append(states)
            all_scores.append(score)
            self.tunneling_ratios.append(tunneling_ratio)
        
        # Apply final layer normalization
        states = self.final_layer_norm(states)
        
        # Project to vocabulary using quantum phase space
        logits = self.output_projection(states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "states": all_states,
            "scores": all_scores,
            "tunneling_ratios": self.tunneling_ratios
        }
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate text using the model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            max_length (int): Maximum length to generate
            temperature (float): Temperature for sampling
            top_k (int): Top-k sampling parameter (0 for disabled)
            top_p (float): Top-p sampling parameter
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # For storing generated tokens
        generated_ids = input_ids.clone()
        
        # Generate tokens up to max_length
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass (use only the last token for efficiency)
            with torch.no_grad():
                outputs = self.forward(generated_ids)
                logits = outputs["logits"]
                
                # Extract the logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / max(temperature, 1e-6)
                
                # Apply top-k filtering
                if top_k > 0:
                    # Extract top-k tokens and their indices
                    values, indices = torch.topk(next_token_logits, top_k)
                    
                    # Create a mask of tokens to keep
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, indices, values)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set all filtered logits to -inf
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, float('-inf'))
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids

# For testing
if __name__ == "__main__":
    # Create configuration
    config = QuantumFluxConfig(
        vocab_size=1000,
        embed_dim=32,
        num_layers=3,
        max_seq_length=128,
        dropout=0.1,
        dt_min=0.01,
        dt_max=0.05,
        min_radius=0.3,
        max_radius=1.0,
        use_tunneling=True,
        apply_sparsity=True
    )
    
    # Create model
    model = QuantumFluxModel(config)
    
    # Create sample input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids)
    
    # Print shapes
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Tunneling ratios: {[ratio.item() for ratio in outputs['tunneling_ratios']]}")
    
    # Test generation
    generated_ids = model.generate(input_ids, max_length=20)
    print(f"Generated shape: {generated_ids.shape}")
