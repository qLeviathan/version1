"""
Quantum state evolution mechanisms.

Implements Heun-Euler integration for quantum state evolution
and quantum tunneling for information propagation.

Physics foundations:
- Imaginary-time Schrödinger equation: ∂ψ/∂τ = -Ĥψ
- Radial diffusion equation: ∂u/∂t = α·[1/r·∂/∂r(r·∂u/∂r)]
- Quantum tunneling via skip connections
"""

import torch
import math
import numpy as np
from .core import simplified_negative_distance

def sample_timestep(batch_size, dt_min=0.01, dt_max=0.05, device=None):
    """
    Sample timesteps for Heun-Euler integration.
    
    This function samples timesteps from a distribution that favors
    smaller timesteps for stability, while occasionally allowing
    larger timesteps for exploration.
    
    Args:
        batch_size (int): Batch size
        dt_min (float): Minimum timestep
        dt_max (float): Maximum timestep
        device (torch.device): Device to create tensor on
        
    Returns:
        torch.Tensor: Sampled timesteps of shape [batch_size, 1, 1]
    """
    # Sample from beta distribution (favors smaller timesteps)
    alpha, beta = 2.0, 5.0
    dt = torch.tensor(np.random.beta(alpha, beta, size=batch_size), device=device).float()
    
    # Scale to [dt_min, dt_max]
    dt = dt_min + (dt_max - dt_min) * dt
    
    # Reshape for broadcasting
    dt = dt.view(batch_size, 1, 1)
    
    return dt

def calculate_tunneling_ratio(dt, dt_max=0.05):
    """
    Calculate quantum tunneling ratio (skip connection) based on timestep.
    
    This implements a direct linear relationship between timestep and
    tunneling probability, without using activation functions.
    
    Physics foundation: Represents quantum tunneling probability where
    information can 'tunnel' through layers based on energy levels.
    
    Args:
        dt (torch.Tensor): Timestep tensor
        dt_max (float): Maximum timestep
        
    Returns:
        torch.Tensor: Tunneling ratio
    """
    # Direct linear relationship
    tunneling_ratio = dt / dt_max
    
    # Clamp to [0, 1] range for valid probability
    tunneling_ratio = torch.clamp(tunneling_ratio, 0.0, 1.0)
    
    return tunneling_ratio

def heun_euler_step(states, dt, causal_mask=None):
    """
    Perform one Heun-Euler integration step for quantum state evolution.
    
    This is a second-order Runge-Kutta method that provides a good
    balance between accuracy and computational efficiency.
    
    Physics foundation: Approximates solutions to the imaginary-time
    Schrödinger equation for quantum state evolution.
    
    Args:
        states (torch.Tensor): Quantum states of shape [batch, seq, dim]
        dt (torch.Tensor): Timestep tensor of shape [batch, 1, 1]
        causal_mask (torch.Tensor, optional): Causal mask for attention
        
    Returns:
        tuple: (evolved_states, attention_score)
    """
    # First step (k1): Compute attention and context
    score = simplified_negative_distance(states)
    if causal_mask is not None:
        score = score + causal_mask
    context = torch.bmm(score, states)
    k1 = dt * context
    
    # Intermediate state
    states_mid = states + k1
    
    # Second step (k2): Compute attention and context for intermediate state
    score_mid = simplified_negative_distance(states_mid)
    if causal_mask is not None:
        score_mid = score_mid + causal_mask
    context_mid = torch.bmm(score_mid, states_mid)
    k2 = dt * context_mid
    
    # Final state (Heun-Euler formula)
    evolved_states = states + 0.5 * (k1 + k2)
    
    return evolved_states, score

def evolve_with_tunneling(states, dt, dt_max=0.05, causal_mask=None):
    """
    Evolve quantum states with tunneling (skip connections).
    
    This function combines Heun-Euler integration with quantum tunneling,
    allowing information to bypass the integration step based on a 
    tunneling probability derived from the timestep.
    
    Args:
        states (torch.Tensor): Quantum states of shape [batch, seq, dim]
        dt (torch.Tensor): Timestep tensor of shape [batch, 1, 1]
        dt_max (float): Maximum timestep
        causal_mask (torch.Tensor, optional): Causal mask for attention
        
    Returns:
        tuple: (evolved_states, attention_score, tunneling_ratio)
    """
    # Store original states for tunneling
    original_states = states.clone()
    
    # Perform Heun-Euler integration
    evolved_states, score = heun_euler_step(states, dt, causal_mask)
    
    # Calculate tunneling ratio
    tunneling_ratio = calculate_tunneling_ratio(dt, dt_max)
    
    # Apply tunneling (skip connection)
    final_states = tunneling_ratio * original_states + (1 - tunneling_ratio) * evolved_states
    
    return final_states, score, tunneling_ratio

# For testing
if __name__ == "__main__":
    import numpy as np
    from .core import encode_quantum_states
    
    # Create test data
    batch_size = 2
    seq_len = 5
    embed_dim = 2
    vocab_size = 100
    
    # Create random token indices
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Encode to quantum states
    states = encode_quantum_states(indices, vocab_size, embed_dim)
    
    # Sample timestep
    dt = sample_timestep(batch_size, device=states.device)
    
    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).unsqueeze(0)
    
    # Test Heun-Euler integration
    evolved_states, score = heun_euler_step(states, dt, causal_mask)
    print("Original states shape:", states.shape)
    print("Evolved states shape:", evolved_states.shape)
    print("Attention score shape:", score.shape)
    
    # Test tunneling
    final_states, score, tunneling_ratio = evolve_with_tunneling(states, dt, causal_mask=causal_mask)
    print("\nFinal states shape:", final_states.shape)
    print("Tunneling ratio:", tunneling_ratio.item())
    
    # Calculate radius before and after evolution
    original_radius = torch.norm(states[:, :, :2], dim=-1).mean().item()
    evolved_radius = torch.norm(evolved_states[:, :, :2], dim=-1).mean().item()
    final_radius = torch.norm(final_states[:, :, :2], dim=-1).mean().item()
    
    print(f"\nRadius evolution:")
    print(f"Original: {original_radius:.4f}")
    print(f"Evolved (no tunneling): {evolved_radius:.4f}")
    print(f"Final (with tunneling): {final_radius:.4f}")
