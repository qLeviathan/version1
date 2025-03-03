"""
Test script for Quantum Flux Neural Network.

This script runs basic tests to verify the implementation of the
Quantum Flux Neural Network components.
"""

import torch
import numpy as np
from quantum_flux import (
    encode_quantum_states,
    simplified_negative_distance,
    heun_euler_step,
    evolve_with_tunneling,
    QuantumFluxConfig,
    QuantumFluxModel
)

def test_quantum_encoding():
    """Test quantum state encoding."""
    print("\n===== Testing Quantum State Encoding =====")
    
    # Create test data
    indices = torch.tensor([[0, 1, 2, 3, 4]])
    vocab_size = 10
    embed_dim = 4
    
    # Encode to quantum states
    states = encode_quantum_states(indices, vocab_size, embed_dim)
    
    # Print states
    print(f"States shape: {states.shape}")
    print("States:")
    print(states)
    
    # Check radius values
    radii = torch.norm(states[:, :, :2], dim=-1)
    print("\nRadii:")
    print(radii)
    print(f"Radii range: [{radii.min().item():.4f}, {radii.max().item():.4f}]")
    
    # Expected range is [0.3, 1.0] scaled by index/vocab_size
    expected_min = 0.3
    expected_max = 0.3 + 0.7 * (4/10)
    print(f"Expected range: [{expected_min:.4f}, {expected_max:.4f}]")
    
    assert abs(radii.min().item() - expected_min) < 1e-4, "Minimum radius is incorrect"
    assert abs(radii.max().item() - expected_max) < 1e-4, "Maximum radius is incorrect"
    
    print("✓ Quantum state encoding works correctly")

def test_simplified_negative_distance():
    """Test simplified negative distance calculation."""
    print("\n===== Testing Simplified Negative Distance =====")
    
    # Create test data
    states = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0]
        ]
    ])
    
    # Calculate negative distance
    score = simplified_negative_distance(states)
    
    # Print score
    print(f"Score shape: {score.shape}")
    print("Score:")
    print(score)
    
    # Check score range
    print(f"Score range: [{score.min().item():.4f}, {score.max().item():.4f}]")
    
    # Expected range is [0, 1]
    assert score.min().item() >= 0, "Minimum score is less than 0"
    assert score.max().item() <= 1, "Maximum score is greater than 1"
    
    # Confirm diagonal is 1 (same state has highest similarity)
    diagonal = torch.diagonal(score, dim1=1, dim2=2)
    print("\nDiagonal (self-similarity):")
    print(diagonal)
    
    assert torch.allclose(diagonal, torch.ones_like(diagonal)), "Self-similarity is not 1"
    
    print("✓ Simplified negative distance works correctly")

def test_heun_euler_integration():
    """Test Heun-Euler integration."""
    print("\n===== Testing Heun-Euler Integration =====")
    
    # Create test data
    batch_size = 2
    seq_len = 5
    embed_dim = 4
    vocab_size = 100
    
    # Create random token indices
    indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Encode to quantum states
    states = encode_quantum_states(indices, vocab_size, embed_dim)
    
    # Set a fixed timestep for testing
    dt = torch.tensor([0.05] * batch_size).view(batch_size, 1, 1)
    
    # Original states
    print(f"Original states shape: {states.shape}")
    original_radii = torch.norm(states[:, :, :2], dim=-1)
    print(f"Original radii mean: {original_radii.mean().item():.4f}")
    
    # Evolve with Heun-Euler integration
    evolved_states, score = heun_euler_step(states, dt)
    
    # Check evolved states
    print(f"Evolved states shape: {evolved_states.shape}")
    evolved_radii = torch.norm(evolved_states[:, :, :2], dim=-1)
    print(f"Evolved radii mean: {evolved_radii.mean().item():.4f}")
    
    # Evolve with tunneling
    final_states, score, tunneling_ratio = evolve_with_tunneling(states, dt)
    
    # Check final states
    print(f"Final states shape: {final_states.shape}")
    final_radii = torch.norm(final_states[:, :, :2], dim=-1)
    print(f"Final radii mean: {final_radii.mean().item():.4f}")
    print(f"Tunneling ratio: {tunneling_ratio.item():.4f}")
    
    # We expect tunneling to mitigate radius growth
    radius_growth_no_tunneling = evolved_radii.mean().item() - original_radii.mean().item()
    radius_growth_with_tunneling = final_radii.mean().item() - original_radii.mean().item()
    
    print(f"Radius growth without tunneling: {radius_growth_no_tunneling:.4f}")
    print(f"Radius growth with tunneling: {radius_growth_with_tunneling:.4f}")
    
    assert radius_growth_with_tunneling < radius_growth_no_tunneling, "Tunneling doesn't mitigate radius growth"
    
    print("✓ Heun-Euler integration and tunneling work correctly")

def test_model():
    """Test model forward pass."""
    print("\n===== Testing Model Forward Pass =====")
    
    # Create configuration
    config = QuantumFluxConfig(
        vocab_size=1000,
        embed_dim=32,
        num_layers=2,
        max_seq_length=16,
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
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    
    # Check outputs
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Number of states: {len(outputs['states'])}")
    print(f"Number of tunneling ratios: {len(outputs['tunneling_ratios'])}")
    
    # Test generation
    generated_ids = model.generate(input_ids[:, :2], max_length=5)
    print(f"Generated shape: {generated_ids.shape}")
    
    assert generated_ids.shape[1] == 5, "Generated sequence length is incorrect"
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size), "Logits shape is incorrect"
    assert len(outputs['states']) == config.num_layers + 1, "Number of states is incorrect"
    assert len(outputs['tunneling_ratios']) == config.num_layers, "Number of tunneling ratios is incorrect"
    
    print("✓ Model forward pass works correctly")

def run_all_tests():
    """Run all tests."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_quantum_encoding()
    test_simplified_negative_distance()
    test_heun_euler_integration()
    test_model()
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    run_all_tests()
