"""
Quantum Flux Neural Network: A physics-inspired neural network without backpropagation.

This package implements a neural network architecture that processes inputs as quantum states
and evolves them through a series of quantum operations using Heun-Euler integration and
quantum tunneling channels.
"""

# Import core components
from .core import (
    polar_to_cartesian,
    cartesian_to_polar,
    encode_radius,
    encode_quantum_states,
    simplified_negative_distance,
    adaptive_threshold,
    get_causal_mask
)

# Import evolution mechanisms
from .evolution import (
    sample_timestep,
    calculate_tunneling_ratio,
    heun_euler_step,
    evolve_with_tunneling
)

# Import model components
from .model import (
    QuantumFluxConfig,
    QuantumProjection,
    QuantumFluxLayer,
    QuantumFluxModel
)

# Import training utilities
from .trainer import (
    TextDataset,
    get_tiny_shakespeare_data,
    get_wikitext103_data,
    QuantumFluxTrainer,
    train_tiny_shakespeare,
    train_wikitext103
)

__all__ = [
    # Core
    'polar_to_cartesian', 'cartesian_to_polar', 'encode_radius',
    'encode_quantum_states', 'simplified_negative_distance',
    'adaptive_threshold', 'get_causal_mask',
    
    # Evolution
    'sample_timestep', 'calculate_tunneling_ratio', 'heun_euler_step',
    'evolve_with_tunneling',
    
    # Model
    'QuantumFluxConfig', 'QuantumProjection', 'QuantumFluxLayer',
    'QuantumFluxModel',
    
    # Training
    'TextDataset', 'get_tiny_shakespeare_data', 'get_wikitext103_data',
    'QuantumFluxTrainer', 'train_tiny_shakespeare', 'train_wikitext103'
]
