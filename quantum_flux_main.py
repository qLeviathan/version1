"""
Training script for Quantum Flux Neural Network.

This script can be used to train the Quantum Flux Neural Network on
either tiny Shakespeare or WikiText-103 datasets.
"""

import os
import argparse
import torch
from quantum_flux import (
    QuantumFluxConfig,
    QuantumFluxModel,
    train_tiny_shakespeare,
    train_wikitext103
)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Quantum Flux Neural Network")
    
    # Dataset options
    parser.add_argument("--dataset", type=str, default="shakespeare", choices=["shakespeare", "wikitext"],
                        help="Dataset to train on (shakespeare or wikitext)")
    
    # Model configuration
    parser.add_argument("--embed_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of quantum layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--dt_min", type=float, default=0.01, help="Minimum timestep for integration")
    parser.add_argument("--dt_max", type=float, default=0.05, help="Maximum timestep for integration")
    parser.add_argument("--min_radius", type=float, default=0.3, help="Minimum radius for quantum states")
    parser.add_argument("--max_radius", type=float, default=1.0, help="Maximum radius for quantum states")
    parser.add_argument("--use_tunneling", action="store_true", help="Use quantum tunneling channels")
    parser.add_argument("--apply_sparsity", action="store_true", help="Apply sparsity to attention")
    
    # Training configuration
    parser.add_argument("--context_length", type=int, default=128, help="Context length for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (for Shakespeare)")
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of steps (for WikiText)")
    parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save models")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create model configuration
    config = QuantumFluxConfig(
        vocab_size=65 if args.dataset == "shakespeare" else 30000,  # Will be overridden by dataset
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        max_seq_length=args.context_length,
        dropout=args.dropout,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        use_tunneling=args.use_tunneling,
        apply_sparsity=args.apply_sparsity
    )
    
    # Train on selected dataset
    if args.dataset == "shakespeare":
        model, trainer, metrics, vocab_info = train_tiny_shakespeare(
            config=config,
            batch_size=args.batch_size,
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            device=args.device
        )
        
        # Generate sample text after training
        chars, char_to_idx, idx_to_char = vocab_info
        model.eval()
        
        # Start with a seed text
        seed_text = "ROMEO:"
        seed_ids = torch.tensor([[char_to_idx[ch] for ch in seed_text]], dtype=torch.long)
        
        # Generate
        generated_ids = model.generate(
            seed_ids.to(args.device),
            max_length=200,
            temperature=0.8
        )
        
        # Convert back to text
        generated_text = ''.join([idx_to_char[idx] for idx in generated_ids[0].cpu().numpy()])
        print(f"\nGenerated text:\n{generated_text}")
        
        # Save vocabulary info for generation
        torch.save(vocab_info, os.path.join(args.save_dir, "shakespeare_vocab.pt"))
        
    else:  # WikiText-103
        model, trainer, metrics = train_wikitext103(
            config=config,
            batch_size=args.batch_size,
            context_length=args.context_length,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            device=args.device
        )
    
    # Save final model and training metrics
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))
    torch.save(metrics, os.path.join(args.save_dir, "training_metrics.pt"))
    
    print(f"Training completed. Final model saved to {os.path.join(args.save_dir, 'final_model.pt')}")

if __name__ == "__main__":
    main()