# run_layer_visualization.py
# QUICK START SCRIPT FOR LAYER-WISE VISUALIZATION

import torch
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# QUICK START: Run this script to visualize your model layers
# ============================================================================

def main():
    """
    Complete example showing how to use layer_wise_visualization.py
    
    Usage:
        python run_layer_visualization.py --checkpoint path/to/best_model.pt --output ./viz_output
    """
    
    import argparse
    parser = argparse.ArgumentParser(description='HFF-Net Layer-wise Visualization')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, help='Path to test data (optional - uses dummy if not provided)')
    parser.add_argument('--output', type=str, default='./layer_viz_output', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to visualize')
    parser.add_argument('--dpi', type=int, default=600, help='DPI for saved visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"HFF-NET LAYER-WISE VISUALIZATION")
    print(f"{'='*80}\n")
    
    # Step 1: Import modules
    print("[1/5] Importing modules...")
    try:
        from model.HFF_MobileNetV3_fixed import HFFNet
        from visualization.layer_wise_visualization import ComprehensiveLayerVisualizationPipeline
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure HFF_MobileNetV3_fixed.py and layer_wise_visualization.py are in correct paths")
        return
    
    # Step 2: Initialize device
    print("\n[2/5] Initializing device...")
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Step 3: Load model
    print("\n[3/5] Loading model...")
    try:
        model = HFFNet(in_chs1=4, in_chs2=16, num_classes=4).to(device)
        
        # Load checkpoint
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DDP)
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            
            model.load_state_dict(new_state, strict=False)
            print(f"✓ Model loaded from: {args.checkpoint}")
        
        model.eval()
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Step 4: Prepare inputs
    print("\n[4/5] Preparing inputs...")
    try:
        if args.data_path:
            print(f"Loading data from: {args.data_path}")
            # TODO: Implement actual data loading
            # For now, use dummy data
            low_freq_input = torch.randn(1, 4, 128, 128, 128)
            high_freq_input = torch.randn(1, 16, 128, 128, 128)
        else:
            print("Using dummy input data (random tensors)")
            low_freq_input = torch.randn(1, 4, 128, 128, 128)
            high_freq_input = torch.randn(1, 16, 128, 128, 128)
        
        print(f"✓ Input shapes: LF={low_freq_input.shape}, HF={high_freq_input.shape}")
    
    except Exception as e:
        print(f"✗ Error preparing inputs: {e}")
        return
    
    # Step 5: Run visualization pipeline
    print("\n[5/5] Running visualization pipeline...")
    print(f"Output directory: {args.output}\n")
    
    try:
        pipeline = ComprehensiveLayerVisualizationPipeline(
            model=model,
            device=device,
            output_base_dir=args.output
        )
        
        # Visualize sample(s)
        for sample_idx in range(args.num_samples):
            sample_name = f'sample_{sample_idx:04d}'
            print(f"\n>>> Processing {sample_name}...")
            
            results = pipeline.analyze_sample(
                low_freq_input=low_freq_input,
                high_freq_input=high_freq_input,
                sample_name=sample_name
            )
            
            print(f"\n✓ Results saved to: {results['session_dir']}")
            print(f"  Report: {results['report_path']}")
            print(f"  Visualizations: {len(results['saved_paths'])} files")
    
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print(f"\n{'='*80}")
    print(f"VISUALIZATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nOutput saved to: {results['session_dir']}")
    print(f"\nNext steps:")
    print(f"  1. Open the PNG files to visualize layer activations")
    print(f"  2. Read {results['report_path']} for activation statistics")
    print(f"  3. Analyze visualization trends across layers")
    print(f"  4. Use for ablation studies and debugging")
    print(f"\nFor integration with eval_new_FIXED.py:")
    print(f"  See LAYER_VIZ_GUIDE.md for examples")
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
