#!/usr/bin/env python3
"""
Test script to verify that the recovered data files can be loaded correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_fea_dataset import RealFEADataset

def test_data_loading():
    """Test loading the recovered data files"""
    print("Testing data recovery...")
    
    try:
        # Test loading with min-max normalization
        print("\n1. Testing min-max normalization...")
        dataset = RealFEADataset(
            nodes_csv_path='data/nodes.csv',
            elements_csv_path='data/mesh_data/element_connectivity.csv',
            normalization='minmax'
        )
        
        # Get the mesh data
        mesh_data = dataset[0]
        print(f"✓ Successfully loaded mesh data")
        print(f"  - Nodes: {mesh_data.x.shape[0]}")
        print(f"  - Node features: {mesh_data.x.shape[1]}")
        print(f"  - Edges: {mesh_data.edge_index.shape[1]}")
        print(f"  - Edge features: {mesh_data.edge_attr.shape[1]}")
        print(f"  - Targets: {mesh_data.y.shape[1]}")
        
        # Test z-score normalization
        print("\n2. Testing z-score normalization...")
        dataset_zscore = RealFEADataset(
            nodes_csv_path='data/nodes.csv',
            elements_csv_path='data/mesh_data/element_connectivity.csv',
            normalization='zscore'
        )
        
        mesh_data_zscore = dataset_zscore[0]
        print(f"✓ Successfully loaded mesh data with z-score normalization")
        
        # Test denormalization
        print("\n3. Testing denormalization...")
        norm_params = dataset.get_normalization_params()
        print(f"✓ Normalization parameters retrieved: {list(norm_params.keys())}")
        
        # Test with some dummy predictions
        dummy_predictions = mesh_data.y[:5]  # First 5 predictions
        denormalized = dataset.denormalize_predictions(dummy_predictions.numpy())
        print(f"✓ Denormalization successful, shape: {denormalized.shape}")
        
        print("\n🎉 All tests passed! Data recovery successful.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\nThe recovered data directory is ready for use!")
        print("You can now run: python main.py")
    else:
        print("\nData recovery failed. Please check the error messages above.") 