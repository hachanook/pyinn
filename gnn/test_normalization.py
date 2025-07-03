#!/usr/bin/env python3
"""
Test script to verify both normalization methods work correctly
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from real_fea_dataset import create_real_fea_dataset
    
    def test_normalization_methods():
        """Test both normalization methods"""
        print("=== Testing Normalization Methods ===")
        
        # Paths to CSV files
        nodes_csv_path = './data/nodes.csv'
        elements_csv_path = './data/mesh_data/element_connectivity.csv'
        
        print(f"Loading from:")
        print(f"  Nodes: {nodes_csv_path}")
        print(f"  Elements: {elements_csv_path}")
        
        # Test minmax normalization
        print("\n--- Testing MinMax Normalization ---")
        try:
            dataset_minmax = create_real_fea_dataset(
                nodes_csv_path, 
                elements_csv_path, 
                normalization='minmax'
            )
            mesh_data_minmax = dataset_minmax[0]
            
            print(f"✓ MinMax normalization successful")
            print(f"Node features range: [{mesh_data_minmax.x.min():.6f}, {mesh_data_minmax.x.max():.6f}]")
            print(f"Target features range: [{mesh_data_minmax.y.min():.6f}, {mesh_data_minmax.y.max():.6f}]")
            
            # Test denormalization
            sample_pred = mesh_data_minmax.y[:5]
            denorm_pred = dataset_minmax.denormalize_predictions(sample_pred.numpy())
            print(f"Sample denormalized predictions: {denorm_pred[:2]}")
            
        except Exception as e:
            print(f"✗ MinMax normalization failed: {e}")
            return False
        
        # Test zscore normalization
        print("\n--- Testing ZScore Normalization ---")
        try:
            dataset_zscore = create_real_fea_dataset(
                nodes_csv_path, 
                elements_csv_path, 
                normalization='zscore'
            )
            mesh_data_zscore = dataset_zscore[0]
            
            print(f"✓ ZScore normalization successful")
            print(f"Node features mean: {mesh_data_zscore.x.mean():.6f}, std: {mesh_data_zscore.x.std():.6f}")
            print(f"Target features mean: {mesh_data_zscore.y.mean():.6f}, std: {mesh_data_zscore.y.std():.6f}")
            
            # Test denormalization
            sample_pred = mesh_data_zscore.y[:5]
            denorm_pred = dataset_zscore.denormalize_predictions(sample_pred.numpy())
            print(f"Sample denormalized predictions: {denorm_pred[:2]}")
            
        except Exception as e:
            print(f"✗ ZScore normalization failed: {e}")
            return False
        
        # Test invalid normalization method
        print("\n--- Testing Invalid Normalization Method ---")
        try:
            dataset_invalid = create_real_fea_dataset(
                nodes_csv_path, 
                elements_csv_path, 
                normalization='invalid'
            )
            print("✗ Should have failed with invalid normalization method")
            return False
        except ValueError as e:
            print(f"✓ Correctly caught invalid normalization method: {e}")
        
        return True
    
    if __name__ == "__main__":
        if test_normalization_methods():
            print("\n✓ All normalization tests passed!")
        else:
            print("\n✗ Some normalization tests failed!")
            sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires torch, pandas, and other dependencies.")
    print("Please install the required packages first.")
    sys.exit(1) 