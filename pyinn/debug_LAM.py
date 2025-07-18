import numpy as np
import pandas as pd
import time
import os
import pickle
import tensorly as tl
from tensorly.decomposition import parafac


def convert_data_to_tensor(train_filename, test_filename):
    """
    Convert CSV data to tensor format and save as numpy arrays.
    
    Returns:
        tuple: (train_tensor, test_tensor) - The converted tensors
    """
    # Paths to CSV files
    train_csv = f'./data/{train_filename}.csv'
    test_csv = f'./data/{test_filename}.csv'
    
    # Read CSV files using pandas
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Convert DataFrames to numpy arrays
    # train_array = train_df.values
    # test_array = test_df.values

    # Get unique values for time steps and flow rates
    unique_nodes = 2750
    unique_timesteps = np.sort(np.array(train_df['t'].unique()))
    unique_flowrates = np.sort(np.array(train_df['sample_value'].unique()))

    # Create mapping from node coordinates to node index
    node_coords = train_df[['y', 'z']].drop_duplicates().values
    node_to_index = {tuple(coord): idx for idx, coord in enumerate(node_coords)}

    # # Initialize tensor
    train_tensor = np.zeros((unique_nodes, len(unique_timesteps), len(unique_flowrates)))

    # Fill tensor
    start_time = time.time()
    for i, row in enumerate(train_df.iterrows()):
        node_idx = node_to_index[(row[1]['y'], row[1]['z'])]
        timestep_idx = np.where(unique_timesteps == row[1]['t'])[0][0]
        flowrate_idx = np.where(unique_flowrates == row[1]['sample_value'])[0][0]
        train_tensor[node_idx, timestep_idx, flowrate_idx] = row[1]['concentration_h2']
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"Iteration {i}: {elapsed:.4f} seconds")
            start_time = time.time()
    train_nodes = {'node_coords': node_coords, 'timesteps': unique_timesteps, 'flowrates': unique_flowrates}

    # Get the fixed flow rate value from the test set
    test_flow_rate = test_df['sample_value'].iloc[0]

    # Get unique timesteps in test set
    test_timesteps = np.sort(np.array(test_df['t'].unique()))

    # Initialize test tensor
    test_tensor = np.zeros((unique_nodes, len(test_timesteps)))

    # Fill test tensor
    for _, row in test_df.iterrows():
        node_idx = node_to_index[(row['y'], row['z'])]
        timestep_idx = np.where(test_timesteps == row['t'])[0][0]
        test_tensor[node_idx, timestep_idx] = row['concentration_h2']
    test_nodes = {'node_coords': node_coords, 'timesteps': test_timesteps, 'flowrates': test_flow_rate}

    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Test tensor shape: {test_tensor.shape}")
    
    # Save tensors to files
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Save node dictionaries
    with open(os.path.join(data_dir, f'{train_filename}.pkl'), 'wb') as f:
        pickle.dump({'tensor': train_tensor, 'nodes': train_nodes}, f)
    with open(os.path.join(data_dir, f'{test_filename}.pkl'), 'wb') as f:
        pickle.dump({'tensor': test_tensor, 'nodes': test_nodes}, f)
    
    print(f"Tensors and node dictionaries saved to {data_dir}/")
    
    # return train_tensor, test_tensor


def load_tensors_from_files(train_filename, test_filename):
    """
    Load the saved tensor files and node dictionaries from the ./data directory.
    
    Returns:
        tuple: (train_tensor, train_nodes, test_tensor, test_nodes) - The loaded tensors and node dictionaries
    """
    data_dir = './data'
    
    # Load train nodes dictionary
    train_nodes_path = os.path.join(data_dir, f'{train_filename}.pkl')
    if os.path.exists(train_nodes_path):
        with open(train_nodes_path, 'rb') as f:
            train_data = pickle.load(f)
            train_tensor = train_data['tensor']
            train_nodes = train_data['nodes']
        print(f"Loaded train tensor with shape: {train_tensor.shape}")
        print(f"Loaded train nodes dictionary with keys: {list(train_nodes.keys())}")
    else:
        print(f"Train data file not found at {train_nodes_path}")
        train_tensor = None
        train_nodes = None
    
    # Load test nodes dictionary
    test_nodes_path = os.path.join(data_dir, f'{test_filename}.pkl')
    if os.path.exists(test_nodes_path):
        with open(test_nodes_path, 'rb') as f:
            test_data = pickle.load(f)
            test_tensor = test_data['tensor']
            test_nodes = test_data['nodes']
        print(f"Loaded test tensor with shape: {test_tensor.shape}")
        print(f"Loaded test nodes dictionary with keys: {list(test_nodes.keys())}")
    else:
        print(f"Test data file not found at {test_nodes_path}")
        test_tensor = None
        test_nodes = None
    
    return train_tensor, train_nodes, test_tensor, test_nodes



def cp_decomposition(tensor, rank, max_iter=1000, tol=1e-8, random_state=42, max_rank_increase=10):
    """
    Perform CP (Canonical Polyadic) tensor decomposition using tensorly.
    If SVD convergence fails, automatically increase rank until convergence.
    
    Args:
        tensor (np.ndarray): Input tensor to decompose
        rank (int): Initial rank of the CP decomposition
        max_iter (int): Maximum number of iterations
        tol (float): Tolerance for convergence
        random_state (int): Random seed for reproducibility
        max_rank_increase (int): Maximum number of rank increases to try
        
    Returns:
        tuple: (weights, factors, error_metrics) - CP decomposition results
    """
    print(f"Attempting CP decomposition with initial rank {rank}...")
    print(f"Input tensor shape: {tensor.shape}")
    
    # Set random seed for reproducibility
    tl.set_backend('numpy')
    np.random.seed(random_state)
    
    current_rank = rank
    max_rank = rank + max_rank_increase
    
    while current_rank <= max_rank:
        try:
            print(f"Trying rank {current_rank}...")
            
            # Perform CP decomposition
            start_time = time.time()
            weights, factors = parafac(tensor, rank=current_rank, n_iter_max=max_iter, 
                                      tol=tol, random_state=random_state)
            elapsed_time = time.time() - start_time
            
            print(f"CP decomposition completed successfully with rank {current_rank} in {elapsed_time:.4f} seconds")
            print(f"Weights shape: {weights.shape}")
            for i, factor in enumerate(factors):
                print(f"Factor {i+1} shape: {factor.shape}")
            
            # Reconstruct tensor to check quality
            reconstructed = tl.cp_to_tensor((weights, factors))
            
            # Calculate various error metrics
            mse = np.mean((tensor - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(tensor - reconstructed))
            
            # Relative errors
            relative_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
            relative_mse = mse / np.mean(tensor ** 2)
            
            # R-squared (coefficient of determination)
            ss_res = np.sum((tensor - reconstructed) ** 2)
            ss_tot = np.sum((tensor - np.mean(tensor)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Print error metrics
            print(f"Error Metrics:")
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  Relative Error: {relative_error:.6f}")
            print(f"  Relative MSE: {relative_mse:.6f}")
            print(f"  R-squared: {r_squared:.6f}")
            
            return weights, factors, {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'relative_error': relative_error,
                'relative_mse': relative_mse,
                'r_squared': r_squared,
                'reconstructed': reconstructed,
                'final_rank': current_rank,
                'rank_increases': current_rank - rank
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if 'svd' in error_msg and ('converge' in error_msg or 'convergence' in error_msg):
                print(f"Rank {current_rank} failed with SVD convergence error: {e}")
                if current_rank < max_rank:
                    current_rank += 1
                    print(f"Increasing rank to {current_rank} and retrying...")
                    continue
                else:
                    print(f"Reached maximum rank {max_rank}. Cannot increase further.")
                    raise e
            else:
                # If it's not an SVD convergence error, re-raise the exception
                print(f"Rank {current_rank} failed with non-SVD error: {e}")
                raise e
    
    # If we get here, all ranks failed
    raise RuntimeError(f"CP decomposition failed for all ranks from {rank} to {max_rank}")


def analyze_decomposition_quality(original_tensor, reconstructed_tensor, error_metrics):
    """
    Analyze the quality of tensor decomposition with detailed metrics.
    
    Args:
        original_tensor (np.ndarray): Original tensor
        reconstructed_tensor (np.ndarray): Reconstructed tensor from decomposition
        error_metrics (dict): Dictionary containing error metrics
    """
    print("\n" + "="*50)
    print("DETAILED DECOMPOSITION QUALITY ANALYSIS")
    print("="*50)
    
    # Original tensor statistics
    print(f"Original Tensor Statistics:")
    print(f"  Shape: {original_tensor.shape}")
    print(f"  Mean: {original_tensor.mean():.6f}")
    print(f"  Std: {original_tensor.std():.6f}")
    print(f"  Min: {original_tensor.min():.6f}")
    print(f"  Max: {original_tensor.max():.6f}")
    print(f"  Non-zero elements: {np.count_nonzero(original_tensor)}")
    print(f"  Sparsity: {1 - np.count_nonzero(original_tensor) / original_tensor.size:.4f}")
    
    # Reconstructed tensor statistics
    print(f"\nReconstructed Tensor Statistics:")
    print(f"  Shape: {reconstructed_tensor.shape}")
    print(f"  Mean: {reconstructed_tensor.mean():.6f}")
    print(f"  Std: {reconstructed_tensor.std():.6f}")
    print(f"  Min: {reconstructed_tensor.min():.6f}")
    print(f"  Max: {reconstructed_tensor.max():.6f}")
    
    # Error analysis by mode
    print(f"\nError Analysis by Mode:")
    for mode in range(original_tensor.ndim):
        mode_errors = np.abs(original_tensor - reconstructed_tensor)
        mode_errors = np.mean(mode_errors, axis=tuple(i for i in range(original_tensor.ndim) if i != mode))
        print(f"  Mode {mode+1} average error: {mode_errors.mean():.6f} ± {mode_errors.std():.6f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if error_metrics['r_squared'] > 0.9:
        quality = "Excellent"
    elif error_metrics['r_squared'] > 0.7:
        quality = "Good"
    elif error_metrics['r_squared'] > 0.5:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"  Overall Quality: {quality} (R² = {error_metrics['r_squared']:.4f})")
    print(f"  Reconstruction captures {error_metrics['r_squared']*100:.1f}% of variance")
    
    return quality


def normalize_tensors(train_tensor, test_tensor):
    """
    Normalize both train_tensor and test_tensor between 0 and 1
    using the min and max values from train_tensor.
    
    Args:
        train_tensor (np.ndarray): Training tensor
        test_tensor (np.ndarray): Test tensor
        
    Returns:
        tuple: (normalized_train, normalized_test, normalization_params)
    """
    print("Normalizing tensors...")
    
    # Calculate min and max from train_tensor
    train_min = np.min(train_tensor)
    train_max = np.max(train_tensor)
    
    print(f"Train tensor - Original range: [{train_min:.6f}, {train_max:.6f}]")
    
    # Check if test_tensor has values outside train range
    test_min = np.min(test_tensor)
    test_max = np.max(test_tensor)
    print(f"Test tensor - Original range: [{test_min:.6f}, {test_max:.6f}]")
    
    if test_min < train_min or test_max > train_max:
        print("Warning: Test tensor contains values outside train tensor range!")
        print("These values will be clipped to [0, 1] after normalization.")
    
    # Normalize tensors
    normalized_train = (train_tensor - train_min) / (train_max - train_min)
    normalized_test = (test_tensor - train_min) / (train_max - train_min)
    
    # Clip test tensor to [0, 1] range if needed
    normalized_test = np.clip(normalized_test, 0, 1)
    
    # Verify normalization
    print(f"Normalized train tensor range: [{normalized_train.min():.6f}, {normalized_train.max():.6f}]")
    print(f"Normalized test tensor range: [{normalized_test.min():.6f}, {normalized_test.max():.6f}]")
    
    # Store normalization parameters for later use
    normalization_params = {
        'train_min': train_min,
        'train_max': train_max,
        'range': train_max - train_min
    }
    
    return normalized_train, normalized_test, normalization_params


def denormalize_tensor(normalized_tensor, normalization_params):
    """
    Denormalize a tensor using the stored normalization parameters.
    
    Args:
        normalized_tensor (np.ndarray): Normalized tensor
        normalization_params (dict): Dictionary containing min, max, and range values
        
    Returns:
        np.ndarray: Denormalized tensor
    """
    return normalized_tensor * normalization_params['range'] + normalization_params['train_min']


def interpolate_flow_rate(flow_rates, test_flow_rate):
    """
    Perform 1D linear interpolation to find adjacent nodes and relative distances.
    
    Args:
        flow_rates (np.ndarray): Array of available flow rates in ascending order
        test_flow_rate (float): Target flow rate to interpolate
        
    Returns:
        dict: Interpolation results containing indices and distances
    """
    print(f"\nFlow rate interpolation:")
    print(f"Available flow rates: {flow_rates}")
    print(f"Test flow rate: {test_flow_rate}")
    
    # Find the element where test_flow_rate is located
    # flow_rates is in ascending order, so we can find the interval
    if test_flow_rate <= flow_rates[0]:
        # Test flow rate is at or below the minimum
        left_idx = 0
        right_idx = 0
        left_distance = 1.0
        right_distance = 0.0
        print(f"Test flow rate is at or below minimum: using index {left_idx}")
    elif test_flow_rate >= flow_rates[-1]:
        # Test flow rate is at or above the maximum
        left_idx = len(flow_rates) - 1
        right_idx = len(flow_rates) - 1
        left_distance = 0.0
        right_distance = 1.0
        print(f"Test flow rate is at or above maximum: using index {right_idx}")
    else:
        # Test flow rate is within the range, find the interval
        for i in range(len(flow_rates) - 1):
            if flow_rates[i] <= test_flow_rate <= flow_rates[i + 1]:
                left_idx = i
                right_idx = i + 1
                
                # Calculate relative distances (linear interpolation)
                total_interval = flow_rates[right_idx] - flow_rates[left_idx]
                right_distance = (flow_rates[right_idx] - test_flow_rate) / total_interval
                left_distance = (test_flow_rate - flow_rates[left_idx]) / total_interval
                
                print(f"Test flow rate is in interval [{flow_rates[left_idx]:.6f}, {flow_rates[right_idx]:.6f}]")
                print(f"Left node index: {left_idx}, Right node index: {right_idx}")
                print(f"Left distance: {left_distance:.6f}, Right distance: {right_distance:.6f}")
                print(f"Sum of distances: {left_distance + right_distance:.6f}")
                break
    
    # Store interpolation results
    interpolation_result = {
        'left_idx': left_idx,
        'right_idx': right_idx,
        'left_distance': left_distance,
        'right_distance': right_distance,
        'left_flow_rate': flow_rates[left_idx],
        'right_flow_rate': flow_rates[right_idx],
        'test_flow_rate': test_flow_rate
    }
    
    print(f"Interpolation result: {interpolation_result}")
    return interpolation_result


if __name__ == "__main__":

    # train_filename = 'all_concentration_data_train10'
    train_filename = 'all_concentration_data_train20'
    test_filename = 'all_concentration_data_test'

    # Convert data to tensor
    convert_data_to_tensor(train_filename, test_filename)

    train_tensor, train_nodes, test_tensor, test_nodes = load_tensors_from_files(train_filename, test_filename) # (nnode, ntime, nflow), (nnode, ntime)
    
    # Check if node dictionaries are loaded successfully
    if train_nodes is None or test_nodes is None:
        print("Error: Node dictionaries could not be loaded. Exiting.")
        exit(1)
    flow_rates = train_nodes['flowrates']
    test_flow_rate = test_nodes['flowrates']
    
    # Perform 1D linear interpolation
    interpolation_result = interpolate_flow_rate(flow_rates, test_flow_rate)

    # Normalize tensors if they exist
    train_tensor, test_tensor, norm_params = normalize_tensors(train_tensor, test_tensor)
    
    # Perform CP decomposition on train_tensor if it exists
    print("\n" + "="*50)
    print("CP TENSOR DECOMPOSITION")
    print("="*50)
    
    # Choose a reasonable rank for decomposition
    # A common heuristic is to use a rank that is smaller than the minimum dimension
    min_dim = min(train_tensor.shape)
    # suggested_rank = min(10, min_dim // 2)  # Conservative choice
    suggested_rank = 10
    print(f"Suggested rank: {suggested_rank}")
    
    # Perform CP decomposition
    weights, factors, error_metrics = cp_decomposition(train_tensor, rank=suggested_rank)
    
    # Save decomposition results
    decomposition_results = {
        'weights': weights,
        'factors': factors,
        'rank': suggested_rank,
        'tensor_shape': train_tensor.shape,
        'error_metrics': error_metrics
    }
    
    # np.save('./data/cp_decomposition_results.npy', decomposition_results)
    # print(f"CP decomposition results saved to ./data/cp_decomposition_results.npy")
    
    # # Display some statistics about the factors
    # print("\nFactor Statistics:")
    # for i, factor in enumerate(factors):
    #     print(f"Factor {i+1}: mean={factor.mean():.4f}, std={factor.std():.4f}, "
    #           f"min={factor.min():.4f}, max={factor.max():.4f}")
    
    # # Analyze decomposition quality
    # quality = analyze_decomposition_quality(train_tensor, error_metrics['reconstructed'], error_metrics)
    # print(f"\nDecomposition Quality: {quality}")


    ################ Check test accuracy ################
    # Replace the factor matrix of the flow rate with a fixed value
    factors_test = factors.copy()
    factors_low = factors[2][[interpolation_result['left_idx']],:]
    factors_high = factors[2][[interpolation_result['right_idx']],:]
    factors_test[2] = (factors_low * interpolation_result['left_distance'] + 
                        factors_high * interpolation_result['right_distance'])
    
    reconstructed_test = tl.cp_to_tensor((weights, factors_test)).squeeze()

    # Calculate error metrics
    mse = np.mean((test_tensor - reconstructed_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(test_tensor - reconstructed_test))

    # Print error metrics
    print(f"CP Decomposition (Test Set) Error Metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Save the trained model
    model_test = {
        'train_nodes': train_nodes,
        'test_nodes': test_nodes,
        'weights': weights,
        'factors': factors_test,
        'norm_params': norm_params,
        # 'interpolation_result': interpolation_result,
    }
    
    # Create model_saved directory if it doesn't exist
    model_dir = './pyinn/model_saved'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f'LAM_{train_filename}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_test, f)
    print(f"Model test saved to {model_path}")

    
    ################ Check test accuracy ################  
    # 1. Denormalize test tensor
    # 2. Reconstruct test tensor
    # 3. Calculate error metrics
    # 4. Print error metrics
    # 5. Plot original vs reconstructed test tensor
    


