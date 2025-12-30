"""
INN Plotting Utilities
----------------------------------------------------------------------------------
Simple plotting functions for training loss visualization.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_loss(model, data_name, interp_method, save_dir='plots'):
    """
    Plot training and validation loss vs epoch.

    Args:
        model: Trained model with loss_history or errors_train/errors_val attributes
        data_name: Name of the dataset (for filename)
        interp_method: Method name (for filename)
        save_dir: Directory to save plots (default: 'plots')
    """
    # Create plots directory if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get loss history from model (support both old and new attribute names)
    if hasattr(model, 'loss_history') and model.loss_history:
        train_loss = model.loss_history
        epochs = list(range(1, len(train_loss) + 1))
        val_loss = getattr(model, 'val_loss_history', None)
    elif hasattr(model, 'errors_train') and model.errors_train:
        train_loss = model.errors_train
        val_loss = getattr(model, 'errors_val', None)
        epochs = getattr(model, 'errors_epoch', list(range(1, len(train_loss) + 1)))
    else:
        print("Warning: No loss history found in model")
        return

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])

    # Plot training loss
    ax.plot(epochs, train_loss, '-', color='#2563eb', linewidth=2, label='Training loss')

    # Plot validation loss if available
    if val_loss is not None and len(val_loss) > 0:
        # Handle case where val_loss may have different length than epochs
        if len(val_loss) == len(epochs):
            ax.plot(epochs, val_loss, '--', color='#dc2626', linewidth=2, label='Validation loss')
        else:
            # Validation computed at different intervals
            val_epochs = np.linspace(epochs[0], epochs[-1], len(val_loss))
            ax.plot(val_epochs, val_loss, '--', color='#dc2626', linewidth=2, label='Validation loss')

    # Formatting
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(0, epochs[-1])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-limits
    all_losses = list(train_loss)
    if val_loss is not None:
        all_losses.extend(val_loss)
    min_loss = min(all_losses) if all_losses else 1e-4
    max_loss = max(all_losses) if all_losses else 1e0

    y_min = max(min_loss * 0.5, 1e-8)
    y_max = min(max_loss * 2, 1e2)
    ax.set_ylim(y_min, y_max)

    ax.legend(shadow=True, borderpad=1, fontsize=12, loc='best')
    ax.set_title(f'{data_name} - {interp_method}', fontsize=14)
    plt.tight_layout()

    # Save figure
    n_epochs = epochs[-1]
    filename = f"{data_name}_{interp_method}_loss_{n_epochs}epoch.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {filepath}")

    plt.show()
    plt.close()


def plot_regression(model, cls_data, config):
    """
    Plot results for regression models.
    Only plots training loss vs epoch.

    Args:
        model: Trained regression model
        cls_data: Data object with data_name attribute
        config: Configuration dictionary with PLOT settings
    """
    bool_plot = config.get('PLOT', {}).get('bool_plot', True)

    if bool_plot:
        data_name = getattr(cls_data, 'data_name', config.get('data_name', 'unknown'))
        interp_method = getattr(model, 'interp_method', config.get('interp_method', 'unknown'))
        plot_loss(model, data_name, interp_method)
    else:
        print("\nPlotting deactivated\n")


def plot_classification(model, cls_data, config):
    """
    Plot results for classification models.
    Only plots training loss vs epoch.

    Args:
        model: Trained classification model
        cls_data: Data object with data_name attribute
        config: Configuration dictionary with PLOT settings
    """
    bool_plot = config.get('PLOT', {}).get('bool_plot', True)

    if bool_plot:
        data_name = getattr(cls_data, 'data_name', config.get('data_name', 'unknown'))
        interp_method = getattr(model, 'interp_method', config.get('interp_method', 'unknown'))
        plot_loss(model, data_name, interp_method)
    else:
        print("\nPlotting deactivated\n")


# Legacy alias for backward compatibility
plot_loss_landscape = plot_loss
