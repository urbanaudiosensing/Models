from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from torchmetrics.functional.regression import concordance_corrcoef, pearson_corrcoef, spearman_corrcoef
import matplotlib.pyplot as plt
import numpy as np
import torch

def regression_report(y_true: list[float], y_pred: list[float], *args, **kwargs):
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print(y_true)
        print(y_pred)
    regression_plot(y_true, y_pred)
    return {'R2': r2_score(y_true, y_pred), 
            'MSE': mean_squared_error(y_true, y_pred),
            'Concordance': concordance_correlation(y_true, y_pred),
            'Pearson': pearson_correlation(y_true, y_pred),
            'Spearman': spearman_correlation(y_true, y_pred),
            'Support': len(y_true)}

def torchmetrics_func(y_true, y_pred, func):
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
    return func(y_pred, y_true)

def concordance_correlation(y_true, y_pred):
    return torchmetrics_func(y_true, y_pred, concordance_corrcoef)

def pearson_correlation(y_true, y_pred):
    return torchmetrics_func(y_true, y_pred, pearson_corrcoef)

def spearman_correlation(y_true, y_pred):
    return torchmetrics_func(y_true, y_pred, spearman_corrcoef)



def regression_plot(y_true, y_pred, save_path="plot.png"):
    """
    Create a scatter plot for regression and save it to a file.

    Parameters:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    - save_path: File path to save the plot. Default is "plot.png".

    Returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.7, label='Predictions')
    
    # Add labels and title
    plt.title('Regression Plot')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')

    # Add a diagonal line for reference (perfect predictions)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=2, label='Ideal')

    plt.grid(True)
    
    # Save the plot to the specified file path
    plt.savefig(save_path)
    plt.close()
