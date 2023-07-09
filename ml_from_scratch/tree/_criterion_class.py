import numpy as np

# ----------------------
# Criterion Recommended by Journal
# ----------------------

def _gini_index(self, y):
    """
    Calculate the Gini index for a given target values.

    Parameters:
    - y (array-like, shape (n_samples,)): The target values.

    Returns:
    - gini (float): The calculated Gini index.
    """
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    gini = 1.0 - np.sum(class_probs ** 2)
    return gini

# ----------------------
# Additional Criterion
# ----------------------

def _entropy(self, y):
    """
    Calculate the entropy for a given target values.

    Parameters:
    - y (array-like, shape (n_samples,)): The target values.

    Returns:
    - entropy (float): The calculated entropy.
    """
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    entropy = -np.sum(class_probs * np.log2(class_probs + 1e-10))
    return entropy

def _log_loss(self, y):
    """
    Calculate the logarithmic loss for a given target values.

    Parameters:
    - y (array-like, shape (n_samples,)): The target values.

    Returns:
    - log_loss (float): The calculated logarithmic loss.
    """
    class_counts = np.bincount(y)
    class_probs = class_counts / len(y)
    max_prob = np.max(class_probs)
    log_loss = -np.log(max_prob)
    return log_loss

def _mean_absolute_error(self, y):
    """
    Calculate the mean absolute error (MAE) for a given target values.

    Parameters:
    - y (array-like, shape (n_samples,)): The target values.

    Returns:
    - mae (float): The calculated mean absolute error.
    """
    mean = np.mean(y)
    mae = np.mean(np.abs(y - mean))
    return mae

def _mean_squared_error(self, y):
    """
    Calculate the mean squared error (MSE) for a given target values.

    Parameters:
    - y (array-like, shape (n_samples,)): The target values.

    Returns:
    - mse (float): The calculated mean squared error.
    """
    mean = np.mean(y)
    mse = np.mean((y - mean) ** 2)
    return mse
