"""
Utility functions for lattice field theory implementations.

This module contains common utility functions used across different
Monte Carlo implementations for lattice field theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None


def autocorrelation_function(data: np.ndarray, max_lag: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the autocorrelation function of a time series.
    
    Args:
        data: Time series data
        max_lag: Maximum lag to compute (default: len(data)//4)
        
    Returns:
        Tuple of (lags, autocorrelation_values)
    """
    if max_lag is None:
        max_lag = len(data) // 4
    
    data_centered = data - np.mean(data)
    lags = np.arange(max_lag + 1)  # Include max_lag itself
    autocorr = []
    
    for lag in lags:
        if lag == 0:
            autocorr.append(1.0)
        else:
            if len(data_centered) > lag:
                corr = np.corrcoef(data_centered[:-lag], data_centered[lag:])[0, 1]
                autocorr.append(corr)
            else:
                autocorr.append(0.0)
    
    return lags, np.array(autocorr)


def integrated_autocorrelation_time(data: np.ndarray, W: Optional[int] = None) -> float:
    """
    Compute the integrated autocorrelation time.
    
    Args:
        data: Time series data
        W: Window size (default: automatic windowing)
        
    Returns:
        Integrated autocorrelation time
    """
    lags, autocorr = autocorrelation_function(data, max_lag=len(data)//4)
    
    if W is None:
        # Automatic windowing: find first lag where autocorr < 0
        try:
            W = np.where(autocorr <= 0)[0][0]
        except IndexError:
            W = len(autocorr)
    
    # Integrated autocorrelation time
    tau_int = 1 + 2 * np.sum(autocorr[1:W])
    return tau_int


def effective_sample_size(data: np.ndarray) -> float:
    """
    Compute the effective sample size.
    
    Args:
        data: Time series data
        
    Returns:
        Effective sample size
    """
    tau_int = integrated_autocorrelation_time(data)
    return len(data) / (2 * tau_int)


def jackknife_error(data: np.ndarray, func: Callable = np.mean) -> Tuple[float, float]:
    """
    Compute jackknife error estimate.
    
    Args:
        data: Time series data
        func: Function to compute statistic (default: mean)
        
    Returns:
        Tuple of (statistic, error)
    """
    n = len(data)
    full_stat = func(data)
    
    jackknife_stats = []
    for i in range(n):
        # Remove i-th element
        jackknife_sample = np.concatenate([data[:i], data[i+1:]])
        jackknife_stats.append(func(jackknife_sample))
    
    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats)
    jackknife_error = np.sqrt((n - 1) / n * np.sum((jackknife_stats - jackknife_mean)**2))
    
    return full_stat, jackknife_error


def bootstrap_error(data: np.ndarray, func: Callable = np.mean, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap error estimate.
    
    Args:
        data: Time series data
        func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (statistic, error)
    """
    n = len(data)
    full_stat = func(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    bootstrap_error = np.std(bootstrap_stats)
    
    return full_stat, bootstrap_error


def binning_analysis(data: np.ndarray, max_bin_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform binning analysis to estimate autocorrelation time.
    
    Args:
        data: Time series data
        max_bin_size: Maximum bin size (default: len(data)//4)
        
    Returns:
        Tuple of (bin_sizes, standard_errors)
    """
    if max_bin_size is None:
        max_bin_size = len(data) // 4
    
    bin_sizes = []
    standard_errors = []
    
    for bin_size in range(1, max_bin_size + 1):
        # Number of complete bins
        n_bins = len(data) // bin_size
        
        if n_bins < 2:
            break
        
        # Create binned data
        binned_data = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size
            bin_mean = np.mean(data[start_idx:end_idx])
            binned_data.append(bin_mean)
        
        binned_data = np.array(binned_data)
        
        # Standard error of binned data
        se = np.std(binned_data) / np.sqrt(len(binned_data))
        
        bin_sizes.append(bin_size)
        standard_errors.append(se)
    
    return np.array(bin_sizes), np.array(standard_errors)


def plot_mcmc_diagnostics(samples: np.ndarray, title: str = "MCMC Diagnostics", 
                         true_value: Optional[float] = None, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot comprehensive MCMC diagnostics.
    
    Args:
        samples: MCMC samples
        title: Plot title
        true_value: True value for comparison (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Trace plot
    axes[0, 0].plot(samples)
    axes[0, 0].set_title('Trace Plot')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Value')
    if true_value is not None:
        axes[0, 0].axhline(y=true_value, color='r', linestyle='--', label=f'True value = {true_value}')
        axes[0, 0].legend()
    
    # Histogram
    axes[0, 1].hist(samples, bins=50, density=True, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Histogram')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    if true_value is not None:
        axes[0, 1].axvline(x=true_value, color='r', linestyle='--', label=f'True value = {true_value}')
        axes[0, 1].legend()
    
    # Autocorrelation
    lags, autocorr = autocorrelation_function(samples)
    axes[1, 0].plot(lags, autocorr)
    axes[1, 0].axhline(y=1/np.e, color='r', linestyle='--', label='1/e')
    axes[1, 0].set_title('Autocorrelation Function')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].legend()
    
    # Running average
    running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    axes[1, 1].plot(running_mean)
    axes[1, 1].set_title('Running Average')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Running Mean')
    if true_value is not None:
        axes[1, 1].axhline(y=true_value, color='r', linestyle='--', label=f'True value = {true_value}')
        axes[1, 1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_binning_analysis(data: np.ndarray, title: str = "Binning Analysis"):
    """
    Plot binning analysis results.
    
    Args:
        data: Time series data
        title: Plot title
    """
    bin_sizes, standard_errors = binning_analysis(data)
    
    plt.figure(figsize=(8, 6))
    plt.plot(bin_sizes, standard_errors, 'o-')
    plt.xlabel('Bin Size')
    plt.ylabel('Standard Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def thermalization_test(data: np.ndarray, n_blocks: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Test for thermalization by comparing different blocks of the data.
    
    Args:
        data: Time series data
        n_blocks: Number of blocks to divide the data into
        
    Returns:
        Tuple of (block_means, block_errors)
    """
    block_size = len(data) // n_blocks
    block_means = []
    block_errors = []
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        block_data = data[start_idx:end_idx]
        
        block_mean = np.mean(block_data)
        block_error = np.std(block_data) / np.sqrt(len(block_data))
        
        block_means.append(block_mean)
        block_errors.append(block_error)
    
    return np.array(block_means), np.array(block_errors)


def plot_thermalization_test(data: np.ndarray, n_blocks: int = 10, title: str = "Thermalization Test"):
    """
    Plot thermalization test results.
    
    Args:
        data: Time series data
        n_blocks: Number of blocks to divide the data into
        title: Plot title
    """
    block_means, block_errors = thermalization_test(data, n_blocks)
    
    plt.figure(figsize=(10, 6))
    block_indices = np.arange(n_blocks)
    plt.errorbar(block_indices, block_means, yerr=block_errors, fmt='o-', capsize=5)
    plt.xlabel('Block Index')
    plt.ylabel('Block Mean')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def gelman_rubin_diagnostic(chains: List[np.ndarray]) -> float:
    """
    Compute the Gelman-Rubin diagnostic (R-hat) for multiple chains.
    
    Args:
        chains: List of MCMC chains
        
    Returns:
        R-hat statistic
    """
    n_chains = len(chains)
    n_samples = len(chains[0])
    
    # Chain means
    chain_means = [np.mean(chain) for chain in chains]
    overall_mean = np.mean(chain_means)
    
    # Within-chain variance
    within_chain_var = np.mean([np.var(chain) for chain in chains])
    
    # Between-chain variance
    between_chain_var = n_samples / (n_chains - 1) * np.sum([(mean - overall_mean)**2 for mean in chain_means])
    
    # Pooled variance estimate
    pooled_var = ((n_samples - 1) / n_samples) * within_chain_var + (1 / n_samples) * between_chain_var
    
    # R-hat
    r_hat = np.sqrt(pooled_var / within_chain_var)
    
    return r_hat


if __name__ == "__main__":
    # Test utility functions with synthetic data
    print("Testing utility functions...")
    
    # Generate synthetic autocorrelated data
    np.random.seed(42)
    n = 1000
    autocorr_time = 10
    
    # AR(1) process
    phi = np.exp(-1/autocorr_time)
    data = np.zeros(n)
    data[0] = np.random.normal()
    
    for i in range(1, n):
        data[i] = phi * data[i-1] + np.sqrt(1 - phi**2) * np.random.normal()
    
    # Test functions
    tau_int = integrated_autocorrelation_time(data)
    eff_size = effective_sample_size(data)
    mean_val, jack_err = jackknife_error(data)
    
    print(f"Integrated autocorrelation time: {tau_int:.2f}")
    print(f"Effective sample size: {eff_size:.2f}")
    print(f"Mean ± jackknife error: {mean_val:.4f} ± {jack_err:.4f}")
    
    # Plot diagnostics
    plot_mcmc_diagnostics(data, title="Test Data Diagnostics")
    plot_binning_analysis(data, title="Test Data Binning Analysis")
