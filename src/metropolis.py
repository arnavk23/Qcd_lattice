"""
Metropolis Algorithm for Gaussian Distribution

This module implements the Metropolis algorithm to sample from a Gaussian distribution.
This serves as a foundational implementation to understand MCMC methods before
applying them to field theory problems.

References:
- Quantum Chromodynamics on the Lattice, Chapter 1
- MCMC for Dummies
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None


class MetropolisGaussian:
    """
    Metropolis algorithm implementation for sampling from a Gaussian distribution.
    
    This class implements the Metropolis algorithm to generate samples from a
    Gaussian distribution N(mu, sigma^2) using only the unnormalized probability
    density function.
    """
    
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        """
        Initialize the Metropolis sampler for Gaussian distribution.
        
        Args:
            mu: Mean of the Gaussian distribution
            sigma: Standard deviation of the Gaussian distribution
        """
        self.mu = mu
        self.sigma = sigma
        self.samples = []
        self.accepted = 0
        self.total_steps = 0
        
    def log_probability(self, x: float) -> float:
        """
        Compute the log probability density (up to normalization constant).
        
        For Gaussian: log P(x) = -0.5 * ((x - mu) / sigma)^2
        We ignore the normalization constant since it cancels in the Metropolis ratio.
        
        Args:
            x: Point at which to evaluate the log probability
            
        Returns:
            Log probability density (unnormalized)
        """
        return -0.5 * ((x - self.mu) / self.sigma)**2
    
    def metropolis_step(self, current_x: float, step_size: float) -> Tuple[float, bool]:
        """
        Perform one Metropolis step.
        
        Args:
            current_x: Current state
            step_size: Step size for proposal distribution
            
        Returns:
            Tuple of (new_state, accepted_flag)
        """
        # Propose new state using uniform random walk
        proposal = current_x + np.random.uniform(-step_size, step_size)
        
        # Compute acceptance probability
        log_alpha = self.log_probability(proposal) - self.log_probability(current_x)
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.random() < alpha:
            return proposal, True
        else:
            return current_x, False
    
    def sample(self, n_samples: int, step_size: float = 1.0, 
               initial_x: Optional[float] = None, burn_in: int = 1000) -> np.ndarray:
        """
        Generate samples using the Metropolis algorithm.
        
        Args:
            n_samples: Number of samples to generate
            step_size: Step size for proposal distribution
            initial_x: Initial state (if None, starts at 0)
            burn_in: Number of burn-in steps to discard
            
        Returns:
            Array of samples
        """
        if initial_x is None:
            initial_x = 0.0
            
        # Reset counters
        self.accepted = 0
        self.total_steps = 0
        
        samples = []
        current_x = initial_x
        
        # Burn-in phase
        for _ in range(burn_in):
            current_x, accepted = self.metropolis_step(current_x, step_size)
            if accepted:
                self.accepted += 1
            self.total_steps += 1
        
        # Sampling phase
        for _ in range(n_samples):
            current_x, accepted = self.metropolis_step(current_x, step_size)
            if accepted:
                self.accepted += 1
            self.total_steps += 1
            samples.append(current_x)
        
        self.samples = np.array(samples)
        return self.samples
    
    def acceptance_rate(self) -> float:
        """Get the acceptance rate of the Metropolis algorithm."""
        return self.accepted / self.total_steps if self.total_steps > 0 else 0.0
    
    def autocorrelation_time(self) -> float:
        """
        Estimate the autocorrelation time of the samples.
        
        Returns:
            Estimated autocorrelation time
        """
        if len(self.samples) < 100:
            return np.nan
        
        # Compute autocorrelation function
        samples_centered = self.samples - np.mean(self.samples)
        correlation = np.correlate(samples_centered, samples_centered, mode='full')
        correlation = correlation[len(correlation)//2:]
        correlation = correlation / correlation[0]
        
        # Find where autocorrelation drops below 1/e
        try:
            tau_idx = np.where(correlation < 1/np.e)[0][0]
            return tau_idx
        except IndexError:
            return len(correlation)
    
    def plot_results(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot the results of the Metropolis sampling.
        
        Args:
            figsize: Figure size for the plots
        """
        if len(self.samples) == 0:
            print("No samples available. Run sample() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Trace plot
        axes[0, 0].plot(self.samples)
        axes[0, 0].set_title('Trace Plot')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('x')
        
        # Plot 2: Histogram vs theoretical
        axes[0, 1].hist(self.samples, bins=50, density=True, alpha=0.7, 
                       color='skyblue', label='Samples')
        
        # Theoretical Gaussian
        x_theory = np.linspace(self.samples.min(), self.samples.max(), 100)
        y_theory = (1 / (self.sigma * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((x_theory - self.mu) / self.sigma)**2)
        axes[0, 1].plot(x_theory, y_theory, 'r-', linewidth=2, label='Theoretical')
        axes[0, 1].set_title('Histogram vs Theoretical')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # Plot 3: Autocorrelation
        samples_centered = self.samples - np.mean(self.samples)
        max_lag = min(len(self.samples) // 4, 100)
        lags = np.arange(max_lag)
        autocorr = []
        
        for lag in lags:
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(samples_centered[:-lag], samples_centered[lag:])[0, 1]
                autocorr.append(corr)
        
        axes[1, 0].plot(lags, autocorr)
        axes[1, 0].axhline(y=1/np.e, color='r', linestyle='--', label='1/e')
        axes[1, 0].set_title('Autocorrelation Function')
        axes[1, 0].set_xlabel('Lag')
        axes[1, 0].set_ylabel('Autocorrelation')
        axes[1, 0].legend()
        
        # Plot 4: Running average
        running_mean = np.cumsum(self.samples) / np.arange(1, len(self.samples) + 1)
        axes[1, 1].plot(running_mean)
        axes[1, 1].axhline(y=self.mu, color='r', linestyle='--', label=f'True mean = {self.mu}')
        axes[1, 1].set_title('Running Average')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Running Mean')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def print_diagnostics(self):
        """Print diagnostic information about the sampling."""
        if len(self.samples) == 0:
            print("No samples available.")
            return
        
        print(f"Metropolis Gaussian Sampling Diagnostics")
        print(f"========================================")
        print(f"Target distribution: N({self.mu}, {self.sigma}²)")
        print(f"Number of samples: {len(self.samples)}")
        print(f"Acceptance rate: {self.acceptance_rate():.3f}")
        print(f"")
        print(f"Sample Statistics:")
        print(f"  Mean: {np.mean(self.samples):.4f} (target: {self.mu})")
        print(f"  Std:  {np.std(self.samples):.4f} (target: {self.sigma})")
        print(f"  Min:  {np.min(self.samples):.4f}")
        print(f"  Max:  {np.max(self.samples):.4f}")
        print(f"")
        print(f"Autocorrelation time: {self.autocorrelation_time():.2f}")


def demonstrate_step_size_effect():
    """
    Demonstrate the effect of different step sizes on the Metropolis algorithm.
    """
    step_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]
    n_samples = 5000
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, step_size in enumerate(step_sizes):
        sampler = MetropolisGaussian(mu=0.0, sigma=1.0)
        samples = sampler.sample(n_samples, step_size=step_size)
        
        if i < len(axes):
            axes[i].plot(samples[:1000])  # Plot first 1000 samples
            axes[i].set_title(f'Step size = {step_size}\nAcceptance = {sampler.acceptance_rate():.3f}')
            axes[i].set_xlabel('Iteration')
            axes[i].set_ylabel('x')
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing Metropolis algorithm for Gaussian distribution...")
    
    # Create sampler
    sampler = MetropolisGaussian(mu=2.0, sigma=1.5)
    
    # Generate samples
    samples = sampler.sample(n_samples=10000, step_size=1.0)
    
    # Print diagnostics
    sampler.print_diagnostics()
    
    # Plot results
    sampler.plot_results()
    
    # Demonstrate step size effect
    print("\nDemonstrating step size effect...")
    demonstrate_step_size_effect()
