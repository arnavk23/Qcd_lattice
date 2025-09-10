"""
In-Depth Metropolis Algorithm Demo

This script provides a comprehensive demonstration of the Metropolis algorithm
using the harmonic oscillator implementation, including:
1. Step size optimization
2. Autocorrelation analysis
3. Convergence studies
4. Error analysis with jackknife and bootstrap
5. Comparison with theoretical predictions
6. Performance analysis

All plots are saved to plots/metropolis/
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from harmonic_oscillator import HarmonicOscillatorMC as HarmonicOscillator
from utils import *

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

def create_plot_dir():
    """Create plot directory if it doesn't exist."""
    plot_dir = Path("plots/metropolis")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def step_size_optimization_demo():
    """Study optimal step size for Metropolis algorithm."""
    print("=" * 60)
    print("STEP SIZE OPTIMIZATION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test range of step sizes
    step_sizes = np.logspace(-2, 1, 25)  # From 0.01 to 10.0
    
    results = []
    
    for step_size in step_sizes:
        print(f"\\nTesting step size: {step_size:.4f}")
        
        # Create harmonic oscillator
        ho = HarmonicOscillator(n_time_steps=50, omega=1.0, lattice_spacing=step_size)
        
        # Run simulation
        n_samples = 2000
        burn_in = 500
        
        results_dict = ho.run_simulation(n_samples, step_size=step_size, burn_in=burn_in)
        
        # Extract position samples
        samples = np.array(results_dict['observables']['position'])
        
        # Calculate statistics
        acceptance_rate = results_dict['acceptance_rate']
        
        # Calculate autocorrelation time
        autocorr_time = integrated_autocorrelation_time(samples)
        
        # Calculate effective sample size
        effective_samples = len(samples) / (2 * autocorr_time + 1)
        
        results.append({
            'step_size': step_size,
            'acceptance_rate': acceptance_rate,
            'autocorr_time': autocorr_time,
            'effective_samples': effective_samples,
            'samples': samples[-500:]  # Keep last 500 samples
        })
        
        print(f"  Acceptance rate: {acceptance_rate:.3f}")
        print(f"  Autocorrelation time: {autocorr_time:.2f}")
        print(f"  Effective samples: {effective_samples:.1f}")
    
    # Create optimization plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    step_values = [r['step_size'] for r in results]
    acceptance_values = [r['acceptance_rate'] for r in results]
    autocorr_values = [r['autocorr_time'] for r in results]
    effective_values = [r['effective_samples'] for r in results]
    
    # Acceptance rate vs step size
    axes[0, 0].semilogx(step_values, acceptance_values, 'bo-', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Optimal range')
    axes[0, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Step Size')
    axes[0, 0].set_ylabel('Acceptance Rate')
    axes[0, 0].set_title('Acceptance Rate vs Step Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Autocorrelation time vs step size
    axes[0, 1].loglog(step_values, autocorr_values, 'ro-', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Step Size')
    axes[0, 1].set_ylabel('Autocorrelation Time')
    axes[0, 1].set_title('Autocorrelation Time vs Step Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Effective sample size vs step size
    axes[1, 0].semilogx(step_values, effective_values, 'go-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Step Size')
    axes[1, 0].set_ylabel('Effective Sample Size')
    axes[1, 0].set_title('Effective Sample Size vs Step Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency metric
    efficiency = np.array(effective_values) / np.array(autocorr_values)
    axes[1, 1].semilogx(step_values, efficiency, 'mo-', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Step Size')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].set_title('Sampling Efficiency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Metropolis Step Size Optimization', fontsize=16)
    plt.tight_layout()
    
    step_path = plot_dir / "step_size_optimization.png"
    plt.savefig(step_path, dpi=300, bbox_inches='tight')
    print(f"\\nStep size optimization plot saved to: {step_path}")
    plt.close()
    
    # Find optimal step size
    optimal_idx = np.argmax(efficiency)
    optimal_step = step_values[optimal_idx]
    
    return results, optimal_step

def convergence_study_demo(optimal_step):
    """Study convergence behavior."""
    print("\\n" + "=" * 60)
    print("CONVERGENCE STUDY DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test different numbers of samples
    sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
    
    results = []
    
    for n_samples in sample_sizes:
        print(f"\\nTesting {n_samples} samples")
        
        # Create harmonic oscillator
        ho = HarmonicOscillator(n_time_steps=50, omega=1.0, lattice_spacing=optimal_step)
        
        # Run simulation
        burn_in = min(1000, n_samples // 4)
        
        start_time = time.time()
        results_dict = ho.run_simulation(n_samples, step_size=0.5, burn_in=burn_in)
        computation_time = time.time() - start_time
        
        # Extract position samples
        samples = np.array(results_dict['observables']['position'])
        
        # Calculate observables
        x_squared = np.mean(samples**2)
        x_fourth = np.mean(samples**4)
        
        # Calculate errors using jackknife
        _, x_squared_error = jackknife_error(samples**2)
        _, x_fourth_error = jackknife_error(samples**4)
        
        # Calculate autocorrelation time
        autocorr_time = integrated_autocorrelation_time(samples)
        
        results.append({
            'n_samples': n_samples,
            'x_squared': x_squared,
            'x_squared_error': x_squared_error,
            'x_fourth': x_fourth,
            'x_fourth_error': x_fourth_error,
            'autocorr_time': autocorr_time,
            'computation_time': computation_time,
            'acceptance_rate': results_dict['acceptance_rate']
        })
        
        print(f"  ⟨x²⟩ = {x_squared:.4f} ± {x_squared_error:.4f}")
        print(f"  ⟨x⁴⟩ = {x_fourth:.4f} ± {x_fourth_error:.4f}")
        print(f"  Autocorr time: {autocorr_time:.2f}")
        print(f"  Computation time: {computation_time:.2f}s")
    
    # Theoretical values
    theoretical_x_squared = 0.5
    theoretical_x_fourth = 0.75
    
    # Create convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    n_values = [r['n_samples'] for r in results]
    x_squared_values = [r['x_squared'] for r in results]
    x_squared_errors = [r['x_squared_error'] for r in results]
    x_fourth_values = [r['x_fourth'] for r in results]
    x_fourth_errors = [r['x_fourth_error'] for r in results]
    
    # ⟨x²⟩ convergence
    axes[0, 0].errorbar(n_values, x_squared_values, yerr=x_squared_errors, 
                       fmt='bo-', alpha=0.7, linewidth=2, capsize=5)
    axes[0, 0].axhline(y=theoretical_x_squared, color='red', linestyle='--', 
                      alpha=0.5, label=f'Theoretical: {theoretical_x_squared}')
    axes[0, 0].set_xlabel('Number of Samples')
    axes[0, 0].set_ylabel('⟨x²⟩')
    axes[0, 0].set_title('Convergence of ⟨x²⟩')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # ⟨x⁴⟩ convergence
    axes[0, 1].errorbar(n_values, x_fourth_values, yerr=x_fourth_errors, 
                       fmt='ro-', alpha=0.7, linewidth=2, capsize=5)
    axes[0, 1].axhline(y=theoretical_x_fourth, color='red', linestyle='--', 
                      alpha=0.5, label=f'Theoretical: {theoretical_x_fourth}')
    axes[0, 1].set_xlabel('Number of Samples')
    axes[0, 1].set_ylabel('⟨x⁴⟩')
    axes[0, 1].set_title('Convergence of ⟨x⁴⟩')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error scaling
    axes[1, 0].loglog(n_values, x_squared_errors, 'go-', alpha=0.7, linewidth=2, label='⟨x²⟩ error')
    axes[1, 0].loglog(n_values, x_fourth_errors, 'mo-', alpha=0.7, linewidth=2, label='⟨x⁴⟩ error')
    
    # Theoretical 1/√N scaling
    theoretical_scaling = x_squared_errors[0] * np.sqrt(n_values[0] / np.array(n_values))
    axes[1, 0].loglog(n_values, theoretical_scaling, 'k--', alpha=0.5, label='1/√N scaling')
    
    axes[1, 0].set_xlabel('Number of Samples')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].set_title('Error Scaling')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Computation time scaling
    comp_times = [r['computation_time'] for r in results]
    axes[1, 1].loglog(n_values, comp_times, 'co-', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Number of Samples')
    axes[1, 1].set_ylabel('Computation Time (s)')
    axes[1, 1].set_title('Computational Scaling')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Study', fontsize=16)
    plt.tight_layout()
    
    conv_path = plot_dir / "convergence_study.png"
    plt.savefig(conv_path, dpi=300, bbox_inches='tight')
    print(f"\\nConvergence study plot saved to: {conv_path}")
    plt.close()
    
    return results

def autocorrelation_analysis_demo(optimal_step):
    """Detailed autocorrelation analysis."""
    print("\\n" + "=" * 60)
    print("AUTOCORRELATION ANALYSIS DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create harmonic oscillator
    ho = HarmonicOscillator(n_time_steps=50, omega=1.0, lattice_spacing=optimal_step)
    
    # Run long simulation
    n_samples = 10000
    burn_in = 1000
    
    print(f"Running {n_samples} samples for autocorrelation analysis...")
    results_dict = ho.run_simulation(n_samples, step_size=0.5, burn_in=burn_in)
    
    # Extract position samples
    samples = np.array(results_dict['observables']['position'])
    
    # Calculate autocorrelation function
    max_lag = 200
    autocorr_func, lags = autocorrelation_function(samples, max_lag)
    
    # Calculate integrated autocorrelation time
    autocorr_time = integrated_autocorrelation_time(samples)
    
    # Calculate windowed autocorrelation times
    window_sizes = [50, 100, 200, 500, 1000, 2000]
    windowed_autocorr = []
    
    for window_size in window_sizes:
        if window_size <= len(samples):
            windowed_time = integrated_autocorrelation_time(samples[:window_size])
            windowed_autocorr.append(windowed_time)
        else:
            windowed_autocorr.append(np.nan)
    
    # Create autocorrelation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Autocorrelation function
    axes[0, 0].plot(lags, autocorr_func, 'b-', linewidth=2, alpha=0.7)
    axes[0, 0].axhline(y=1/np.e, color='red', linestyle='--', alpha=0.5, label='1/e')
    axes[0, 0].axvline(x=autocorr_time, color='green', linestyle='--', alpha=0.5, 
                      label=f'τ = {autocorr_time:.2f}')
    axes[0, 0].set_xlabel('Lag')
    axes[0, 0].set_ylabel('Autocorrelation')
    axes[0, 0].set_title('Autocorrelation Function')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log plot of autocorrelation function
    axes[0, 1].semilogy(lags, np.abs(autocorr_func), 'r-', linewidth=2, alpha=0.7)
    axes[0, 1].axhline(y=1/np.e, color='red', linestyle='--', alpha=0.5, label='1/e')
    axes[0, 1].axvline(x=autocorr_time, color='green', linestyle='--', alpha=0.5, 
                      label=f'τ = {autocorr_time:.2f}')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('|Autocorrelation|')
    axes[0, 1].set_title('Autocorrelation Function (Log Scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Windowed autocorrelation times
    valid_windows = [w for w, t in zip(window_sizes, windowed_autocorr) if not np.isnan(t)]
    valid_autocorr = [t for t in windowed_autocorr if not np.isnan(t)]
    
    axes[1, 0].semilogx(valid_windows, valid_autocorr, 'go-', alpha=0.7, linewidth=2)
    axes[1, 0].axhline(y=autocorr_time, color='red', linestyle='--', alpha=0.5, 
                      label=f'Full series: {autocorr_time:.2f}')
    axes[1, 0].set_xlabel('Window Size')
    axes[1, 0].set_ylabel('Autocorrelation Time')
    axes[1, 0].set_title('Windowed Autocorrelation Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample trace with autocorrelation time marked
    trace_samples = samples[:500]  # Show first 500 samples
    axes[1, 1].plot(range(len(trace_samples)), trace_samples, 'b-', alpha=0.7, linewidth=1)
    
    # Mark autocorrelation time intervals
    for i in range(0, len(trace_samples), int(autocorr_time)):
        axes[1, 1].axvline(x=i, color='red', linestyle='--', alpha=0.3)
    
    axes[1, 1].set_xlabel('Sample Number')
    axes[1, 1].set_ylabel('x')
    axes[1, 1].set_title('Sample Trace with Autocorrelation Intervals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Autocorrelation Analysis', fontsize=16)
    plt.tight_layout()
    
    autocorr_path = plot_dir / "autocorrelation_analysis.png"
    plt.savefig(autocorr_path, dpi=300, bbox_inches='tight')
    print(f"\\nAutocorrelation analysis plot saved to: {autocorr_path}")
    plt.close()
    
    # Print statistics
    print(f"\\nAutocorrelation Statistics:")
    print(f"  Integrated autocorrelation time: {autocorr_time:.2f}")
    print(f"  Effective sample size: {len(samples) / (2 * autocorr_time + 1):.1f}")
    print(f"  Acceptance rate: {results_dict['acceptance_rate']:.3f}")
    
    return samples, autocorr_time

def error_analysis_demo(optimal_step):
    """Detailed error analysis using jackknife and bootstrap."""
    print("\\n" + "=" * 60)
    print("ERROR ANALYSIS DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create harmonic oscillator
    ho = HarmonicOscillator(n_time_steps=50, omega=1.0, lattice_spacing=optimal_step)
    
    # Run simulation
    n_samples = 8000
    burn_in = 1000
    
    print(f"Running {n_samples} samples for error analysis...")
    results_dict = ho.run_simulation(n_samples, step_size=optimal_step, burn_in=burn_in)
    
    # Extract position samples
    samples = np.array(results_dict['observables']['position'])
    
    # Calculate observables
    x_squared_samples = samples**2
    x_fourth_samples = samples**4
    
    # Calculate means
    x_squared_mean = np.mean(x_squared_samples)
    x_fourth_mean = np.mean(x_fourth_samples)
    
    # Calculate errors using different methods
    _, x_squared_jackknife = jackknife_error(x_squared_samples)
    _, x_fourth_jackknife = jackknife_error(x_fourth_samples)
    
    _, x_squared_bootstrap = bootstrap_error(x_squared_samples)
    _, x_fourth_bootstrap = bootstrap_error(x_fourth_samples)
    
    # Binning analysis
    bin_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    x_squared_binned_errors = []
    x_fourth_binned_errors = []
    
    for bin_size in bin_sizes:
        bin_sizes_array, x_squared_errors_array = binning_analysis(x_squared_samples, bin_size)
        _, x_fourth_errors_array = binning_analysis(x_fourth_samples, bin_size)
        # Take the error at the maximum bin size available
        if len(x_squared_errors_array) > 0:
            x_squared_binned_errors.append(x_squared_errors_array[-1])
        else:
            x_squared_binned_errors.append(0)
        if len(x_fourth_errors_array) > 0:
            x_fourth_binned_errors.append(x_fourth_errors_array[-1])
        else:
            x_fourth_binned_errors.append(0)
    
    # Create error analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Comparison of error methods
    methods = ['Jackknife', 'Bootstrap', 'Binning (64)']
    x_squared_errors = [x_squared_jackknife, x_squared_bootstrap, x_squared_binned_errors[6]]
    x_fourth_errors = [x_fourth_jackknife, x_fourth_bootstrap, x_fourth_binned_errors[6]]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, x_squared_errors, width, label='⟨x²⟩', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, x_fourth_errors, width, label='⟨x⁴⟩', alpha=0.7)
    axes[0, 0].set_xlabel('Error Method')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].set_title('Error Method Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(methods)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binning analysis
    axes[0, 1].semilogx(bin_sizes, x_squared_binned_errors, 'bo-', 
                       alpha=0.7, linewidth=2, label='⟨x²⟩')
    axes[0, 1].semilogx(bin_sizes, x_fourth_binned_errors, 'ro-', 
                       alpha=0.7, linewidth=2, label='⟨x⁴⟩')
    axes[0, 1].set_xlabel('Bin Size')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].set_title('Binning Analysis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bootstrap distribution
    n_bootstrap = 1000
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        bootstrap_indices = np.random.choice(len(x_squared_samples), 
                                           size=len(x_squared_samples), replace=True)
        bootstrap_sample = x_squared_samples[bootstrap_indices]
        bootstrap_samples.append(np.mean(bootstrap_sample))
    
    axes[1, 0].hist(bootstrap_samples, bins=50, density=True, alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=x_squared_mean, color='red', linestyle='-', 
                      linewidth=2, label=f'Original: {x_squared_mean:.4f}')
    axes[1, 0].axvline(x=np.mean(bootstrap_samples), color='green', linestyle='--', 
                      linewidth=2, label=f'Bootstrap: {np.mean(bootstrap_samples):.4f}')
    axes[1, 0].set_xlabel('⟨x²⟩')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].set_title('Bootstrap Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Theoretical comparison
    theoretical_x_squared = 0.5
    theoretical_x_fourth = 0.75
    
    observables = ['⟨x²⟩', '⟨x⁴⟩']
    measured_values = [x_squared_mean, x_fourth_mean]
    measured_errors = [x_squared_jackknife, x_fourth_jackknife]
    theoretical_values = [theoretical_x_squared, theoretical_x_fourth]
    
    x_pos = np.arange(len(observables))
    
    axes[1, 1].errorbar(x_pos, measured_values, yerr=measured_errors, 
                       fmt='bo', capsize=5, linewidth=2, label='Measured')
    axes[1, 1].plot(x_pos, theoretical_values, 'rs', markersize=10, 
                   label='Theoretical')
    axes[1, 1].set_xlabel('Observable')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Theoretical Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(observables)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Error Analysis', fontsize=16)
    plt.tight_layout()
    
    error_path = plot_dir / "error_analysis.png"
    plt.savefig(error_path, dpi=300, bbox_inches='tight')
    print(f"\\nError analysis plot saved to: {error_path}")
    plt.close()
    
    # Print results
    print(f"\\nError Analysis Results:")
    print(f"⟨x²⟩:")
    print(f"  Value: {x_squared_mean:.4f}")
    print(f"  Jackknife error: {x_squared_jackknife:.4f}")
    print(f"  Bootstrap error: {x_squared_bootstrap:.4f}")
    print(f"  Theoretical: {theoretical_x_squared:.4f}")
    print(f"⟨x⁴⟩:")
    print(f"  Value: {x_fourth_mean:.4f}")
    print(f"  Jackknife error: {x_fourth_jackknife:.4f}")
    print(f"  Bootstrap error: {x_fourth_bootstrap:.4f}")
    print(f"  Theoretical: {theoretical_x_fourth:.4f}")
    
    return samples

def main():
    """Main function to run all demos."""
    print("METROPOLIS ALGORITHM - IN-DEPTH DEMO")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots will be saved to: plots/metropolis/")
    
    # Run optimization demo first
    step_results, optimal_step = step_size_optimization_demo()
    print(f"\\nOptimal step size determined: {optimal_step:.4f}")
    
    # Run other demos with optimal parameters
    conv_results = convergence_study_demo(optimal_step)
    samples, autocorr_time = autocorrelation_analysis_demo(optimal_step)
    error_samples = error_analysis_demo(optimal_step)
    
    print("\\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots saved to: plots/metropolis/")
    
    # Create summary report
    plot_dir = create_plot_dir()
    summary_path = plot_dir / "demo_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("METROPOLIS ALGORITHM - IN-DEPTH DEMO SUMMARY\\n")
        f.write("=" * 50 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("STEP SIZE OPTIMIZATION:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Optimal step size: {optimal_step:.4f}\\n")
        f.write(f"Tested {len(step_results)} different step sizes\\n\\n")
        
        f.write("CONVERGENCE STUDY:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Tested {len(conv_results)} different sample sizes\\n")
        f.write(f"Range: {min(r['n_samples'] for r in conv_results)} to {max(r['n_samples'] for r in conv_results)} samples\\n\\n")
        
        f.write("AUTOCORRELATION ANALYSIS:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Autocorrelation time: {autocorr_time:.2f}\\n")
        f.write(f"Effective sample size: {len(samples) / (2 * autocorr_time + 1):.1f}\\n\\n")
        
        f.write("ERROR ANALYSIS:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Comprehensive error analysis with jackknife, bootstrap, and binning\\n")
        f.write(f"Theoretical comparison with exact harmonic oscillator results\\n")
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
