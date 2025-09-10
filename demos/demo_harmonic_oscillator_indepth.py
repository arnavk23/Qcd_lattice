"""
In-Depth Harmonic Osys.path.append(str(Path(__file__).parent / "src"))

from harmonic_oscillator import HarmonicOscillatorMC as HarmonicOscillator
from utils import *lator Demo

This script provides a comprehensive demonstration of the harmonic oscillator
implementation including:
1. Parameter studies
2. Convergence analysis
3. Autocorrelation analysis
4. Comparison with theoretical predictions
5. Statistical validation
6. Visualization of all results

All plots are saved to plots/harmonic_oscillator/
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from harmonic_oscillator import HarmonicOscillatorMC
from utils import *

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

def create_plot_dir():
    """Create plot directory if it doesn't exist."""
    plot_dir = Path("plots/harmonic_oscillator")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def parameter_study_demo():
    """Demonstrate parameter dependence."""
    print("=" * 60)
    print("PARAMETER STUDY DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Different parameter sets
    parameter_sets = [
        {'name': 'Standard', 'mass': 1.0, 'omega': 1.0, 'dt': 0.5, 'N': 50},
        {'name': 'High Frequency', 'mass': 1.0, 'omega': 2.0, 'dt': 0.25, 'N': 50},
        {'name': 'Heavy Mass', 'mass': 2.0, 'omega': 1.0, 'dt': 0.5, 'N': 50},
        {'name': 'Fine Lattice', 'mass': 1.0, 'omega': 1.0, 'dt': 0.2, 'N': 100},
    ]
    
    results_summary = []
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, params in enumerate(parameter_sets):
        print(f"\nTesting: {params['name']}")
        print(f"  Parameters: m={params['mass']}, ω={params['omega']}, Δt={params['dt']}, N={params['N']}")
        
        # Create oscillator
        oscillator = HarmonicOscillatorMC(
            n_time_steps=params['N'],
            mass=params['mass'],
            omega=params['omega'],
            lattice_spacing=params['dt']
        )
        
        # Run simulation
        results = oscillator.run_simulation(
            n_sweeps=3000,
            step_size=0.3,
            burn_in=1000,
            measurement_interval=5
        )
        
        # Analyze results
        pos_data = np.array(results['observables']['position'])
        pos_mean = np.mean(pos_data)
        pos_err = np.std(pos_data) / np.sqrt(len(pos_data))
        
        pos_sq_data = np.array(results['observables']['position_squared'])
        pos_sq_mean = np.mean(pos_sq_data)
        pos_sq_theory = 0.5 / params['omega']
        
        results_summary.append({
            'name': params['name'],
            'pos_mean': pos_mean,
            'pos_err': pos_err,
            'pos_sq_mean': pos_sq_mean,
            'pos_sq_theory': pos_sq_theory,
            'acceptance': results['acceptance_rate']
        })
        
        # Plot final path
        axes[i].plot(range(params['N']), results['final_path'], 'b-', alpha=0.7, linewidth=2)
        axes[i].set_title(f"{params['name']}\\n⟨q⟩ = {pos_mean:.4f} ± {pos_err:.4f}")
        axes[i].set_xlabel('Time Step')
        axes[i].set_ylabel('Position q(t)')
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        print(f"  ⟨q⟩ = {pos_mean:.5f} ± {pos_err:.5f}")
        print(f"  ⟨q²⟩ = {pos_sq_mean:.4f} (theory: {pos_sq_theory:.4f})")
        print(f"  Acceptance rate: {results['acceptance_rate']:.1%}")
        
        # Save individual results plot
        individual_save_path = plot_dir / f"results_{params['name'].lower().replace(' ', '_')}.png"
        oscillator.plot_results(results, save_path=str(individual_save_path))
    
    plt.suptitle('Parameter Study: Final Path Configurations', fontsize=16)
    plt.tight_layout()
    
    param_study_path = plot_dir / "parameter_study.png"
    plt.savefig(param_study_path, dpi=300, bbox_inches='tight')
    print(f"\\nParameter study plot saved to: {param_study_path}")
    plt.close()
    
    # Summary table
    print(f"\\nPARAMETER STUDY SUMMARY")
    print("-" * 70)
    print(f"{'Parameter Set':<15} {'⟨q⟩':<12} {'⟨q²⟩':<10} {'Accept':<8} {'Status':<8}")
    print("-" * 70)
    
    for result in results_summary:
        pos_check = abs(result['pos_mean']) < 3 * result['pos_err']
        status = "PASS" if pos_check else "FAIL"
        print(f"{result['name']:<15} {result['pos_mean']:>6.4f}±{result['pos_err']:.4f} "
              f"{result['pos_sq_mean']:>6.3f} {result['acceptance']:>6.1%} {status:<8}")
    
    return results_summary

def convergence_analysis_demo():
    """Demonstrate convergence with simulation length."""
    print("\\n" + "=" * 60)
    print("CONVERGENCE ANALYSIS DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create oscillator
    oscillator = HarmonicOscillatorMC(
        n_time_steps=50,
        mass=1.0,
        omega=1.0,
        lattice_spacing=0.5
    )
    
    # Run long simulation
    print("Running long simulation for convergence analysis...")
    results = oscillator.run_simulation(
        n_sweeps=10000,
        step_size=0.3,
        burn_in=2000,
        measurement_interval=5
    )
    
    pos_data = np.array(results['observables']['position'])
    
    # Analyze convergence
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Running average
    running_avg = np.cumsum(pos_data) / np.arange(1, len(pos_data) + 1)
    axes[0, 0].plot(running_avg, alpha=0.8, linewidth=2)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('Running Average Convergence')
    axes[0, 0].set_xlabel('Measurement')
    axes[0, 0].set_ylabel('Cumulative ⟨q⟩')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binned error analysis
    bin_sizes = np.logspace(1, 3, 20).astype(int)
    bin_sizes = bin_sizes[bin_sizes < len(pos_data)//4]
    
    binned_errors = []
    for bin_size in bin_sizes:
        n_bins = len(pos_data) // bin_size
        binned_data = pos_data[:n_bins * bin_size].reshape(n_bins, bin_size)
        bin_means = np.mean(binned_data, axis=1)
        binned_error = np.std(bin_means) / np.sqrt(n_bins)
        binned_errors.append(binned_error)
    
    axes[0, 1].loglog(bin_sizes, binned_errors, 'bo-', alpha=0.7)
    axes[0, 1].loglog(bin_sizes, binned_errors[0] * np.sqrt(bin_sizes[0] / bin_sizes), 'r--', alpha=0.7, label='1/√N')
    axes[0, 1].set_title('Binned Error Analysis')
    axes[0, 1].set_xlabel('Bin Size')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrelation function
    autocorr = autocorrelation_function(pos_data, max_lag=200)
    lags = np.arange(len(autocorr))
    axes[1, 0].plot(lags, autocorr, 'b-', alpha=0.7, linewidth=2)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Autocorrelation Function')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Integrated autocorrelation time
    tau_int = integrated_autocorrelation_time(pos_data)
    axes[1, 0].axvline(x=tau_int, color='green', linestyle='--', alpha=0.7, label=f'τ_int = {tau_int:.1f}')
    axes[1, 0].legend()
    
    # Distribution of measurements
    axes[1, 1].hist(pos_data, bins=50, density=True, alpha=0.7, color='skyblue')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    axes[1, 1].set_title('Distribution of ⟨q⟩ Measurements')
    axes[1, 1].set_xlabel('⟨q⟩')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Analysis', fontsize=16)
    plt.tight_layout()
    
    convergence_path = plot_dir / "convergence_analysis.png"
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis plot saved to: {convergence_path}")
    plt.close()
    
    # Print analysis
    final_mean = np.mean(pos_data)
    final_error = np.std(pos_data) / np.sqrt(len(pos_data))
    effective_samples = len(pos_data) / (2 * tau_int + 1)
    
    print(f"\\nCONVERGENCE ANALYSIS RESULTS")
    print("-" * 40)
    print(f"Total measurements: {len(pos_data)}")
    print(f"Final ⟨q⟩: {final_mean:.6f} ± {final_error:.6f}")
    print(f"Integrated autocorrelation time: {tau_int:.2f}")
    print(f"Effective samples: {effective_samples:.1f}")
    
    return results

def theoretical_comparison_demo():
    """Compare with theoretical predictions."""
    print("\\n" + "=" * 60)
    print("THEORETICAL COMPARISON DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test different frequencies
    omega_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    simulation_results = []
    theoretical_results = []
    
    for omega in omega_values:
        print(f"\\nTesting ω = {omega}")
        
        # Create oscillator
        oscillator = HarmonicOscillatorMC(
            n_time_steps=50,
            mass=1.0,
            omega=omega,
            lattice_spacing=0.5/omega  # Adjust dt for stability
        )
        
        # Run simulation
        results = oscillator.run_simulation(
            n_sweeps=4000,
            step_size=0.3,
            burn_in=1000,
            measurement_interval=5
        )
        
        # Analyze results
        pos_data = np.array(results['observables']['position'])
        pos_sq_data = np.array(results['observables']['position_squared'])
        
        pos_mean = np.mean(pos_data)
        pos_err = np.std(pos_data) / np.sqrt(len(pos_data))
        pos_sq_mean = np.mean(pos_sq_data)
        pos_sq_err = np.std(pos_sq_data) / np.sqrt(len(pos_sq_data))
        
        simulation_results.append({
            'omega': omega,
            'pos_mean': pos_mean,
            'pos_err': pos_err,
            'pos_sq_mean': pos_sq_mean,
            'pos_sq_err': pos_sq_err,
            'acceptance': results['acceptance_rate']
        })
        
        # Theoretical predictions
        theoretical_pos_sq = 0.5 / omega
        theoretical_results.append({
            'omega': omega,
            'pos_sq_theory': theoretical_pos_sq
        })
        
        print(f"  ⟨q⟩ = {pos_mean:.5f} ± {pos_err:.5f}")
        print(f"  ⟨q²⟩ = {pos_sq_mean:.4f} ± {pos_sq_err:.4f} (theory: {theoretical_pos_sq:.4f})")
        print(f"  Acceptance: {results['acceptance_rate']:.1%}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Position average comparison
    omegas = [r['omega'] for r in simulation_results]
    pos_means = [r['pos_mean'] for r in simulation_results]
    pos_errs = [r['pos_err'] for r in simulation_results]
    
    axes[0].errorbar(omegas, pos_means, yerr=pos_errs, fmt='bo-', capsize=5, label='Simulation')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Theory: ⟨q⟩ = 0')
    axes[0].set_xlabel('Frequency ω')
    axes[0].set_ylabel('⟨q⟩')
    axes[0].set_title('Position Average vs Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Position squared comparison
    pos_sq_means = [r['pos_sq_mean'] for r in simulation_results]
    pos_sq_errs = [r['pos_sq_err'] for r in simulation_results]
    pos_sq_theory = [r['pos_sq_theory'] for r in theoretical_results]
    
    axes[1].errorbar(omegas, pos_sq_means, yerr=pos_sq_errs, fmt='bo-', capsize=5, label='Simulation')
    axes[1].plot(omegas, pos_sq_theory, 'r--', linewidth=2, label='Theory: ⟨q²⟩ = 1/(2ω)')
    axes[1].set_xlabel('Frequency ω')
    axes[1].set_ylabel('⟨q²⟩')
    axes[1].set_title('Position Squared vs Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Theoretical Comparison', fontsize=16)
    plt.tight_layout()
    
    theory_path = plot_dir / "theoretical_comparison.png"
    plt.savefig(theory_path, dpi=300, bbox_inches='tight')
    print(f"\\nTheoretical comparison plot saved to: {theory_path}")
    plt.close()
    
    return simulation_results, theoretical_results

def statistical_validation_demo():
    """Perform statistical validation tests."""
    print("\\n" + "=" * 60)
    print("STATISTICAL VALIDATION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create oscillator
    oscillator = HarmonicOscillatorMC(
        n_time_steps=50,
        mass=1.0,
        omega=1.0,
        lattice_spacing=0.5
    )
    
    # Run simulation
    print("Running simulation for statistical validation...")
    results = oscillator.run_simulation(
        n_sweeps=8000,
        step_size=0.3,
        burn_in=2000,
        measurement_interval=5
    )
    
    pos_data = np.array(results['observables']['position'])
    
    # Jackknife analysis
    print("\\nPerforming jackknife analysis...")
    jack_mean, jack_err = jackknife_error(pos_data)
    
    # Bootstrap analysis
    print("Performing bootstrap analysis...")
    boot_mean, boot_err = bootstrap_error(pos_data, n_bootstrap=1000)
    
    # Binning analysis
    print("Performing binning analysis...")
    bin_sizes, bin_errors = binning_analysis(pos_data)
    
    # Create validation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error comparison
    methods = ['Naive', 'Jackknife', 'Bootstrap']
    naive_err = np.std(pos_data) / np.sqrt(len(pos_data))
    errors = [naive_err, jack_err, boot_err]
    
    axes[0, 0].bar(methods, errors, alpha=0.7, color=['blue', 'red', 'green'])
    axes[0, 0].set_title('Error Estimation Methods')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binning analysis
    axes[0, 1].semilogx(bin_sizes, bin_errors, 'bo-', alpha=0.7)
    axes[0, 1].axhline(y=jack_err, color='red', linestyle='--', alpha=0.7, label='Jackknife')
    axes[0, 1].set_title('Binning Analysis')
    axes[0, 1].set_xlabel('Bin Size')
    axes[0, 1].set_ylabel('Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bootstrap distribution
    bootstrap_samples = [bootstrap_error(pos_data, n_bootstrap=1)[0] for _ in range(1000)]
    axes[1, 0].hist(bootstrap_samples, bins=50, density=True, alpha=0.7, color='green')
    axes[1, 0].axvline(x=np.mean(bootstrap_samples), color='red', linestyle='--', alpha=0.8)
    axes[1, 0].set_title('Bootstrap Distribution')
    axes[1, 0].set_xlabel('⟨q⟩')
    axes[1, 0].set_ylabel('Probability Density')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistical tests
    # Normality test (visual)
    from scipy import stats
    sorted_data = np.sort(pos_data)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
    axes[1, 1].plot(theoretical_quantiles, sorted_data, 'bo', alpha=0.5, markersize=3)
    axes[1, 1].plot(theoretical_quantiles, theoretical_quantiles * np.std(pos_data) + np.mean(pos_data), 'r-', alpha=0.8)
    axes[1, 1].set_title('Q-Q Plot (Normality Test)')
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Validation', fontsize=16)
    plt.tight_layout()
    
    validation_path = plot_dir / "statistical_validation.png"
    plt.savefig(validation_path, dpi=300, bbox_inches='tight')
    print(f"\\nStatistical validation plot saved to: {validation_path}")
    plt.close()
    
    # Print results
    print(f"\\nSTATISTICAL VALIDATION RESULTS")
    print("-" * 40)
    print(f"Naive error: {naive_err:.6f}")
    print(f"Jackknife: {jack_mean:.6f} ± {jack_err:.6f}")
    print(f"Bootstrap: {boot_mean:.6f} ± {boot_err:.6f}")
    print(f"Autocorrelation time: {integrated_autocorrelation_time(pos_data):.2f}")
    
    return {
        'naive_err': naive_err,
        'jack_mean': jack_mean,
        'jack_err': jack_err,
        'boot_mean': boot_mean,
        'boot_err': boot_err
    }

def main():
    """Main function to run all demos."""
    print("HARMONIC OSCILLATOR - IN-DEPTH DEMO")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots will be saved to: plots/harmonic_oscillator/")
    
    # Run all demos
    param_results = parameter_study_demo()
    conv_results = convergence_analysis_demo()
    theory_results = theoretical_comparison_demo()
    stats_results = statistical_validation_demo()
    
    print("\\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots saved to: plots/harmonic_oscillator/")
    
    # Create summary report
    plot_dir = create_plot_dir()
    summary_path = plot_dir / "demo_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("HARMONIC OSCILLATOR - IN-DEPTH DEMO SUMMARY\\n")
        f.write("=" * 50 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("PARAMETER STUDY RESULTS:\\n")
        f.write("-" * 30 + "\\n")
        for result in param_results:
            f.write(f"{result['name']}: ⟨q⟩ = {result['pos_mean']:.5f} ± {result['pos_err']:.5f}\\n")
        
        f.write("\\nSTATISTICAL VALIDATION:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"Jackknife: {stats_results['jack_mean']:.6f} ± {stats_results['jack_err']:.6f}\\n")
        f.write(f"Bootstrap: {stats_results['boot_mean']:.6f} ± {stats_results['boot_err']:.6f}\\n")
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
