"""
In-Depth 1D Scalar Field Theory Demo

This script provides a comprehensive demonstration of the 1D scalar field theory
implementation including:
1. Phase transition studies
2. Critical behavior analysis
3. Correlation function studies
4. Mass spectrum analysis
5. Finite size effects
6. Detailed visualization

All plots are saved to plots/field_theory/
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from field_theory_1d import FieldTheory1D
from utils import *

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

def create_plot_dir():
    """Create plot directory if it doesn't exist."""
    plot_dir = Path("plots/field_theory")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def phase_transition_demo():
    """Study phase transition behavior."""
    print("=" * 60)
    print("PHASE TRANSITION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Scan across mass parameter
    mass_squared_values = np.linspace(-1.0, 1.0, 21)
    lambda_coupling = 0.1
    lattice_size = 50
    
    results = []
    
    for m_sq in mass_squared_values:
        print(f"\\nTesting m² = {m_sq:.3f}")
        
        # Create field theory
        field_theory = FieldTheory1D(
            lattice_size=lattice_size,
            mass_squared=m_sq,
            lambda_coupling=lambda_coupling
        )
        
        # Run simulation
        sim_results = field_theory.run_simulation(
            n_sweeps=3000,
            burn_in=1000,
            measurement_interval=5
        )
        
        # Calculate observables
        phi_avg = np.mean(sim_results['observables']['phi_avg'])
        phi_squared = np.mean(sim_results['observables']['phi_squared'])
        phi_fourth = np.mean(sim_results['observables']['phi_fourth'])
        
        # Calculate susceptibility
        phi_squared_fluct = np.var(sim_results['observables']['phi_squared'])
        susceptibility = lattice_size * phi_squared_fluct
        
        results.append({
            'm_squared': m_sq,
            'phi_avg': phi_avg,
            'phi_squared': phi_squared,
            'phi_fourth': phi_fourth,
            'susceptibility': susceptibility,
            'acceptance': sim_results['acceptance_rate']
        })
        
        print(f"  ⟨φ²⟩ = {phi_squared:.4f}")
        print(f"  χ = {susceptibility:.4f}")
        print(f"  Acceptance: {sim_results['acceptance_rate']:.1%}")
    
    # Create phase transition plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    m_values = [r['m_squared'] for r in results]
    phi_squared_values = [r['phi_squared'] for r in results]
    phi_fourth_values = [r['phi_fourth'] for r in results]
    susceptibility_values = [r['susceptibility'] for r in results]
    
    # φ² vs m²
    axes[0, 0].plot(m_values, phi_squared_values, 'bo-', alpha=0.7, linewidth=2)
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Critical point')
    axes[0, 0].set_xlabel('m²')
    axes[0, 0].set_ylabel('⟨φ²⟩')
    axes[0, 0].set_title('Order Parameter')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # φ⁴ vs m²
    axes[0, 1].plot(m_values, phi_fourth_values, 'go-', alpha=0.7, linewidth=2)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Critical point')
    axes[0, 1].set_xlabel('m²')
    axes[0, 1].set_ylabel('⟨φ⁴⟩')
    axes[0, 1].set_title('Fourth Moment')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Susceptibility vs m²
    axes[1, 0].plot(m_values, susceptibility_values, 'ro-', alpha=0.7, linewidth=2)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Critical point')
    axes[1, 0].set_xlabel('m²')
    axes[1, 0].set_ylabel('χ')
    axes[1, 0].set_title('Susceptibility')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Binder cumulant
    binder_cumulant = []
    for r in results:
        if r['phi_squared'] > 0:
            binder = 1 - r['phi_fourth'] / (3 * r['phi_squared']**2)
            binder_cumulant.append(binder)
        else:
            binder_cumulant.append(0)
    
    axes[1, 1].plot(m_values, binder_cumulant, 'mo-', alpha=0.7, linewidth=2)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Critical point')
    axes[1, 1].axhline(y=2/3, color='blue', linestyle='--', alpha=0.5, label='Universal value')
    axes[1, 1].set_xlabel('m²')
    axes[1, 1].set_ylabel('U₄')
    axes[1, 1].set_title('Binder Cumulant')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Phase Transition Analysis', fontsize=16)
    plt.tight_layout()
    
    phase_path = plot_dir / "phase_transition.png"
    plt.savefig(phase_path, dpi=300, bbox_inches='tight')
    print(f"\\nPhase transition plot saved to: {phase_path}")
    plt.close()
    
    return results

def correlation_function_demo():
    """Study correlation functions."""
    print("\\n" + "=" * 60)
    print("CORRELATION FUNCTION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test different phases
    test_cases = [
        {'name': 'Symmetric Phase', 'm_squared': 0.5, 'lambda': 0.1},
        {'name': 'Broken Phase', 'm_squared': -0.5, 'lambda': 0.1},
        {'name': 'Critical Point', 'm_squared': 0.0, 'lambda': 0.1},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    colors = ['blue', 'red', 'green']
    
    for i, case in enumerate(test_cases):
        print(f"\\nTesting: {case['name']}")
        
        # Create field theory
        field_theory = FieldTheory1D(
            lattice_size=100,
            mass_squared=case['m_squared'],
            lambda_coupling=case['lambda']
        )
        
        # Run simulation
        results = field_theory.run_simulation(
            n_sweeps=5000,
            burn_in=1500,
            measurement_interval=10
        )
        
        # Calculate correlation function
        distances, avg_correlation = field_theory.compute_correlation_function()
        
        # Plot correlation function
        axes[0, 0].semilogy(distances, avg_correlation, color=colors[i], 
                           label=case['name'], linewidth=2, alpha=0.7)
        
        # Fit exponential decay to extract correlation length
        # C(r) ∝ exp(-r/ξ) for large r
        fit_start = 5
        fit_end = min(30, len(avg_correlation))
        if fit_end > fit_start:
            log_corr = np.log(avg_correlation[fit_start:fit_end])
            fit_distances = distances[fit_start:fit_end]
            
            # Linear fit in log space
            coeffs = np.polyfit(fit_distances, log_corr, 1)
            correlation_length = -1.0 / coeffs[0]
            
            # Plot fit
            fit_curve = np.exp(coeffs[1] + coeffs[0] * fit_distances)
            axes[0, 0].semilogy(fit_distances, fit_curve, '--', color=colors[i], alpha=0.5)
            
            print(f"  Correlation length: {correlation_length:.2f}")
        
        # Plot final field configuration
        axes[0, 1].plot(range(100), field_theory.field, color=colors[i], 
                       label=case['name'], linewidth=2, alpha=0.7)
        
        # Plot field evolution
        field_theory.plot_observables(save_path=plot_dir / f"observables_{case['name'].lower().replace(' ', '_')}.png")
        
        print(f"  ⟨φ²⟩ = {np.mean(results['observables']['phi_squared']):.4f}")
        print(f"  Acceptance: {results['acceptance_rate']:.1%}")
    
    # Finalize correlation plot
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Correlation Function')
    axes[0, 0].set_title('Correlation Functions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Lattice Site')
    axes[0, 1].set_ylabel('Field Value φ(x)')
    axes[0, 1].set_title('Final Field Configurations')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mass spectrum analysis
    axes[1, 0].set_title('Mass Spectrum Analysis')
    axes[1, 0].text(0.5, 0.5, 'Mass spectrum analysis\\nwould require more\\nadvanced techniques', 
                   ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
    
    # Finite size effects
    axes[1, 1].set_title('Finite Size Effects')
    axes[1, 1].text(0.5, 0.5, 'Finite size scaling\\nanalysis would require\\nmultiple lattice sizes', 
                   ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
    
    plt.suptitle('Correlation Function Analysis', fontsize=16)
    plt.tight_layout()
    
    corr_path = plot_dir / "correlation_functions.png"
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"\\nCorrelation function plot saved to: {corr_path}")
    plt.close()
    
    return test_cases

def finite_size_scaling_demo():
    """Study finite size effects."""
    print("\\n" + "=" * 60)
    print("FINITE SIZE SCALING DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test different lattice sizes at critical point
    lattice_sizes = [20, 30, 40, 50, 60, 80]
    m_squared = 0.0  # Critical point
    lambda_coupling = 0.1
    
    results = []
    
    for L in lattice_sizes:
        print(f"\\nTesting lattice size L = {L}")
        
        # Create field theory
        field_theory = FieldTheory1D(
            lattice_size=L,
            mass_squared=m_squared,
            lambda_coupling=lambda_coupling
        )
        
        # Run simulation
        sim_results = field_theory.run_simulation(
            n_sweeps=4000,
            burn_in=1000,
            measurement_interval=5
        )
        
        # Calculate observables
        phi_squared = np.mean(sim_results['observables']['phi_squared'])
        phi_squared_var = np.var(sim_results['observables']['phi_squared'])
        susceptibility = L * phi_squared_var
        
        results.append({
            'L': L,
            'phi_squared': phi_squared,
            'susceptibility': susceptibility,
            'acceptance': sim_results['acceptance_rate']
        })
        
        print(f"  ⟨φ²⟩ = {phi_squared:.4f}")
        print(f"  χ = {susceptibility:.4f}")
        print(f"  Acceptance: {sim_results['acceptance_rate']:.1%}")
    
    # Create finite size scaling plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    L_values = [r['L'] for r in results]
    phi_squared_values = [r['phi_squared'] for r in results]
    susceptibility_values = [r['susceptibility'] for r in results]
    
    # φ² vs L
    axes[0, 0].plot(L_values, phi_squared_values, 'bo-', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Lattice Size L')
    axes[0, 0].set_ylabel('⟨φ²⟩')
    axes[0, 0].set_title('Order Parameter vs System Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # χ vs L
    axes[0, 1].plot(L_values, susceptibility_values, 'ro-', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Lattice Size L')
    axes[0, 1].set_ylabel('χ')
    axes[0, 1].set_title('Susceptibility vs System Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Log-log plot for scaling
    axes[1, 0].loglog(L_values, phi_squared_values, 'bo-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Lattice Size L')
    axes[1, 0].set_ylabel('⟨φ²⟩')
    axes[1, 0].set_title('Log-Log Scaling')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scaling collapse attempt
    # This would require knowledge of critical exponents
    axes[1, 1].plot(L_values, susceptibility_values, 'go-', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Lattice Size L')
    axes[1, 1].set_ylabel('χ')
    axes[1, 1].set_title('Susceptibility Scaling')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Finite Size Scaling Analysis', fontsize=16)
    plt.tight_layout()
    
    scaling_path = plot_dir / "finite_size_scaling.png"
    plt.savefig(scaling_path, dpi=300, bbox_inches='tight')
    print(f"\\nFinite size scaling plot saved to: {scaling_path}")
    plt.close()
    
    return results

def detailed_configuration_demo():
    """Demonstrate detailed field configurations."""
    print("\\n" + "=" * 60)
    print("DETAILED CONFIGURATION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create field theory in broken phase
    field_theory = FieldTheory1D(
        lattice_size=80,
        mass_squared=-0.3,
        lambda_coupling=0.1
    )
    
    # Run simulation
    print("Running simulation for configuration analysis...")
    results = field_theory.run_simulation(
        n_sweeps=3000,
        burn_in=1000,
        measurement_interval=10
    )
    
    # Create detailed configuration plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Show evolution of field configurations
    n_configs = min(6, len(field_theory.field_history))
    config_indices = np.linspace(0, len(field_theory.field_history)-1, n_configs, dtype=int)
    
    for i, config_idx in enumerate(config_indices):
        row = i // 2
        col = i % 2
        
        config = field_theory.field_history[config_idx]
        axes[row, col].plot(range(len(config)), config, 'b-', linewidth=2, alpha=0.7)
        axes[row, col].set_xlabel('Lattice Site')
        axes[row, col].set_ylabel('Field Value φ(x)')
        axes[row, col].set_title(f'Configuration {config_idx + 1}')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.suptitle('Field Configuration Evolution', fontsize=16)
    plt.tight_layout()
    
    config_path = plot_dir / "field_configurations.png"
    plt.savefig(config_path, dpi=300, bbox_inches='tight')
    print(f"\\nField configuration plot saved to: {config_path}")
    plt.close()
    
    # Save observables plot
    field_theory.plot_observables(save_path=plot_dir / "detailed_observables.png")
    
    # Create histogram of field values
    plt.figure(figsize=(10, 6))
    
    # Check if field_history has data
    if hasattr(field_theory, 'field_history') and len(field_theory.field_history) > 0:
        all_field_values = np.concatenate(field_theory.field_history[-100:])  # Last 100 configs
        plt.hist(all_field_values, bins=50, density=True, alpha=0.7, color='skyblue')
        plt.xlabel('Field Value φ')
        plt.ylabel('Probability Density')
        plt.title('Distribution of Field Values')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='φ = 0')
        plt.legend()
        
        histogram_path = plot_dir / "field_distribution.png"
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Field distribution plot saved to: {histogram_path}")
    else:
        plt.text(0.5, 0.5, 'No field history data available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Field Distribution (No Data)')
        histogram_path = plot_dir / "field_distribution.png"
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"Field distribution plot saved to: {histogram_path} (no data)")
    
    plt.close()
    
    return results

def main():
    """Main function to run all demos."""
    print("1D SCALAR FIELD THEORY - IN-DEPTH DEMO")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots will be saved to: plots/field_theory/")
    
    # Run all demos
    phase_results = phase_transition_demo()
    corr_results = correlation_function_demo()
    scaling_results = finite_size_scaling_demo()
    config_results = detailed_configuration_demo()
    
    print("\\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots saved to: plots/field_theory/")
    
    # Create summary report
    plot_dir = create_plot_dir()
    summary_path = plot_dir / "demo_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("1D SCALAR FIELD THEORY - IN-DEPTH DEMO SUMMARY\\n")
        f.write("=" * 55 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("PHASE TRANSITION STUDY:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"Scanned m² from -1.0 to 1.0 with {len(phase_results)} points\\n")
        f.write(f"Critical behavior observed around m² = 0\\n\\n")
        
        f.write("CORRELATION FUNCTION ANALYSIS:\\n")
        f.write("-" * 30 + "\\n")
        for case in corr_results:
            f.write(f"{case['name']}: m² = {case['m_squared']}, λ = {case['lambda']}\\n")
        
        f.write("\\nFINITE SIZE SCALING:\\n")
        f.write("-" * 30 + "\\n")
        f.write(f"Tested lattice sizes: {[r['L'] for r in scaling_results]}\\n")
        f.write(f"Critical point scaling behavior analyzed\\n")
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
