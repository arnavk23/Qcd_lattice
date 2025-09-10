"""
In-Depth Hybrid Monte Carlo Demo

This script provides a comprehensive demonstration of the HMC implementation
including:
1. Step size optimization
2. Trajectory length studies
3. Acceptance rate analysis
4. Energy conservation studies
5. Comparison with Metropolis algorithm
6. Performance benchmarking

All plots are saved to plots/hmc/
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

from hmc import HMCFieldTheory1D
from harmonic_oscillator import HarmonicOscillatorMC
from utils import *

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

class HMC:
    """Simple HMC implementation for harmonic oscillator."""
    
    def __init__(self, n_dim, step_size, n_steps, potential, gradient):
        self.n_dim = n_dim
        self.step_size = step_size
        self.n_steps = n_steps
        self.potential = potential
        self.gradient = gradient
        
    def leapfrog_trajectory(self, x, p):
        """Leapfrog integration for trajectory."""
        x_traj = [x.copy()]
        p_traj = [p.copy()]
        
        x_current = x.copy()
        p_current = p.copy()
        
        for i in range(self.n_steps):
            # Half step for momentum
            p_current = p_current - 0.5 * self.step_size * self.gradient(x_current)
            
            # Full step for position
            x_current = x_current + self.step_size * p_current
            
            # Half step for momentum
            p_current = p_current - 0.5 * self.step_size * self.gradient(x_current)
            
            x_traj.append(x_current.copy())
            p_traj.append(p_current.copy())
        
        return np.array(x_traj), np.array(p_traj)
    
    def step(self, x):
        """Single HMC step."""
        # Generate random momentum
        p = np.random.randn(self.n_dim)
        
        # Calculate initial energy
        kinetic_initial = 0.5 * np.sum(p**2)
        potential_initial = self.potential(x)
        energy_initial = kinetic_initial + potential_initial
        
        # Run trajectory
        x_traj, p_traj = self.leapfrog_trajectory(x, p)
        
        # Calculate final energy
        x_final = x_traj[-1]
        p_final = p_traj[-1]
        kinetic_final = 0.5 * np.sum(p_final**2)
        potential_final = self.potential(x_final)
        energy_final = kinetic_final + potential_final
        
        # Accept/reject
        delta_E = energy_final - energy_initial
        accept_prob = min(1, np.exp(-delta_E))
        
        if np.random.rand() < accept_prob:
            return x_final, True, delta_E
        else:
            return x, False, delta_E

# Set matplotlib backend to avoid display issues
plt.switch_backend('Agg')

def create_plot_dir():
    """Create plot directory if it doesn't exist."""
    plot_dir = Path("plots/hmc")
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir

def step_size_optimization_demo():
    """Study optimal step size for HMC."""
    print("=" * 60)
    print("STEP SIZE OPTIMIZATION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test range of step sizes
    step_sizes = np.logspace(-2, 0, 20)  # From 0.01 to 1.0
    n_steps = 50
    
    results = []
    
    for step_size in step_sizes:
        print(f"\\nTesting step size: {step_size:.4f}")
        
        # Create HMC sampler
        hmc = HMC(
            n_dim=1,
            step_size=step_size,
            n_steps=n_steps,
            potential=lambda x: 0.5 * x[0]**2,  # Harmonic potential
            gradient=lambda x: np.array([x[0]])
        )
        
        # Run short simulation
        n_samples = 1000
        samples = []
        energies = []
        acceptances = []
        
        x = np.array([0.0])
        
        for i in range(n_samples):
            x_new, accepted, energy_diff = hmc.step(x)
            samples.append(x_new[0])
            energies.append(energy_diff)
            acceptances.append(accepted)
            x = x_new
        
        acceptance_rate = np.mean(acceptances)
        avg_energy_diff = np.mean(np.abs(energies))
        
        results.append({
            'step_size': step_size,
            'acceptance_rate': acceptance_rate,
            'avg_energy_diff': avg_energy_diff,
            'samples': samples[-500:]  # Keep last 500 samples
        })
        
        print(f"  Acceptance rate: {acceptance_rate:.3f}")
        print(f"  Avg |ΔE|: {avg_energy_diff:.6f}")
    
    # Create optimization plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    step_values = [r['step_size'] for r in results]
    acceptance_values = [r['acceptance_rate'] for r in results]
    energy_values = [r['avg_energy_diff'] for r in results]
    
    # Acceptance rate vs step size
    axes[0, 0].semilogx(step_values, acceptance_values, 'bo-', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=0.65, color='red', linestyle='--', alpha=0.5, label='Optimal range')
    axes[0, 0].axhline(y=0.85, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Step Size')
    axes[0, 0].set_ylabel('Acceptance Rate')
    axes[0, 0].set_title('Acceptance Rate vs Step Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy conservation
    axes[0, 1].loglog(step_values, energy_values, 'ro-', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Step Size')
    axes[0, 1].set_ylabel('Average |ΔE|')
    axes[0, 1].set_title('Energy Conservation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Efficiency metric (acceptance rate / energy error)
    efficiency = np.array(acceptance_values) / (np.array(energy_values) + 1e-10)
    axes[1, 0].semilogx(step_values, efficiency, 'go-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Step Size')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_title('Sampling Efficiency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample distribution for optimal step size
    optimal_idx = np.argmax(efficiency)
    optimal_step = step_values[optimal_idx]
    optimal_samples = results[optimal_idx]['samples']
    
    axes[1, 1].hist(optimal_samples, bins=30, density=True, alpha=0.7, color='skyblue')
    
    # Overlay theoretical distribution
    x_theory = np.linspace(-3, 3, 100)
    p_theory = np.exp(-0.5 * x_theory**2) / np.sqrt(2 * np.pi)
    axes[1, 1].plot(x_theory, p_theory, 'r-', linewidth=2, label='Theoretical')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Probability Density')
    axes[1, 1].set_title(f'Optimal Step Size: {optimal_step:.4f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('HMC Step Size Optimization', fontsize=16)
    plt.tight_layout()
    
    step_path = plot_dir / "step_size_optimization.png"
    plt.savefig(step_path, dpi=300, bbox_inches='tight')
    print(f"\\nStep size optimization plot saved to: {step_path}")
    plt.close()
    
    return results, optimal_step

def trajectory_length_demo(optimal_step):
    """Study effect of trajectory length."""
    print("\\n" + "=" * 60)
    print("TRAJECTORY LENGTH DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Test different trajectory lengths
    n_steps_range = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200])
    
    results = []
    
    for n_steps in n_steps_range:
        print(f"\\nTesting trajectory length: {n_steps} steps")
        
        # Create HMC sampler
        hmc = HMC(
            n_dim=1,
            step_size=optimal_step,
            n_steps=n_steps,
            potential=lambda x: 0.5 * x[0]**2,
            gradient=lambda x: np.array([x[0]])
        )
        
        # Run simulation
        n_samples = 800
        samples = []
        energies = []
        acceptances = []
        
        x = np.array([0.0])
        
        start_time = time.time()
        for i in range(n_samples):
            x_new, accepted, energy_diff = hmc.step(x)
            samples.append(x_new[0])
            energies.append(energy_diff)
            acceptances.append(accepted)
            x = x_new
        
        computation_time = time.time() - start_time
        
        acceptance_rate = np.mean(acceptances)
        avg_energy_diff = np.mean(np.abs(energies))
        
        # Calculate autocorrelation time
        autocorr_time = integrated_autocorrelation_time(samples[100:])  # Skip burn-in
        
        results.append({
            'n_steps': n_steps,
            'acceptance_rate': acceptance_rate,
            'avg_energy_diff': avg_energy_diff,
            'autocorr_time': autocorr_time,
            'computation_time': computation_time,
            'samples': samples[-400:]  # Keep last 400 samples
        })
        
        print(f"  Acceptance rate: {acceptance_rate:.3f}")
        print(f"  Autocorrelation time: {autocorr_time:.2f}")
        print(f"  Computation time: {computation_time:.2f}s")
    
    # Create trajectory length plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    n_steps_values = [r['n_steps'] for r in results]
    acceptance_values = [r['acceptance_rate'] for r in results]
    autocorr_values = [r['autocorr_time'] for r in results]
    time_values = [r['computation_time'] for r in results]
    
    # Acceptance rate vs trajectory length
    axes[0, 0].plot(n_steps_values, acceptance_values, 'bo-', alpha=0.7, linewidth=2)
    axes[0, 0].set_xlabel('Trajectory Length (steps)')
    axes[0, 0].set_ylabel('Acceptance Rate')
    axes[0, 0].set_title('Acceptance Rate vs Trajectory Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Autocorrelation time
    axes[0, 1].plot(n_steps_values, autocorr_values, 'ro-', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Trajectory Length (steps)')
    axes[0, 1].set_ylabel('Autocorrelation Time')
    axes[0, 1].set_title('Autocorrelation Time vs Trajectory Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Computational efficiency
    efficiency = np.array(acceptance_values) / (np.array(autocorr_values) * np.array(time_values))
    axes[1, 0].plot(n_steps_values, efficiency, 'go-', alpha=0.7, linewidth=2)
    axes[1, 0].set_xlabel('Trajectory Length (steps)')
    axes[1, 0].set_ylabel('Efficiency')
    axes[1, 0].set_title('Computational Efficiency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Computation time
    axes[1, 1].plot(n_steps_values, time_values, 'mo-', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Trajectory Length (steps)')
    axes[1, 1].set_ylabel('Computation Time (s)')
    axes[1, 1].set_title('Computation Time vs Trajectory Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('HMC Trajectory Length Analysis', fontsize=16)
    plt.tight_layout()
    
    traj_path = plot_dir / "trajectory_length.png"
    plt.savefig(traj_path, dpi=300, bbox_inches='tight')
    print(f"\\nTrajectory length plot saved to: {traj_path}")
    plt.close()
    
    return results

def energy_conservation_demo(optimal_step):
    """Detailed study of energy conservation."""
    print("\\n" + "=" * 60)
    print("ENERGY CONSERVATION DEMO")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create HMC sampler
    hmc = HMC(
        n_dim=1,
        step_size=optimal_step,
        n_steps=50,
        potential=lambda x: 0.5 * x[0]**2,
        gradient=lambda x: np.array([x[0]])
    )
    
    # Run simulation and collect detailed trajectory information
    n_samples = 500
    trajectory_data = []
    
    x = np.array([0.0])
    
    for i in range(n_samples):
        # Store initial state
        x_initial = x.copy()
        p_initial = np.random.randn(1)
        
        # Calculate initial energy
        kinetic_initial = 0.5 * np.sum(p_initial**2)
        potential_initial = hmc.potential(x_initial)
        energy_initial = kinetic_initial + potential_initial
        
        # Run HMC trajectory
        x_traj, p_traj = hmc.leapfrog_trajectory(x_initial, p_initial)
        
        # Calculate final energy
        kinetic_final = 0.5 * np.sum(p_traj[-1]**2)
        potential_final = hmc.potential(x_traj[-1])
        energy_final = kinetic_final + potential_final
        
        # Store trajectory data
        trajectory_data.append({
            'x_traj': x_traj,
            'p_traj': p_traj,
            'energy_initial': energy_initial,
            'energy_final': energy_final,
            'energy_diff': energy_final - energy_initial
        })
        
        x = x_traj[-1]
    
    # Analyze energy conservation
    energy_diffs = [t['energy_diff'] for t in trajectory_data]
    
    # Create energy conservation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Energy differences histogram
    axes[0, 0].hist(energy_diffs, bins=50, density=True, alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('Energy Difference ΔE')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Distribution of Energy Differences')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Perfect conservation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy differences vs trajectory
    axes[0, 1].plot(range(len(energy_diffs)), energy_diffs, 'b-', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Trajectory Number')
    axes[0, 1].set_ylabel('Energy Difference ΔE')
    axes[0, 1].set_title('Energy Conservation Over Time')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase space trajectory (first few trajectories)
    for i in range(min(5, len(trajectory_data))):
        traj = trajectory_data[i]
        axes[1, 0].plot(traj['x_traj'], traj['p_traj'], alpha=0.7, linewidth=2)
        axes[1, 0].scatter(traj['x_traj'][0], traj['p_traj'][0], c='green', s=50, alpha=0.7)
        axes[1, 0].scatter(traj['x_traj'][-1], traj['p_traj'][-1], c='red', s=50, alpha=0.7)
    
    axes[1, 0].set_xlabel('Position x')
    axes[1, 0].set_ylabel('Momentum p')
    axes[1, 0].set_title('Phase Space Trajectories')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Energy evolution during single trajectory
    sample_traj = trajectory_data[0]
    x_traj = sample_traj['x_traj']
    p_traj = sample_traj['p_traj']
    
    kinetic_energy = [0.5 * p**2 for p in p_traj]
    potential_energy = [0.5 * x**2 for x in x_traj]
    total_energy = [k + v for k, v in zip(kinetic_energy, potential_energy)]
    
    steps = range(len(x_traj))
    axes[1, 1].plot(steps, kinetic_energy, 'r-', label='Kinetic', linewidth=2)
    axes[1, 1].plot(steps, potential_energy, 'b-', label='Potential', linewidth=2)
    axes[1, 1].plot(steps, total_energy, 'k--', label='Total', linewidth=2)
    axes[1, 1].set_xlabel('Leapfrog Step')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Energy Evolution During Trajectory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Energy Conservation Analysis', fontsize=16)
    plt.tight_layout()
    
    energy_path = plot_dir / "energy_conservation.png"
    plt.savefig(energy_path, dpi=300, bbox_inches='tight')
    print(f"\\nEnergy conservation plot saved to: {energy_path}")
    plt.close()
    
    # Print statistics
    print(f"\\nEnergy Conservation Statistics:")
    print(f"  Mean ΔE: {np.mean(energy_diffs):.6f}")
    print(f"  Std ΔE: {np.std(energy_diffs):.6f}")
    print(f"  Max |ΔE|: {np.max(np.abs(energy_diffs)):.6f}")
    
    return trajectory_data

def hmc_vs_metropolis_demo():
    """Compare HMC with Metropolis algorithm."""
    print("\\n" + "=" * 60)
    print("HMC VS METROPOLIS COMPARISON")
    print("=" * 60)
    
    plot_dir = create_plot_dir()
    
    # Create HMC sampler
    hmc = HMC(
        n_dim=1,
        step_size=0.1,
        n_steps=50,
        potential=lambda x: 0.5 * x[0]**2,
        gradient=lambda x: np.array([x[0]])
    )
    
    # Create Metropolis sampler (simple implementation)
    def metropolis_step(x, step_size):
        x_new = x + step_size * np.random.randn(1)
        
        # Accept/reject
        energy_diff = 0.5 * (x_new[0]**2 - x[0]**2)
        accept_prob = min(1, np.exp(-energy_diff))
        
        if np.random.rand() < accept_prob:
            return x_new, True
        else:
            return x, False
    
    # Run both algorithms
    n_samples = 2000
    
    # HMC
    print("\\nRunning HMC...")
    hmc_samples = []
    hmc_acceptances = []
    x = np.array([0.0])
    
    start_time = time.time()
    for i in range(n_samples):
        x_new, accepted, _ = hmc.step(x)
        hmc_samples.append(x_new[0])
        hmc_acceptances.append(accepted)
        x = x_new
    hmc_time = time.time() - start_time
    
    # Metropolis
    print("Running Metropolis...")
    metro_samples = []
    metro_acceptances = []
    x = np.array([0.0])
    
    start_time = time.time()
    for i in range(n_samples):
        x_new, accepted = metropolis_step(x, 0.5)
        metro_samples.append(x_new[0])
        metro_acceptances.append(accepted)
        x = x_new
    metro_time = time.time() - start_time
    
    # Calculate autocorrelation times
    hmc_autocorr = integrated_autocorrelation_time(hmc_samples[200:])
    metro_autocorr = integrated_autocorrelation_time(metro_samples[200:])
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sample traces
    axes[0, 0].plot(range(n_samples), hmc_samples, 'b-', alpha=0.7, linewidth=1, label='HMC')
    axes[0, 0].plot(range(n_samples), metro_samples, 'r-', alpha=0.7, linewidth=1, label='Metropolis')
    axes[0, 0].set_xlabel('Sample Number')
    axes[0, 0].set_ylabel('x')
    axes[0, 0].set_title('Sample Traces')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Histograms
    axes[0, 1].hist(hmc_samples[200:], bins=50, density=True, alpha=0.7, 
                   color='blue', label='HMC')
    axes[0, 1].hist(metro_samples[200:], bins=50, density=True, alpha=0.7, 
                   color='red', label='Metropolis')
    
    x_theory = np.linspace(-3, 3, 100)
    p_theory = np.exp(-0.5 * x_theory**2) / np.sqrt(2 * np.pi)
    axes[0, 1].plot(x_theory, p_theory, 'k-', linewidth=2, label='Theoretical')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('Sampling Distributions')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Autocorrelation functions
    max_lag = 100
    hmc_autocorr_func, _ = autocorrelation_function(np.array(hmc_samples[200:]), max_lag)
    metro_autocorr_func, _ = autocorrelation_function(np.array(metro_samples[200:]), max_lag)
    
    lags = range(len(hmc_autocorr_func))
    axes[1, 0].plot(lags, hmc_autocorr_func, 'b-', linewidth=2, label='HMC')
    axes[1, 0].plot(lags, metro_autocorr_func, 'r-', linewidth=2, label='Metropolis')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_title('Autocorrelation Functions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance comparison
    categories = ['Acceptance Rate', 'Autocorr Time', 'Computation Time (s)']
    hmc_values = [np.mean(hmc_acceptances), hmc_autocorr, hmc_time]
    metro_values = [np.mean(metro_acceptances), metro_autocorr, metro_time]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, hmc_values, width, label='HMC', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, metro_values, width, label='Metropolis', alpha=0.7)
    axes[1, 1].set_xlabel('Metric')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categories, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('HMC vs Metropolis Comparison', fontsize=16)
    plt.tight_layout()
    
    comparison_path = plot_dir / "hmc_vs_metropolis.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\\nComparison plot saved to: {comparison_path}")
    plt.close()
    
    # Print comparison statistics
    print(f"\\nComparison Results:")
    print(f"HMC:")
    print(f"  Acceptance rate: {np.mean(hmc_acceptances):.3f}")
    print(f"  Autocorrelation time: {hmc_autocorr:.2f}")
    print(f"  Computation time: {hmc_time:.2f}s")
    print(f"Metropolis:")
    print(f"  Acceptance rate: {np.mean(metro_acceptances):.3f}")
    print(f"  Autocorrelation time: {metro_autocorr:.2f}")
    print(f"  Computation time: {metro_time:.2f}s")
    
    return hmc_samples, metro_samples

def main():
    """Main function to run all demos."""
    print("HYBRID MONTE CARLO - IN-DEPTH DEMO")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots will be saved to: plots/hmc/")
    
    # Run optimization demo first
    step_results, optimal_step = step_size_optimization_demo()
    print(f"\\nOptimal step size determined: {optimal_step:.4f}")
    
    # Run other demos with optimal parameters
    traj_results = trajectory_length_demo(optimal_step)
    energy_data = energy_conservation_demo(optimal_step)
    hmc_samples, metro_samples = hmc_vs_metropolis_demo()
    
    print("\\n" + "=" * 60)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"All plots saved to: plots/hmc/")
    
    # Create summary report
    plot_dir = create_plot_dir()
    summary_path = plot_dir / "demo_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("HYBRID MONTE CARLO - IN-DEPTH DEMO SUMMARY\\n")
        f.write("=" * 50 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("STEP SIZE OPTIMIZATION:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Optimal step size: {optimal_step:.4f}\\n")
        f.write(f"Tested {len(step_results)} different step sizes\\n\\n")
        
        f.write("TRAJECTORY LENGTH STUDY:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Tested {len(traj_results)} different trajectory lengths\\n")
        f.write(f"Range: {min(r['n_steps'] for r in traj_results)} to {max(r['n_steps'] for r in traj_results)} steps\\n\\n")
        
        f.write("ENERGY CONSERVATION:\\n")
        f.write("-" * 25 + "\\n")
        energy_diffs = [t['energy_diff'] for t in energy_data]
        f.write(f"Mean energy difference: {np.mean(energy_diffs):.6f}\\n")
        f.write(f"Energy conservation quality: {np.std(energy_diffs):.6f}\\n\\n")
        
        f.write("HMC VS METROPOLIS:\\n")
        f.write("-" * 25 + "\\n")
        f.write(f"Compared {len(hmc_samples)} samples from each algorithm\\n")
        f.write(f"Detailed autocorrelation and efficiency analysis performed\\n")
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
