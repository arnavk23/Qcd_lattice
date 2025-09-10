"""
Hybrid Monte Carlo (HMC) Implementation for 1D Field Theory

This module implements the Hybrid Monte Carlo algorithm for 1D scalar field theory.
HMC combines molecular dynamics with Monte Carlo to achieve better performance
for field theory simulations, especially for critical slowing down.

References:
- Quantum Chromodynamics on the Lattice, Chapter 1
- Creutz article on Monte Carlo methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.field_theory_1d import FieldTheory1D
from src.utils import plot_mcmc_diagnostics, integrated_autocorrelation_time


class HMCFieldTheory1D(FieldTheory1D):
    """
    Hybrid Monte Carlo implementation for 1D scalar field theory.
    
    This class extends the basic FieldTheory1D class to implement HMC,
    which uses Hamilton's equations to propose new configurations.
    """
    
    def __init__(self, lattice_size: int, mass_squared: float, lambda_coupling: float,
                 boundary_conditions: str = 'periodic'):
        """
        Initialize the HMC field theory.
        
        Args:
            lattice_size: Number of lattice sites
            mass_squared: Mass squared parameter (m²)
            lambda_coupling: Quartic coupling constant (λ)
            boundary_conditions: 'periodic' or 'open'
        """
        super().__init__(lattice_size, mass_squared, lambda_coupling, boundary_conditions)
        
        # HMC-specific statistics
        self.hmc_accepted = 0
        self.hmc_total = 0
        self.trajectory_length_history = []
        self.hamiltonian_history = []
        self.energy_violations = []
    
    def compute_force(self, field: np.ndarray) -> np.ndarray:
        """
        Compute the force -∂S/∂φ for the molecular dynamics evolution.
        
        The force is the negative gradient of the action:
        F_i = -∂S/∂φ_i = -∂/∂φ_i [kinetic + potential]
        
        For the kinetic term: 1/2 Σ_j (φ_{j+1} - φ_j)²
        ∂/∂φ_i gives contributions from terms where φ_i appears:
        - From (φ_i - φ_{i-1})²: coefficient +1 on φ_i
        - From (φ_{i+1} - φ_i)²: coefficient -1 on φ_i
        
        Args:
            field: Current field configuration
            
        Returns:
            Force array
        """
        force = np.zeros_like(field)
        
        # Kinetic energy contribution: -∂/∂φ_i [1/2 Σ_j (φ_{j+1} - φ_j)²]
        if self.boundary_conditions == 'periodic':
            # For periodic boundaries, use vectorized operations
            force += 2 * field - np.roll(field, 1) - np.roll(field, -1)
        else:
            # Open boundary conditions
            for i in range(self.N):
                # Contribution from (φ_i - φ_{i-1})² term
                if i > 0:
                    force[i] += (field[i] - field[i-1])
                # Contribution from (φ_{i+1} - φ_i)² term  
                if i < self.N - 1:
                    force[i] -= (field[i+1] - field[i])
        
        # Potential energy contribution: -∂/∂φ_i [1/2 m² φ_i² + λ φ_i⁴]
        force += self.m_squared * field + 4 * self.lambda_coupling * field**3
        
        # Return negative gradient (force is -∂S/∂φ)
        return -force
    
    def hamiltonian(self, field: np.ndarray, momentum: np.ndarray) -> float:
        """
        Compute the Hamiltonian H = T + S, where T is kinetic energy of momenta
        and S is the action (potential energy in the extended phase space).
        
        Args:
            field: Field configuration
            momentum: Momentum configuration
            
        Returns:
            Hamiltonian value
        """
        kinetic_momenta = 0.5 * np.sum(momentum**2)
        potential_action = self.action(field)
        return kinetic_momenta + potential_action
    
    def leapfrog_step(self, field: np.ndarray, momentum: np.ndarray, 
                     step_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform one leapfrog integration step.
        
        The leapfrog algorithm is:
        1. p_{n+1/2} = p_n - (ε/2) * ∂S/∂φ_n
        2. φ_{n+1} = φ_n + ε * p_{n+1/2}
        3. p_{n+1} = p_{n+1/2} - (ε/2) * ∂S/∂φ_{n+1}
        
        Args:
            field: Current field configuration
            momentum: Current momentum configuration
            step_size: Integration step size (ε)
            
        Returns:
            Tuple of (new_field, new_momentum)
        """
        # Step 1: Half momentum update
        force = self.compute_force(field)
        momentum_half = momentum + 0.5 * step_size * force
        
        # Step 2: Full field update
        new_field = field + step_size * momentum_half
        
        # Step 3: Half momentum update
        force_new = self.compute_force(new_field)
        new_momentum = momentum_half + 0.5 * step_size * force_new
        
        return new_field, new_momentum
    
    def test_energy_conservation(self, step_size: float = 0.1, n_steps: int = 10, 
                                n_tests: int = 5) -> list:
        """
        Test energy conservation in molecular dynamics evolution.
        This is a diagnostic tool to verify the correctness of the force computation.
        
        Args:
            step_size: Integration step size
            n_steps: Number of integration steps
            n_tests: Number of independent tests
            
        Returns:
            List of energy violations
        """
        print(f"Testing energy conservation with ε={step_size}, N={n_steps}")
        print("=" * 50)
        
        energy_violations = []
        
        for test in range(n_tests):
            # Random initial configuration
            field = np.random.normal(0, 1, size=self.N)
            momentum = np.random.normal(0, 1, size=self.N)
            
            # Initial energy
            initial_energy = self.hamiltonian(field, momentum)
            
            # Evolve
            final_field, final_momentum = self.molecular_dynamics(
                field, momentum, step_size, n_steps
            )
            
            # Final energy
            final_energy = self.hamiltonian(final_field, final_momentum)
            
            # Energy violation
            delta_h = final_energy - initial_energy
            energy_violations.append(delta_h)
            
            print(f"Test {test+1}: ΔH = {delta_h:.8f}")
        
        print(f"\nSummary:")
        print(f"Mean ΔH: {np.mean(energy_violations):.8f}")
        print(f"RMS ΔH:  {np.sqrt(np.mean(np.array(energy_violations)**2)):.8f}")
        print(f"Max |ΔH|: {np.max(np.abs(energy_violations)):.8f}")
        
        # Good energy conservation should have |ΔH| ~ O(ε²)
        expected_violation = step_size**2
        print(f"Expected ΔH ~ O(ε²) ~ {expected_violation:.8f}")
        
        return energy_violations

    def molecular_dynamics(self, field: np.ndarray, momentum: np.ndarray,
                          step_size: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform molecular dynamics evolution using leapfrog integration.
        
        Args:
            field: Initial field configuration
            momentum: Initial momentum configuration
            step_size: Integration step size
            n_steps: Number of integration steps
            
        Returns:
            Tuple of (final_field, final_momentum)
        """
        current_field = field.copy()
        current_momentum = momentum.copy()
        
        for _ in range(n_steps):
            current_field, current_momentum = self.leapfrog_step(
                current_field, current_momentum, step_size
            )
        
        return current_field, current_momentum
    
    def hmc_step(self, step_size: float, n_md_steps: int) -> Tuple[bool, float]:
        """
        Perform one HMC step.
        
        Args:
            step_size: Molecular dynamics step size
            n_md_steps: Number of molecular dynamics steps
            
        Returns:
            Tuple of (accepted, delta_hamiltonian)
        """
        # Save initial configuration
        initial_field = self.field.copy()
        
        # Generate random momentum
        momentum = np.random.normal(0, 1, size=self.N)
        
        # Compute initial Hamiltonian
        initial_hamiltonian = self.hamiltonian(initial_field, momentum)
        
        # Perform molecular dynamics
        final_field, final_momentum = self.molecular_dynamics(
            initial_field, momentum, step_size, n_md_steps
        )
        
        # Compute final Hamiltonian
        final_hamiltonian = self.hamiltonian(final_field, final_momentum)
        
        # Compute energy change
        delta_h = final_hamiltonian - initial_hamiltonian
        
        # Metropolis acceptance criterion
        if delta_h < 0 or np.random.random() < np.exp(-delta_h):
            # Accept
            self.field = final_field
            self.hmc_accepted += 1
            accepted = True
        else:
            # Reject - keep initial field
            accepted = False
        
        self.hmc_total += 1
        
        # Store diagnostics
        self.hamiltonian_history.append(initial_hamiltonian)
        self.energy_violations.append(delta_h)
        
        return accepted, delta_h
    
    def run_hmc_simulation(self, n_trajectories: int, step_size: float = 0.05,
                          n_md_steps: int = 20, burn_in: int = 1000,
                          measurement_interval: int = 1) -> Dict[str, Any]:
        """
        Run HMC simulation.
        
        Args:
            n_trajectories: Number of HMC trajectories
            step_size: Molecular dynamics step size
            n_md_steps: Number of molecular dynamics steps per trajectory
            burn_in: Number of burn-in trajectories
            measurement_interval: Measure observables every N trajectories
            
        Returns:
            Dictionary containing simulation results
        """
        # Reset statistics
        self.hmc_accepted = 0
        self.hmc_total = 0
        self.field_history = []
        self.action_history = []
        self.observables_history = {'phi_avg': [], 'phi_squared': [], 'phi_fourth': []}
        self.hamiltonian_history = []
        self.energy_violations = []
        
        # Burn-in phase
        for _ in tqdm(range(burn_in), desc="Burn-in"):
            self.hmc_step(step_size, n_md_steps)
        
        # Reset counters after burn-in
        self.hmc_accepted = 0
        self.hmc_total = 0
        
        # Measurement phase
        for traj in tqdm(range(n_trajectories), desc="Trajectories"):
            accepted, delta_h = self.hmc_step(step_size, n_md_steps)
            
            # Measure observables
            if traj % measurement_interval == 0:
                self.measure_observables()
        
        # Compute final results
        results = {
            'acceptance_rate': self.hmc_accepted / self.hmc_total,
            'final_field': self.field.copy(),
            'observables': self.observables_history,
            'hamiltonian_history': self.hamiltonian_history,
            'energy_violations': self.energy_violations,
            'step_size': step_size,
            'n_md_steps': n_md_steps,
            'trajectory_length': step_size * n_md_steps
        }
        
        return results
    
    def plot_hmc_diagnostics(self, figsize: Tuple[int, int] = (14, 10)):
        """Plot HMC-specific diagnostics."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Field configuration
        axes[0, 0].plot(range(self.N), self.field, 'o-', markersize=4)
        axes[0, 0].set_title('Final Field Configuration')
        axes[0, 0].set_xlabel('Lattice Site')
        axes[0, 0].set_ylabel('φ(x)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hamiltonian history
        axes[0, 1].plot(self.hamiltonian_history)
        axes[0, 1].set_title('Hamiltonian History')
        axes[0, 1].set_xlabel('Trajectory')
        axes[0, 1].set_ylabel('H')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy violations
        axes[0, 2].hist(self.energy_violations, bins=50, alpha=0.7)
        axes[0, 2].set_title('Energy Violations (ΔH)')
        axes[0, 2].set_xlabel('ΔH')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Observables
        axes[1, 0].plot(self.observables_history['phi_squared'])
        axes[1, 0].set_title('⟨φ²⟩')
        axes[1, 0].set_xlabel('Measurement')
        axes[1, 0].set_ylabel('⟨φ²⟩')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action history
        axes[1, 1].plot(self.action_history)
        axes[1, 1].set_title('Action History')
        axes[1, 1].set_xlabel('Measurement')
        axes[1, 1].set_ylabel('S[φ]')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Energy violations vs trajectory
        axes[1, 2].plot(self.energy_violations)
        axes[1, 2].set_title('Energy Violations vs Trajectory')
        axes[1, 2].set_xlabel('Trajectory')
        axes[1, 2].set_ylabel('ΔH')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_hmc_results(self):
        """Analyze and print HMC simulation results."""
        print("\nHMC Simulation Results")
        print("=" * 25)
        print(f"Acceptance rate: {self.hmc_accepted / self.hmc_total:.3f}")
        print(f"Mean energy violation: {np.mean(self.energy_violations):.6f}")
        print(f"RMS energy violation: {np.sqrt(np.mean(np.array(self.energy_violations)**2)):.6f}")
        
        # Analyze observables
        for obs_name, obs_data in self.observables_history.items():
            if len(obs_data) > 0:
                obs_array = np.array(obs_data)
                mean_val = np.mean(obs_array)
                std_val = np.std(obs_array)
                tau_int = integrated_autocorrelation_time(obs_array)
                
                print(f"\n{obs_name}:")
                print(f"  Mean: {mean_val:.6f}")
                print(f"  Std:  {std_val:.6f}")
                print(f"  Autocorrelation time: {tau_int:.2f}")


def compare_hmc_metropolis():
    """Compare HMC vs Metropolis algorithms."""
    lattice_size = 50
    mass_squared = 0.5
    lambda_coupling = 0.1
    
    # Run Metropolis
    metropolis = FieldTheory1D(lattice_size, mass_squared, lambda_coupling)
    metropolis_results = metropolis.run_simulation(n_sweeps=10000, burn_in=2000)
    
    # Run HMC
    hmc = HMCFieldTheory1D(lattice_size, mass_squared, lambda_coupling)
    hmc_results = hmc.run_hmc_simulation(n_trajectories=2000, step_size=0.1, 
                                        n_md_steps=10, burn_in=400)
    
    # For φ² observable
    metropolis_phi2 = np.array(metropolis_results['observables']['phi_squared'])
    hmc_phi2 = np.array(hmc_results['observables']['phi_squared'])
    
    tau_metropolis = integrated_autocorrelation_time(metropolis_phi2)
    tau_hmc = integrated_autocorrelation_time(hmc_phi2)
    
    print(f"⟨φ²⟩ Autocorrelation times:")
    print(f"  Metropolis: {tau_metropolis:.2f}")
    print(f"  HMC:        {tau_hmc:.2f}")
    print(f"  Improvement factor: {tau_metropolis/tau_hmc:.2f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Trace plots
    axes[0, 0].plot(metropolis_phi2)
    axes[0, 0].set_title('Metropolis: ⟨φ²⟩ trace')
    axes[0, 0].set_xlabel('Measurement')
    axes[0, 0].set_ylabel('⟨φ²⟩')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(hmc_phi2)
    axes[0, 1].set_title('HMC: ⟨φ²⟩ trace')
    axes[0, 1].set_xlabel('Measurement')
    axes[0, 1].set_ylabel('⟨φ²⟩')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histograms
    axes[1, 0].hist(metropolis_phi2, bins=50, alpha=0.7, label='Metropolis')
    axes[1, 0].hist(hmc_phi2, bins=50, alpha=0.7, label='HMC')
    axes[1, 0].set_title('⟨φ²⟩ distributions')
    axes[1, 0].set_xlabel('⟨φ²⟩')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Autocorrelation functions
    lags_metro, autocorr_metro = [], []
    lags_hmc, autocorr_hmc = [], []
    
    max_lag = min(len(metropolis_phi2)//4, len(hmc_phi2)//4, 100)
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr_metro.append(1.0)
            autocorr_hmc.append(1.0)
        else:
            # Metropolis
            if lag < len(metropolis_phi2):
                corr = np.corrcoef(metropolis_phi2[:-lag], metropolis_phi2[lag:])[0, 1]
                autocorr_metro.append(corr)
            else:
                autocorr_metro.append(0.0)
            
            # HMC
            if lag < len(hmc_phi2):
                corr = np.corrcoef(hmc_phi2[:-lag], hmc_phi2[lag:])[0, 1]
                autocorr_hmc.append(corr)
            else:
                autocorr_hmc.append(0.0)
        
        lags_metro.append(lag)
        lags_hmc.append(lag)
    
    axes[1, 1].plot(lags_metro, autocorr_metro, label='Metropolis')
    axes[1, 1].plot(lags_hmc, autocorr_hmc, label='HMC')
    axes[1, 1].axhline(y=1/np.e, color='r', linestyle='--', alpha=0.5, label='1/e')
    axes[1, 1].set_title('Autocorrelation Functions')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def optimize_hmc_parameters():
    """Optimize HMC parameters (step size and number of MD steps)."""
    lattice_size = 50
    mass_squared = 0.5
    lambda_coupling = 0.1
    
    step_sizes = [0.05, 0.1, 0.15, 0.2]
    md_steps = [5, 10, 15, 20]
    
    results = {}
    
    print("Optimizing HMC parameters...")
    
    for step_size in step_sizes:
        for n_md_steps in md_steps:
            print(f"\nTesting: step_size={step_size}, md_steps={n_md_steps}")
            
            hmc = HMCFieldTheory1D(lattice_size, mass_squared, lambda_coupling)
            result = hmc.run_hmc_simulation(
                n_trajectories=1000, 
                step_size=step_size,
                n_md_steps=n_md_steps,
                burn_in=200
            )
            
            key = f"ε={step_size}_N={n_md_steps}"
            results[key] = {
                'acceptance_rate': result['acceptance_rate'],
                'energy_violation_rms': np.sqrt(np.mean(np.array(result['energy_violations'])**2)),
                'step_size': step_size,
                'n_md_steps': n_md_steps,
                'trajectory_length': step_size * n_md_steps
            }
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Acceptance rate vs trajectory length
    traj_lengths = [results[key]['trajectory_length'] for key in results]
    acceptance_rates = [results[key]['acceptance_rate'] for key in results]
    
    axes[0].scatter(traj_lengths, acceptance_rates)
    axes[0].set_xlabel('Trajectory Length')
    axes[0].set_ylabel('Acceptance Rate')
    axes[0].set_title('Acceptance Rate vs Trajectory Length')
    axes[0].grid(True, alpha=0.3)
    
    # Energy violation vs trajectory length
    energy_violations = [results[key]['energy_violation_rms'] for key in results]
    
    axes[1].scatter(traj_lengths, energy_violations)
    axes[1].set_xlabel('Trajectory Length')
    axes[1].set_ylabel('RMS Energy Violation')
    axes[1].set_title('Energy Violation vs Trajectory Length')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print optimal parameters
    print("\nOptimization Results:")
    print("=" * 25)
    for key, result in results.items():
        print(f"{key}: acc={result['acceptance_rate']:.3f}, ΔH_rms={result['energy_violation_rms']:.4f}")


if __name__ == "__main__":
    # Example usage
    print("HMC 1D Field Theory Simulation")
    print("=" * 35)
    
    # Create HMC field theory instance
    hmc = HMCFieldTheory1D(
        lattice_size=50,
        mass_squared=0.5,
        lambda_coupling=0.1
    )
    
    # Run HMC simulation
    results = hmc.run_hmc_simulation(
        n_trajectories=5000,
        step_size=0.1,
        n_md_steps=10,
        burn_in=1000
    )
    
    # Analyze results
    hmc.analyze_hmc_results()
    
    # Plot diagnostics
    hmc.plot_hmc_diagnostics()
    
    # Compare with Metropolis
    compare_hmc_metropolis()
    
    # Optimize parameters
    optimize_hmc_parameters()
