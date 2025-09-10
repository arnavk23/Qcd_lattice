"""
1D Field Theory Implementation using Metropolis Algorithm

This module implements a 1D scalar field theory on a lattice using the Metropolis algorithm.
This follows the approach outlined in Chapter 1 of "Quantum Chromodynamics on the Lattice"
and the Creutz article.

The action for a 1D scalar field φ(x) is:
S[φ] = Σ_x [1/2 (φ(x+1) - φ(x))² + 1/2 m² φ(x)² + λ φ(x)⁴]

where:
- The first term is the kinetic energy (discrete derivative)
- The second term is the mass term
- The third term is the self-interaction (quartic potential)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import plot_mcmc_diagnostics, autocorrelation_function, integrated_autocorrelation_time


class FieldTheory1D:
    """
    1D scalar field theory implementation using Metropolis algorithm.
    
    This class implements the Metropolis algorithm for a 1D scalar field theory
    with action S[φ] = Σ_x [1/2 (φ(x+1) - φ(x))² + 1/2 m² φ(x)² + λ φ(x)⁴]
    """
    
    def __init__(self, lattice_size: int, mass_squared: float, lambda_coupling: float, 
                 boundary_conditions: str = 'periodic'):
        """
        Initialize the 1D field theory.
        
        Args:
            lattice_size: Number of lattice sites
            mass_squared: Mass squared parameter (m²)
            lambda_coupling: Quartic coupling constant (λ)
            boundary_conditions: 'periodic' or 'open'
        """
        self.N = lattice_size
        self.m_squared = mass_squared
        self.lambda_coupling = lambda_coupling
        self.boundary_conditions = boundary_conditions
        
        # Initialize field configuration
        self.field = np.random.normal(0, 1, size=self.N)
        
        # Statistics
        self.accepted = 0
        self.total_steps = 0
        
        # Observables history
        self.field_history = []
        self.action_history = []
        self.observables_history = {}
    
    def kinetic_energy(self, field: np.ndarray) -> float:
        """
        Compute the kinetic energy part of the action.
        
        Args:
            field: Field configuration
            
        Returns:
            Kinetic energy contribution
        """
        if self.boundary_conditions == 'periodic':
            # Periodic boundary conditions: φ(N) = φ(0)
            kinetic = 0.5 * np.sum((np.roll(field, -1) - field)**2)
        else:
            # Open boundary conditions
            kinetic = 0.5 * np.sum((field[1:] - field[:-1])**2)
        
        return kinetic
    
    def potential_energy(self, field: np.ndarray) -> float:
        """
        Compute the potential energy part of the action.
        
        Args:
            field: Field configuration
            
        Returns:
            Potential energy contribution
        """
        mass_term = 0.5 * self.m_squared * np.sum(field**2)
        interaction_term = self.lambda_coupling * np.sum(field**4)
        return mass_term + interaction_term
    
    def action(self, field: np.ndarray) -> float:
        """
        Compute the total action for a field configuration.
        
        Args:
            field: Field configuration
            
        Returns:
            Total action
        """
        return self.kinetic_energy(field) + self.potential_energy(field)
    
    def local_action_change(self, site: int, old_value: float, new_value: float) -> float:
        """
        Compute the change in action when updating a single site.
        
        This is more efficient than computing the full action for each update.
        
        Args:
            site: Site index to update
            old_value: Current field value at site
            new_value: Proposed field value at site
            
        Returns:
            Change in action (S_new - S_old)
        """
        # Get neighboring sites
        if self.boundary_conditions == 'periodic':
            left = (site - 1) % self.N
            right = (site + 1) % self.N
        else:
            left = site - 1 if site > 0 else None
            right = site + 1 if site < self.N - 1 else None
        
        # Kinetic energy change
        kinetic_change = 0.0
        
        # Contribution from (φ(site) - φ(left))²
        if left is not None:
            kinetic_change += 0.5 * ((new_value - self.field[left])**2 - (old_value - self.field[left])**2)
        
        # Contribution from (φ(right) - φ(site))²
        if right is not None:
            kinetic_change += 0.5 * ((self.field[right] - new_value)**2 - (self.field[right] - old_value)**2)
        
        # Potential energy change
        potential_change = 0.5 * self.m_squared * (new_value**2 - old_value**2)
        potential_change += self.lambda_coupling * (new_value**4 - old_value**4)
        
        return kinetic_change + potential_change
    
    def metropolis_step(self, step_size: float) -> bool:
        """
        Perform one Metropolis step by updating a random site.
        
        Args:
            step_size: Step size for field update
            
        Returns:
            True if move was accepted
        """
        # Choose random site
        site = np.random.randint(0, self.N)
        
        # Current field value
        old_value = self.field[site]
        
        # Propose new value
        new_value = old_value + np.random.uniform(-step_size, step_size)
        
        # Compute action change
        delta_action = self.local_action_change(site, old_value, new_value)
        
        # Metropolis acceptance criterion
        if delta_action < 0 or np.random.random() < np.exp(-delta_action):
            self.field[site] = new_value
            self.accepted += 1
            self.total_steps += 1
            return True
        else:
            self.total_steps += 1
            return False
    
    def metropolis_sweep(self, step_size: float) -> int:
        """
        Perform one complete sweep through the lattice.
        
        Args:
            step_size: Step size for field updates
            
        Returns:
            Number of accepted moves in this sweep
        """
        accepted_in_sweep = 0
        for _ in range(self.N):
            if self.metropolis_step(step_size):
                accepted_in_sweep += 1
        return accepted_in_sweep
    
    def run_simulation(self, n_sweeps: int, step_size: float = 0.5, 
                      burn_in: int = 1000, measurement_interval: int = 1) -> dict:
        """
        Run the Monte Carlo simulation.
        
        Args:
            n_sweeps: Number of sweeps to perform
            step_size: Step size for field updates
            burn_in: Number of burn-in sweeps
            measurement_interval: Measure observables every N sweeps
            
        Returns:
            Dictionary containing simulation results
        """
        # Reset statistics
        self.accepted = 0
        self.total_steps = 0
        self.field_history = []
        self.action_history = []
        self.observables_history = {'phi_avg': [], 'phi_squared': [], 'phi_fourth': []}
        
        # Burn-in phase
        for _ in tqdm(range(burn_in), desc="Burn-in"):
            self.metropolis_sweep(step_size)
        
        # Reset counters after burn-in
        self.accepted = 0
        self.total_steps = 0
        
        # Measurement phase
        for sweep in tqdm(range(n_sweeps), desc="Sweeps"):
            self.metropolis_sweep(step_size)
            
            # Measure observables
            if sweep % measurement_interval == 0:
                self.measure_observables()
        
        # Compute final results
        results = {
            'acceptance_rate': self.accepted / self.total_steps,
            'final_field': self.field.copy(),
            'observables': self.observables_history,
            'action_history': self.action_history
        }
        
        return results
    
    def measure_observables(self):
        """Measure and store observables."""
        # Average field value
        phi_avg = np.mean(self.field)
        
        # Field squared
        phi_squared = np.mean(self.field**2)
        
        # Field fourth power
        phi_fourth = np.mean(self.field**4)
        
        # Store in history
        self.observables_history['phi_avg'].append(phi_avg)
        self.observables_history['phi_squared'].append(phi_squared)
        self.observables_history['phi_fourth'].append(phi_fourth)
        
        # Store current action
        current_action = self.action(self.field)
        self.action_history.append(current_action)
        
        # Store field configuration (optional, memory intensive)
        # self.field_history.append(self.field.copy())
    
    def compute_correlation_function(self, max_distance: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the two-point correlation function ⟨φ(0)φ(r)⟩.
        
        Args:
            max_distance: Maximum distance to compute (default: N//2)
            
        Returns:
            Tuple of (distances, correlations)
        """
        if max_distance is None:
            max_distance = self.N // 2
        
        distances = np.arange(max_distance + 1)
        correlations = []
        
        for r in distances:
            if r == 0:
                # Auto-correlation
                corr = np.mean(self.field**2)
            else:
                # Cross-correlation
                if self.boundary_conditions == 'periodic':
                    corr = np.mean(self.field * np.roll(self.field, r))
                else:
                    if r < self.N:
                        corr = np.mean(self.field[:-r] * self.field[r:])
                    else:
                        corr = 0.0
            
            correlations.append(corr)
        
        return distances, np.array(correlations)
    
    def plot_field_configuration(self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None):
        """Plot the current field configuration."""
        plt.figure(figsize=figsize)
        plt.plot(range(self.N), self.field, 'o-', markersize=4)
        plt.xlabel('Lattice Site')
        plt.ylabel('Field Value φ(x)')
        plt.title('1D Field Configuration')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_observables(self, figsize: Tuple[int, int] = (12, 8), save_path: Optional[str] = None):
        """Plot the time evolution of observables."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Field average
        axes[0, 0].plot(self.observables_history['phi_avg'])
        axes[0, 0].set_title('⟨φ⟩')
        axes[0, 0].set_xlabel('Measurement')
        axes[0, 0].set_ylabel('⟨φ⟩')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Field squared
        axes[0, 1].plot(self.observables_history['phi_squared'])
        axes[0, 1].set_title('⟨φ²⟩')
        axes[0, 1].set_xlabel('Measurement')
        axes[0, 1].set_ylabel('⟨φ²⟩')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Field fourth power
        axes[1, 0].plot(self.observables_history['phi_fourth'])
        axes[1, 0].set_title('⟨φ⁴⟩')
        axes[1, 0].set_xlabel('Measurement')
        axes[1, 0].set_ylabel('⟨φ⁴⟩')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Action
        axes[1, 1].plot(self.action_history)
        axes[1, 1].set_title('Action')
        axes[1, 1].set_xlabel('Measurement')
        axes[1, 1].set_ylabel('S[φ]')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_correlation_function(self, figsize: Tuple[int, int] = (8, 6)):
        """Plot the two-point correlation function."""
        distances, correlations = self.compute_correlation_function()
        
        plt.figure(figsize=figsize)
        plt.plot(distances, correlations, 'o-', markersize=4)
        plt.xlabel('Distance r')
        plt.ylabel('⟨φ(0)φ(r)⟩')
        plt.title('Two-Point Correlation Function')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def analyze_results(self):
        """Analyze and print simulation results."""
        print("\n1D Field Theory Simulation Results")
        print("=" * 40)
        print(f"Lattice size: {self.N}")
        print(f"Mass squared: {self.m_squared}")
        print(f"Lambda coupling: {self.lambda_coupling}")
        print(f"Boundary conditions: {self.boundary_conditions}")
        print(f"Acceptance rate: {self.accepted / self.total_steps:.3f}")
        
        # Compute statistics for observables
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


def compare_parameters():
    """Compare different parameter settings."""
    parameter_sets = [
        {'m_squared': 0.1, 'lambda_coupling': 0.1, 'label': 'Light, weak coupling'},
        {'m_squared': 1.0, 'lambda_coupling': 0.1, 'label': 'Heavy, weak coupling'},
        {'m_squared': 0.1, 'lambda_coupling': 1.0, 'label': 'Light, strong coupling'},
        {'m_squared': 1.0, 'lambda_coupling': 1.0, 'label': 'Heavy, strong coupling'}
    ]
    
    lattice_size = 50
    n_sweeps = 5000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, params in enumerate(parameter_sets):
        # Create and run simulation
        field_theory = FieldTheory1D(
            lattice_size=lattice_size,
            mass_squared=params['m_squared'],
            lambda_coupling=params['lambda_coupling']
        )
        
        results = field_theory.run_simulation(n_sweeps=n_sweeps, burn_in=1000)
        
        # Plot field configuration
        axes[i].plot(range(lattice_size), field_theory.field, 'o-', markersize=3)
        axes[i].set_title(f"{params['label']}\n⟨φ²⟩ = {np.mean(results['observables']['phi_squared']):.3f}")
        axes[i].set_xlabel('Lattice Site')
        axes[i].set_ylabel('φ(x)')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("1D Field Theory Simulation")
    print("=" * 30)
    
    # Create field theory instance
    field_theory = FieldTheory1D(
        lattice_size=50,
        mass_squared=0.5,
        lambda_coupling=0.1
    )
    
    # Run simulation
    results = field_theory.run_simulation(
        n_sweeps=10000,
        step_size=0.5,
        burn_in=2000,
        measurement_interval=1
    )
    
    # Analyze results
    field_theory.analyze_results()
    
    # Plot results
    field_theory.plot_field_configuration()
    field_theory.plot_observables()
    field_theory.plot_correlation_function()
    
    # Compare different parameters
    print("\nComparing different parameter settings...")
    compare_parameters()
