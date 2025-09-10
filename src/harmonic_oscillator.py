"""
Harmonic Oscillator Implementation using Metropolis Algorithm

This module implements the quantum harmonic oscillator on a Euclidean lattice
following Creutz's paper. The path integral formulation gives us:

Z = ∫ Dq exp(-S[q])

where the action for a harmonic oscillator is:
S[q] = (1/2) Σ_t [m(q_{t+1} - q_t)² + k q_t²]

where:
- m: mass parameter
- k: spring constant (ω² in natural units)
- q_t: position at time t
- The kinetic term comes from the discretized time derivative

The equilibrium distribution should give ⟨q⟩ = 0 for the harmonic oscillator.

References:
- Creutz, "Monte Carlo Study of Quantized SU(2) Gauge Theory"
- Creutz, "Quantum fields on the computer"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils import *


class HarmonicOscillatorMC:
    """
    Monte Carlo simulation of the quantum harmonic oscillator using Metropolis algorithm.
    
    This class implements the path integral formulation of the quantum harmonic oscillator
    on a Euclidean lattice, following Creutz's methodology.
    """
    
    def __init__(self, n_time_steps: int, mass: float = 1.0, omega: float = 1.0, 
                 lattice_spacing: float = 0.1):
        """
        Initialize the harmonic oscillator Monte Carlo simulation.
        
        Args:
            n_time_steps: Number of time steps in the path
            mass: Mass of the oscillator
            omega: Angular frequency (natural frequency)
            lattice_spacing: Temporal lattice spacing (Δt)
        """
        self.N = n_time_steps
        self.m = mass
        self.omega = omega
        self.dt = lattice_spacing
        
        # Derived parameters
        self.k = mass * omega**2  # Spring constant
        
        # Initialize path (position as function of time)
        self.path = np.random.normal(0, 1, size=self.N)
        
        # Statistics
        self.accepted = 0
        self.total_steps = 0
        
        # Observables history
        self.observables_history = {
            'position': [],
            'position_squared': [],
            'kinetic_energy': [],
            'potential_energy': [],
            'total_energy': []
        }
    
    def kinetic_action(self, path: np.ndarray) -> float:
        """
        Compute the kinetic part of the action.
        
        For the harmonic oscillator, the kinetic term in the action is:
        S_kinetic = (m/2Δt) * Σ_t (q_{t+1} - q_t)²
        
        Args:
            path: Position path q(t)
            
        Returns:
            Kinetic action contribution
        """
        # Periodic boundary conditions: q(N) = q(0)
        path_shifted = np.roll(path, -1)
        kinetic = 0.5 * self.m * np.sum((path_shifted - path)**2) / self.dt
        return kinetic
    
    def potential_action(self, path: np.ndarray) -> float:
        """
        Compute the potential part of the action.
        
        For the harmonic oscillator:
        S_potential = (m ω²/2) * Δt * Σ_t q_t²
        
        Args:
            path: Position path q(t)
            
        Returns:
            Potential action contribution
        """
        potential = 0.5 * self.m * self.omega**2 * self.dt * np.sum(path**2)
        return potential
    
    def action(self, path: np.ndarray) -> float:
        """
        Compute the total action S = S_kinetic + S_potential.
        
        Args:
            path: Position path q(t)
            
        Returns:
            Total action
        """
        return self.kinetic_action(path) + self.potential_action(path)
    
    def local_action_change(self, time_index: int, old_position: float, 
                           new_position: float) -> float:
        """
        Compute the change in action when updating a single time step.
        
        This is more efficient than computing the full action.
        
        Args:
            time_index: Time index to update
            old_position: Current position at time_index
            new_position: Proposed new position
            
        Returns:
            Change in action (S_new - S_old)
        """
        t = time_index
        
        # Get neighboring time indices (periodic boundary conditions)
        t_prev = (t - 1) % self.N
        t_next = (t + 1) % self.N
        
        # Old kinetic energy contributions
        old_kinetic = (0.5 * self.m / self.dt) * (
            (old_position - self.path[t_prev])**2 + 
            (self.path[t_next] - old_position)**2
        )
        
        # New kinetic energy contributions
        new_kinetic = (0.5 * self.m / self.dt) * (
            (new_position - self.path[t_prev])**2 + 
            (self.path[t_next] - new_position)**2
        )
        
        # Potential energy change
        old_potential = 0.5 * self.m * self.omega**2 * self.dt * old_position**2
        new_potential = 0.5 * self.m * self.omega**2 * self.dt * new_position**2
        
        return (new_kinetic - old_kinetic) + (new_potential - old_potential)
    
    def metropolis_step(self, step_size: float) -> bool:
        """
        Perform one Metropolis step by updating a random time step.
        
        Args:
            step_size: Step size for position update
            
        Returns:
            True if move was accepted
        """
        # Choose random time step
        t = np.random.randint(0, self.N)
        
        # Current position
        old_position = self.path[t]
        
        # Propose new position
        new_position = old_position + np.random.uniform(-step_size, step_size)
        
        # Compute action change
        delta_action = self.local_action_change(t, old_position, new_position)
        
        # Metropolis acceptance criterion
        if delta_action < 0 or np.random.random() < np.exp(-delta_action):
            self.path[t] = new_position
            self.accepted += 1
            self.total_steps += 1
            return True
        else:
            self.total_steps += 1
            return False
    
    def metropolis_sweep(self, step_size: float) -> int:
        """
        Perform one complete sweep through all time steps.
        
        Args:
            step_size: Step size for position updates
            
        Returns:
            Number of accepted moves in this sweep
        """
        accepted_in_sweep = 0
        for _ in range(self.N):
            if self.metropolis_step(step_size):
                accepted_in_sweep += 1
        return accepted_in_sweep
    
    def measure_observables(self):
        """Measure and store observables."""
        # Position average
        position_avg = np.mean(self.path)
        
        # Position squared
        position_squared = np.mean(self.path**2)
        
        # Kinetic energy (from the action, not the Hamiltonian)
        # This is the discretized version of the kinetic energy
        path_shifted = np.roll(self.path, -1)
        kinetic_energy = 0.5 * self.m * np.mean((path_shifted - self.path)**2) / self.dt**2
        
        # Potential energy
        potential_energy = 0.5 * self.m * self.omega**2 * np.mean(self.path**2)
        
        # Total energy (note: this is not the Hamiltonian energy)
        total_energy = kinetic_energy + potential_energy
        
        # Store observables
        self.observables_history['position'].append(position_avg)
        self.observables_history['position_squared'].append(position_squared)
        self.observables_history['kinetic_energy'].append(kinetic_energy)
        self.observables_history['potential_energy'].append(potential_energy)
        self.observables_history['total_energy'].append(total_energy)
    
    def run_simulation(self, n_sweeps: int, step_size: float = 0.5, 
                      burn_in: int = 1000, measurement_interval: int = 1) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation.
        
        Args:
            n_sweeps: Number of sweeps to perform
            step_size: Step size for position updates
            burn_in: Number of burn-in sweeps
            measurement_interval: Measure observables every N sweeps
            
        Returns:
            Dictionary containing simulation results
        """
        # Reset statistics
        self.accepted = 0
        self.total_steps = 0
        self.observables_history = {
            'position': [],
            'position_squared': [],
            'kinetic_energy': [],
            'potential_energy': [],
            'total_energy': []
        }
        
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
            'final_path': self.path.copy(),
            'observables': self.observables_history
        }
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]):
        """Analyze and print simulation results."""
        # Theoretical expectations
        theoretical_energy = 0.5 * self.omega  # Ground state energy
        theoretical_position_sq = 0.5 / self.omega  # ⟨q²⟩ for ground state
        
        # Position average (should be 0)
        pos_data = np.array(results['observables']['position'])
        pos_mean = np.mean(pos_data)
        pos_err = np.std(pos_data) / np.sqrt(len(pos_data))
        
        # Position squared
        pos_sq_data = np.array(results['observables']['position_squared'])
        pos_sq_mean = np.mean(pos_sq_data)
        pos_sq_err = np.std(pos_sq_data) / np.sqrt(len(pos_sq_data))
        pos_sq_diff = abs(pos_sq_mean - theoretical_position_sq)
        
        # Total energy
        energy_data = np.array(results['observables']['total_energy'])
        energy_mean = np.mean(energy_data)
        energy_err = np.std(energy_data) / np.sqrt(len(energy_data))
        energy_diff = abs(energy_mean - theoretical_energy)
        
        # Autocorrelation analysis
        tau_pos = integrated_autocorrelation_time(pos_data)
        tau_energy = integrated_autocorrelation_time(energy_data)
        
        # Validation checks
        pos_check = abs(pos_mean) < 3 * pos_err  # Position should be consistent with 0
        energy_check = energy_diff < 0.1  # Energy should be close to theoretical
        
        return pos_check and energy_check
    
    def plot_results(self, results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8), 
                    save_path: Optional[str] = None):
        """Plot simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Path configuration
        axes[0, 0].plot(range(self.N), results['final_path'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Final Path Configuration')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Position q(t)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Position time series
        pos_data = results['observables']['position']
        axes[0, 1].plot(pos_data, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='⟨q⟩ = 0')
        axes[0, 1].set_title('Position Average vs Time')
        axes[0, 1].set_xlabel('Measurement')
        axes[0, 1].set_ylabel('⟨q⟩')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy time series
        energy_data = results['observables']['total_energy']
        axes[1, 0].plot(energy_data, alpha=0.7)
        theoretical_energy = 0.5 * self.omega
        axes[1, 0].axhline(y=theoretical_energy, color='r', linestyle='--', 
                          alpha=0.5, label=f'E₀ = {theoretical_energy:.3f}')
        axes[1, 0].set_title('Total Energy vs Time')
        axes[1, 0].set_xlabel('Measurement')
        axes[1, 0].set_ylabel('⟨E⟩')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Position histogram
        axes[1, 1].hist(results['final_path'], bins=30, density=True, alpha=0.7, 
                       color='skyblue', label='Simulation')
        
        # Theoretical ground state wavefunction |ψ₀(q)|²
        q_theory = np.linspace(results['final_path'].min(), results['final_path'].max(), 100)
        psi_squared = np.sqrt(self.omega / np.pi) * np.exp(-self.omega * q_theory**2)
        axes[1, 1].plot(q_theory, psi_squared, 'r-', linewidth=2, 
                       label='|ψ₀(q)|²')
        axes[1, 1].set_title('Position Distribution')
        axes[1, 1].set_xlabel('Position q')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        plt.close()
