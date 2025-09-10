# Path Integral Formalism and Lattice Field Theory

## Theoretical Foundations

This document provides the theoretical background for understanding lattice field theory and the path integral approach to quantum mechanics and quantum field theory.

## 1. Path Integral Formulation of Quantum Mechanics

### 1.1 From Schrödinger to Feynman

The path integral formulation provides an alternative to the canonical approach to quantum mechanics. Instead of operators and wavefunctions, we consider all possible paths a particle can take between two points.

**Key Concept**: The quantum amplitude for a particle to go from point A to point B is given by summing over all possible paths, weighted by the exponential of the classical action.

```
⟨x_f, t_f | x_i, t_i⟩ = ∫ D[x(t)] exp(iS[x(t)]/ℏ)
```

### 1.2 Euclidean Path Integrals

In statistical mechanics and lattice field theory, we often work with **Euclidean** (imaginary time) path integrals:

```
Z = ∫ D[φ] exp(-S_E[φ])
```

where `S_E` is the Euclidean action. This formulation:
- Converts oscillatory integrals to convergent Gaussian-like integrals
- Provides natural connection to statistical mechanics
- Enables Monte Carlo sampling methods

### 1.3 Harmonic Oscillator Example

For a quantum harmonic oscillator, the Euclidean action is:

```
S_E = ∫₀^β dτ [m/2 (dx/dτ)² + mω²/2 x²]
```

**Lattice Discretization**:
```
S_E = Σₜ [m/(2Δτ) (x_{t+1} - x_t)² + mω²Δτ/2 x_t²]
```

## 2. Monte Carlo Methods in Quantum Field Theory

### 2.1 Why Monte Carlo?

Path integrals involve infinite-dimensional integrations that cannot be solved analytically for most interesting theories. Monte Carlo methods provide a way to:

1. **Sample** field configurations according to their probability weight
2. **Estimate** expectation values of observables
3. **Study** non-perturbative physics

### 2.2 Importance Sampling

Instead of random sampling, we use **importance sampling** where configurations are generated with probability proportional to their Boltzmann weight:

```
P[φ] ∝ exp(-S[φ])
```

### 2.3 Observable Measurement

Once we have a representative sample of configurations, observables are calculated as:

```
⟨O⟩ = (1/N) Σᵢ O[φᵢ]
```

## 3. Markov Chain Monte Carlo (MCMC)

### 3.1 Markov Chain Basics

A Markov chain is a sequence of configurations where each new configuration depends only on the current one, not the history:

```
P(φₙ₊₁ | φₙ, φₙ₋₁, ..., φ₁) = P(φₙ₊₁ | φₙ)
```

### 3.2 Detailed Balance

For correct sampling, the transition probabilities must satisfy **detailed balance**:

```
P(φ → φ') π(φ) = P(φ' → φ) π(φ')
```

where π(φ) is the target distribution.

### 3.3 Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm ensures detailed balance by:

1. **Propose** a new configuration φ' from φ
2. **Calculate** acceptance probability: `A = min(1, exp(-(S[φ'] - S[φ])))`
3. **Accept** or reject the proposal based on A

## 4. Critical Slowing Down

### 4.1 Autocorrelation and Critical Slowing Down

Near phase transitions, the **autocorrelation time** τ diverges:

```
τ ∼ ξᶻ
```

where ξ is the correlation length and z is the **dynamic critical exponent**.

### 4.2 Consequences

- **Longer simulations** needed for decorrelated samples
- **Increased computational cost** near critical points
- **Poor scaling** with system size

### 4.3 Mitigation Strategies

1. **Improved Algorithms**:
   - Hybrid Monte Carlo (HMC)
   - Overrelaxation
   - Multi-grid methods

2. **Machine Learning Approaches**:
   - Normalizing flows
   - Neural network parametrizations
   - Autoregressive models

3. **Fourier Acceleration**:
   - Momentum-space updates
   - Preconditioning techniques

## 5. Hybrid Monte Carlo (HMC)

### 5.1 Motivation

HMC addresses the problem of random walk behavior in Metropolis by:
- Using **molecular dynamics** for deterministic evolution
- **Reducing autocorrelations** through global updates
- **Maintaining exact sampling** through Metropolis accept/reject

### 5.2 Algorithm

1. **Refresh momenta**: p ~ exp(-½p²)
2. **Molecular dynamics**: Evolve (φ, p) using Hamiltonian dynamics
3. **Metropolis step**: Accept/reject based on energy difference

### 5.3 Advantages

- **Reduced autocorrelation** times
- **Global updates** instead of local
- **Scalable** to larger systems

## 6. Lattice Gauge Theory Introduction

### 6.1 From Continuum to Lattice

Gauge theories require special treatment on the lattice:
- **Gauge invariance** must be preserved
- **Link variables** U_{μ}(x) replace gauge fields
- **Wilson action** for pure gauge theory

### 6.2 Wilson Action

```
S_W = β Σ_P [1 - (1/N_c) Re Tr U_P]
```

where U_P is the plaquette (smallest closed loop).

### 6.3 Fermionic Fields

Fermions on the lattice face additional challenges:
- **Fermion doubling** problem
- **Chiral symmetry** breaking
- **Improved actions** (Wilson, Domain Wall, Overlap)

## 7. Educational Exercises

### 7.1 Basic Exercises

1. **Implement** simple Metropolis for 1D Ising model
2. **Study** autocorrelation functions
3. **Measure** critical exponents

### 7.2 Intermediate Exercises

1. **Compare** Metropolis vs HMC for scalar field
2. **Investigate** step-size optimization
3. **Analyze** critical slowing down

### 7.3 Advanced Projects

1. **Implement** gauge theory simulations
2. **Apply** machine learning acceleration
3. **Study** finite temperature transitions

## References

1. Creutz, M. "Quarks, Gluons and Lattices"
2. Rothe, H.J. "Lattice Gauge Theories: An Introduction"
3. Gattringer, C. & Lang, C.B. "Quantum Chromodynamics on the Lattice"
4. Berg, B. "Markov Chain Monte Carlo Simulations and Their Statistical Analysis"

## Learning Outcomes

Upon completion of this module, students should be able to:

1. **Understand** the path integral formulation of quantum mechanics
2. **Implement** Monte Carlo algorithms for quantum systems
3. **Recognize** and mitigate critical slowing down
4. **Apply** lattice methods to specific field theories
5. **Analyze** statistical data from simulations
6. **Appreciate** the connection between statistical mechanics and quantum field theory
