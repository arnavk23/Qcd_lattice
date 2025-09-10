# Educational Exercises and Tutorials

This document provides a structured learning path for students studying lattice QCD and Monte Carlo methods.

## Learning Path Structure

### Phase 1: Foundations (Weeks 1-2)
**Objective**: Understand basic concepts and simple implementations

### Phase 2: Monte Carlo Methods (Weeks 3-4)
**Objective**: Master MCMC algorithms and statistical analysis

### Phase 3: Advanced Techniques (Weeks 5-6)
**Objective**: Learn acceleration methods and modern approaches

### Phase 4: Applications (Weeks 7-8)
**Objective**: Apply methods to realistic physical problems

---

## Phase 1: Foundations

### Exercise 1.1: Path Integral for Free Particle
**Duration**: 2 hours  
**Difficulty**: Beginner

**Task**: Implement the path integral for a free particle in 1D
```python
# Template provided in notebooks/exercise_1_1.ipynb
def free_particle_path_integral(x_initial, x_final, time, mass, n_paths=1000):
    """
    Calculate transition amplitude for free particle using path integral
    """
    # Student implementation here
    pass
```

**Learning Goals**:
- Understand discretization of path integrals
- Practice numerical integration
- Visualize different paths and their contributions

### Exercise 1.2: Harmonic Oscillator Path Integral
**Duration**: 3 hours  
**Difficulty**: Beginner-Intermediate

**Task**: Extend to harmonic oscillator and compare with analytical result
- Use the provided `HarmonicOscillatorMC` class
- Verify ⟨q⟩ = 0 and ⟨q²⟩ = 1/(2ω)
- Study the effect of lattice spacing

**Key Code**:
```python
from src.harmonic_oscillator import HarmonicOscillatorMC

# Student implementation
oscillator = HarmonicOscillatorMC(n_time_steps=50, mass=1.0, omega=1.0)
results = oscillator.run_simulation(n_sweeps=5000, step_size=0.3)

# Analysis tasks:
# 1. Plot ⟨q⟩ vs sweep number
# 2. Calculate autocorrelation time
# 3. Compare with theoretical predictions
```

---

## Phase 2: Monte Carlo Methods

### Exercise 2.1: Metropolis Algorithm Deep Dive
**Duration**: 3 hours  
**Difficulty**: Intermediate

**Task**: Optimize the Metropolis algorithm for Gaussian sampling
- Study acceptance rate vs step size
- Measure autocorrelation times
- Implement adaptive step size

**Template**:
```python
from src.metropolis import MetropolisGaussian

def optimize_step_size(sampler, target_acceptance=0.5):
    """
    Find optimal step size for given target acceptance rate
    """
    # Student implementation
    pass

def measure_autocorrelation(samples):
    """
    Calculate and plot autocorrelation function
    """
    # Use utils.autocorrelation_function
    pass
```

### Exercise 2.2: Critical Slowing Down Investigation
**Duration**: 4 hours  
**Difficulty**: Intermediate-Advanced

**Task**: Study critical slowing down in the 2D Ising model
- Implement 2D Ising model with Metropolis
- Measure autocorrelation time vs temperature
- Identify critical temperature and dynamic exponent

**Starter Code**:
```python
class Ising2D:
    def __init__(self, L, T):
        self.L = L  # Lattice size
        self.T = T  # Temperature
        self.beta = 1.0 / T
        self.spins = np.random.choice([-1, 1], size=(L, L))
    
    def local_energy(self, i, j):
        """Calculate local energy at site (i,j)"""
        # Student implementation
        pass
    
    def metropolis_step(self):
        """Single Metropolis update"""
        # Student implementation
        pass
```

---

## Phase 3: Advanced Techniques

### Exercise 3.1: Hybrid Monte Carlo Implementation
**Duration**: 4 hours  
**Difficulty**: Advanced

**Task**: Implement and optimize HMC for scalar field theory
- Use provided `HMCFieldTheory1D` as reference
- Study molecular dynamics step size and trajectory length
- Compare performance with Metropolis

**Key Focus Areas**:
1. **Leapfrog Integration**: Understand symplectic properties
2. **Energy Conservation**: Monitor Hamiltonian violations
3. **Acceptance Rates**: Optimize parameters for efficiency

```python
def hmc_parameter_scan(field_theory, step_sizes, traj_lengths):
    """
    Scan HMC parameters and measure performance
    """
    results = {}
    for dt in step_sizes:
        for n_steps in traj_lengths:
            # Run HMC with these parameters
            # Measure: acceptance rate, autocorrelation time, computational cost
            pass
    return results
```

### Exercise 3.2: Machine Learning Acceleration (Optional)
**Duration**: 6 hours  
**Difficulty**: Advanced

**Task**: Implement a simple normalizing flow for field sampling
- Use TensorFlow/PyTorch for neural network implementation
- Train on configurations from standard MCMC
- Test sampling efficiency

**Framework**:
```python
import tensorflow as tf

class SimpleFlow(tf.keras.Model):
    def __init__(self, lattice_size):
        super().__init__()
        self.lattice_size = lattice_size
        # Define flow layers
    
    def call(self, z):
        """Transform noise to field configuration"""
        # Student implementation
        pass
    
    def log_prob(self, x):
        """Calculate log probability of configuration"""
        # Student implementation
        pass
```

---

## Phase 4: Applications

### Exercise 4.1: QED in 1+1 Dimensions (Schwinger Model)
**Duration**: 5 hours  
**Difficulty**: Advanced

**Task**: Implement the Schwinger model (QED in 2D)
- Gauge field on links, fermions on sites
- Study confinement and string tension
- Measure hadron spectrum

**Theoretical Background**:
- Gauge invariant action
- Wilson fermions
- String tension measurement

### Exercise 4.2: Critical Behavior Study
**Duration**: 4 hours  
**Difficulty**: Advanced

**Task**: Study phase transitions in scalar field theory
- Vary mass parameter to find critical point
- Measure critical exponents
- Finite size scaling analysis

---

## Assessment and Evaluation

### Continuous Assessment (60%)

**Weekly Reports**: Each exercise includes:
1. **Theory Summary** (20%): Brief explanation of physics concepts
2. **Implementation** (40%): Working code with comments
3. **Analysis** (40%): Plots, measurements, and interpretation

### Final Project (40%)

**Choose one**:
1. **Original Research**: Implement a new algorithm or study a new system
2. **Advanced Analysis**: Deep dive into critical phenomena or finite size effects
3. **Machine Learning**: Develop and test ML-accelerated sampling methods

### Grading Rubric

| Component | Excellent (A) | Good (B) | Satisfactory (C) | Needs Work (D/F) |
|-----------|---------------|----------|------------------|-------------------|
| **Physics Understanding** | Deep insight, connects theory to computation | Good grasp of concepts | Basic understanding | Missing key concepts |
| **Code Quality** | Clean, well-documented, efficient | Mostly clean and documented | Works but messy | Buggy or incomplete |
| **Analysis** | Thorough statistical analysis, clear plots | Good analysis with minor gaps | Basic analysis completed | Insufficient analysis |
| **Presentation** | Clear, professional writing | Good communication | Adequate presentation | Poor communication |

---

## Resources and Support

### Required Software
- Python 3.7+
- NumPy, SciPy, Matplotlib
- Jupyter notebooks
- Git for version control

### Recommended Reading
1. **Week 1-2**: Creutz chapters 1-3
2. **Week 3-4**: Creutz chapters 4-6
3. **Week 5-6**: Gattringer & Lang chapters 1-3
4. **Week 7-8**: Selected research papers

### Office Hours
- **Instructor**: Tuesdays 2-4 PM
- **TA Session**: Fridays 1-3 PM
- **Online Forum**: 24/7 discussion board

### Technical Support
- **Computational Issues**: Contact IT support
- **Physics Questions**: Use course forum or office hours
- **Code Reviews**: Available upon request

---

## Advanced Extensions

For students who complete the basic exercises early:

### Extension 1: Parallel Computing
- Implement MPI parallelization
- Study scaling behavior
- Compare different parallelization strategies

### Extension 2: GPU Acceleration
- Port critical loops to CUDA/OpenCL
- Benchmark performance improvements
- Analyze memory access patterns

### Extension 3: Advanced Machine Learning
- Implement more sophisticated flows
- Study different network architectures
- Compare with traditional methods

### Extension 4: Real QCD
- Study quenched QCD simulations
- Implement different fermion actions
- Measure hadronic observables

---

## Learning Outcomes Assessment

By the end of this course, students should demonstrate:

### Knowledge and Understanding
- [ ] Path integral formulation of quantum mechanics
- [ ] Monte Carlo methods and statistical analysis
- [ ] Lattice field theory principles
- [ ] Critical phenomena and phase transitions

### Practical Skills
- [ ] Implement MCMC algorithms from scratch
- [ ] Optimize computational parameters
- [ ] Analyze statistical data with error estimation
- [ ] Visualize and present scientific results

### Problem-Solving Abilities
- [ ] Debug computational physics codes
- [ ] Identify and solve convergence issues
- [ ] Design efficient sampling strategies
- [ ] Adapt methods to new problems

### Communication Skills
- [ ] Write clear technical reports
- [ ] Present results with appropriate visualizations
- [ ] Collaborate effectively on coding projects
- [ ] Explain complex concepts to peers
