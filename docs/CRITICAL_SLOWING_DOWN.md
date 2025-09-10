# Critical Slowing Down and Acceleration Methods

## Introduction to Critical Slowing Down

Critical slowing down is one of the most significant computational challenges in lattice field theory simulations, especially near phase transitions. This document explores the phenomenon and various mitigation strategies.

## 1. Understanding Critical Slowing Down

### 1.1 Definition and Manifestation

**Critical slowing down** occurs when the autocorrelation time τ of Monte Carlo simulations diverges as we approach a critical point:

```
τ ∼ ξᶻ
```

where:
- ξ is the correlation length
- z is the **dynamic critical exponent** (algorithm-dependent)
- Near criticality: ξ ∼ |T - T_c|^(-ν)

### 1.2 Physical Origin

At critical points:
- **Long-range correlations** develop
- **Collective excitations** become important
- **Local updates** become inefficient
- **Acceptance rates** may decrease

### 1.3 Algorithm Dependence

Different algorithms exhibit different dynamic exponents:

| Algorithm | System | Dynamic Exponent z |
|-----------|--------|-------------------|
| Metropolis | 2D Ising | ~2.1 |
| Heat Bath | 2D Ising | ~2.1 |
| Cluster (Swendsen-Wang) | 2D Ising | ~0.25 |
| HMC | φ⁴ theory | ~1-2 |
| Overrelaxation | Pure gauge | ~1 |

## 2. Measuring Critical Slowing Down

### 2.1 Autocorrelation Function

The autocorrelation function measures how long observables remain correlated:

```python
def autocorrelation_function(data, max_lag=None):
    """
    Compute autocorrelation function C(t) = ⟨O(0)O(t)⟩ / ⟨O²⟩
    """
    if max_lag is None:
        max_lag = len(data) // 4
    
    data_centered = data - np.mean(data)
    autocorr = []
    
    for lag in range(max_lag):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = np.corrcoef(data_centered[:-lag], data_centered[lag:])[0, 1]
            autocorr.append(corr)
    
    return np.array(autocorr)
```

### 2.2 Integrated Autocorrelation Time

The integrated autocorrelation time quantifies the correlation length in Monte Carlo time:

```python
def integrated_autocorrelation_time(data, W=None):
    """
    Compute τ_int = 1 + 2∑_{t=1}^W C(t)
    """
    autocorr = autocorrelation_function(data)
    
    if W is None:
        # Automatic windowing
        W = np.where(autocorr <= 0)[0][0] if np.any(autocorr <= 0) else len(autocorr)
    
    tau_int = 1 + 2 * np.sum(autocorr[1:W])
    return tau_int
```

### 2.3 Effective Sample Size

The effective number of independent samples:

```python
def effective_sample_size(data):
    """
    N_eff = N / (2τ_int)
    """
    tau_int = integrated_autocorrelation_time(data)
    return len(data) / (2 * tau_int)
```

## 3. Traditional Acceleration Methods

### 3.1 Overrelaxation

Overrelaxation reduces critical slowing down by making larger, deterministic updates:

```python
class Overrelaxation:
    def __init__(self, field_theory):
        self.field_theory = field_theory
    
    def overrelax_step(self, field, site):
        """
        Overrelaxation update: reflect about optimal value
        """
        # Calculate optimal field value at this site
        neighbors_sum = (field[(site-1) % self.N] + field[(site+1) % self.N])
        optimal = neighbors_sum / (2 + self.field_theory.m_squared)
        
        # Reflect current value about optimal
        field[site] = 2 * optimal - field[site]
        
        return field
```

### 3.2 Multi-Grid Methods

Multi-grid algorithms work on multiple length scales simultaneously:

```python
class MultiGrid:
    def __init__(self, levels):
        self.levels = levels
    
    def coarsen(self, field):
        """Restrict field to coarser grid"""
        coarse_field = []
        for i in range(0, len(field), 2):
            # Average neighboring sites
            coarse_field.append((field[i] + field[i+1]) / 2)
        return np.array(coarse_field)
    
    def prolongate(self, coarse_field):
        """Interpolate to finer grid"""
        fine_field = []
        for value in coarse_field:
            fine_field.extend([value, value])  # Simple duplication
        return np.array(fine_field)
    
    def v_cycle(self, field):
        """V-cycle multigrid update"""
        # Implement V-cycle algorithm
        pass
```

### 3.3 Fourier Acceleration

Transform to momentum space where different modes decorrelate:

```python
class FourierAcceleration:
    def __init__(self, lattice_size):
        self.N = lattice_size
        self.k_modes = 2 * np.pi * np.fft.fftfreq(lattice_size)
    
    def accelerated_update(self, field):
        """
        Update in Fourier space with mode-dependent step sizes
        """
        # Transform to momentum space
        field_k = np.fft.fft(field)
        
        # Mode-dependent updates
        for i, k in enumerate(self.k_modes):
            step_size = self.optimal_step_size(k)
            # Apply update with appropriate step size
            field_k[i] += step_size * self.force_k[i]
        
        # Transform back
        return np.real(np.fft.ifft(field_k))
    
    def optimal_step_size(self, k):
        """Optimal step size for momentum mode k"""
        return 1.0 / (4 * np.sin(k/2)**2 + self.mass_squared)
```

## 4. Machine Learning Approaches

### 4.1 Normalizing Flows

Normalizing flows learn the probability distribution and can generate decorrelated samples:

```python
import tensorflow as tf

class RealNVP(tf.keras.Model):
    """Real-valued Non-Volume Preserving flow"""
    
    def __init__(self, lattice_size, n_layers=8):
        super().__init__()
        self.lattice_size = lattice_size
        self.n_layers = n_layers
        
        # Coupling layers
        self.coupling_layers = []
        for i in range(n_layers):
            self.coupling_layers.append(
                CouplingLayer(lattice_size // 2, hidden_units=256)
            )
    
    def call(self, z):
        """Transform noise z to field configuration x"""
        x = z
        log_det_jacobian = 0
        
        for layer in self.coupling_layers:
            x, ldj = layer(x)
            log_det_jacobian += ldj
        
        return x, log_det_jacobian
    
    def inverse(self, x):
        """Transform field configuration x to noise z"""
        z = x
        log_det_jacobian = 0
        
        for layer in reversed(self.coupling_layers):
            z, ldj = layer.inverse(z)
            log_det_jacobian += ldj
        
        return z, log_det_jacobian

class CouplingLayer(tf.keras.layers.Layer):
    """Coupling layer for Real NVP"""
    
    def __init__(self, split_dim, hidden_units=256):
        super().__init__()
        self.split_dim = split_dim
        
        # Neural networks for scale and translation
        self.scale_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(split_dim, activation='tanh')
        ])
        
        self.translate_net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(split_dim)
        ])
    
    def call(self, x):
        """Forward transformation"""
        x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        y1 = x1
        y2 = x2 * tf.exp(s) + t
        
        log_det_jacobian = tf.reduce_sum(s, axis=1)
        
        return tf.concat([y1, y2], axis=1), log_det_jacobian
```

### 4.2 Training the Flow

```python
class FlowTrainer:
    def __init__(self, flow, target_action):
        self.flow = flow
        self.target_action = target_action
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def train_step(self, batch_size=128):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Sample from prior (noise)
            z = tf.random.normal((batch_size, self.flow.lattice_size))
            
            # Transform to field configurations
            x, log_det_jac = self.flow(z)
            
            # Compute target log probability (negative action)
            log_prob_target = -self.target_action(x)
            
            # Compute flow log probability
            log_prob_flow = -0.5 * tf.reduce_sum(z**2, axis=1) + log_det_jac
            
            # KL divergence loss
            loss = tf.reduce_mean(log_prob_flow - log_prob_target)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        
        return loss

    def train(self, n_epochs=1000):
        """Train the normalizing flow"""
        losses = []
        for epoch in range(n_epochs):
            loss = self.train_step()
            losses.append(loss.numpy())
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
```

### 4.3 Generative Adversarial Networks (GANs)

Alternative approach using adversarial training:

```python
class FieldGAN:
    def __init__(self, lattice_size):
        self.lattice_size = lattice_size
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
    def build_generator(self):
        """Build generator network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.lattice_size, activation='tanh')
        ])
        return model
    
    def build_discriminator(self):
        """Build discriminator network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(self.lattice_size,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
```

## 5. Comparison of Methods

### 5.1 Performance Metrics

When comparing acceleration methods, consider:

1. **Autocorrelation time reduction**
2. **Computational overhead**
3. **Implementation complexity**
4. **Systematic errors introduced**
5. **Scalability to larger systems**

### 5.2 Benchmark Results

Example comparison for 2D φ⁴ theory:

| Method | τ_int | Overhead | z_eff | Notes |
|--------|-------|----------|-------|-------|
| Metropolis | 100 | 1.0× | 2.0 | Baseline |
| HMC | 20 | 3.0× | 1.5 | Good for continuous fields |
| Overrelaxation | 50 | 1.2× | 1.8 | Simple to implement |
| Multi-grid | 15 | 5.0× | 1.2 | Complex implementation |
| ML Flow | 5 | 10.0× | 0.8 | Requires training |

### 5.3 Implementation Strategy

```python
class AccelerationComparison:
    def __init__(self, field_theory):
        self.field_theory = field_theory
        self.methods = {
            'metropolis': MetropolisUpdater(),
            'hmc': HMCUpdater(),
            'overrelax': OverrelaxationUpdater(),
            'flow': FlowSampler()
        }
    
    def benchmark_method(self, method_name, n_sweeps=10000):
        """Benchmark a specific method"""
        method = self.methods[method_name]
        
        start_time = time.time()
        configurations = method.generate_samples(self.field_theory, n_sweeps)
        end_time = time.time()
        
        # Measure autocorrelation time
        observable = [self.compute_observable(config) for config in configurations]
        tau_int = integrated_autocorrelation_time(observable)
        
        # Compute effective samples per second
        eff_samples_per_sec = effective_sample_size(observable) / (end_time - start_time)
        
        return {
            'tau_int': tau_int,
            'wall_time': end_time - start_time,
            'eff_samples_per_sec': eff_samples_per_sec
        }
    
    def full_comparison(self):
        """Compare all methods"""
        results = {}
        for method_name in self.methods:
            print(f"Benchmarking {method_name}...")
            results[method_name] = self.benchmark_method(method_name)
        
        return results
```

## 6. Future Directions

### 6.1 Advanced ML Techniques

- **Flow-based models** with better expressivity
- **Variational autoencoders** for configuration generation
- **Reinforcement learning** for adaptive sampling
- **Graph neural networks** for gauge theories

### 6.2 Hardware Acceleration

- **GPU implementations** of traditional algorithms
- **Tensor processing units** for ML approaches
- **Quantum computing** for specific problems

### 6.3 Hybrid Approaches

Combining multiple acceleration techniques:
- **ML-guided HMC** with learned proposal distributions
- **Multi-scale flows** incorporating physical priors
- **Adaptive algorithms** that switch methods dynamically

## 7. Practical Implementation Guidelines

### 7.1 When to Use Each Method

**Metropolis/Heat Bath**:
- Simple systems
- Proof-of-concept studies
- Educational purposes

**HMC**:
- Continuous field theories
- When derivatives are available
- Large system sizes

**ML Methods**:
- Critical regions
- Complex target distributions
- When training data is available

### 7.2 Implementation Checklist

- [ ] Validate new methods against known results
- [ ] Monitor for systematic biases
- [ ] Test on multiple system sizes
- [ ] Compare computational efficiency
- [ ] Document parameter choices and tuning

### 7.3 Common Pitfalls

1. **Insufficient thermalization** after algorithm changes
2. **Biased training data** for ML approaches
3. **Overfitting** in neural network methods
4. **Incorrect error estimation** with correlated samples
5. **Hardware-specific optimizations** that don't generalize
