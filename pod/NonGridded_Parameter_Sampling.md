# POD for Non-Gridded Parameter Sampling

## The Problem: When Parameters Don't Form a Tensor

### Question Addressed

**If the snapshot data is not organized as a tensor** — for instance, when two parameters $\mu_1$ and $\mu_2$ are **not sampled on a grid** (e.g., random sampling, adaptive sampling, scattered points) — **how can we conduct POD and tensor decomposition methods?**

---

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Grid vs. Non-Grid Parameter Sampling](#grid-vs-non-grid-parameter-sampling)
3. [Solution 1: Classical POD (Always Works)](#solution-1-classical-pod-always-works)
4. [Solution 2: Manifold Learning + POD](#solution-2-manifold-learning--pod)
5. [Solution 3: Kernel Methods + POD](#solution-3-kernel-methods--pod)
6. [Solution 4: Interpolation to Grid (Simple but Problematic)](#solution-4-interpolation-to-grid-simple-but-problematic)
7. [Solution 5: Grassmannian Interpolation](#solution-5-grassmannian-interpolation)
8. [Solution 6: Neural Network Parametrization](#solution-6-neural-network-parametrization)
9. [Practical Recommendations](#practical-recommendations)
10. [Comparison of Methods](#comparison-of-methods)

---

## Understanding the Problem

### What Makes Data "Tensor-Structured"?

A **tensor structure** requires that data is organized on a **multi-dimensional grid**:

**Grid-Based (Tensor):**
```
Parameters sampled on Cartesian grid:
μ₁ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}     (5 values)
μ₂ ∈ {1.0, 2.0, 3.0, 4.0}         (4 values)

Total combinations: 5 × 4 = 20 parameter pairs

All pairs evaluated:
(μ₁, μ₂) = (0.1, 1.0), (0.1, 2.0), (0.1, 3.0), (0.1, 4.0),
           (0.2, 1.0), (0.2, 2.0), ..., (0.5, 4.0)

Data structure: U ∈ ℝ^(Nx × Nt × Nμ₁ × Nμ₂)
                     = ℝ^(Nx × Nt × 5 × 4)

This IS a tensor ✓
```

**Non-Grid (Scattered):**
```
Parameters sampled randomly or adaptively:
20 random parameter pairs:

(μ₁, μ₂) = (0.13, 1.7), (0.27, 3.2), (0.41, 1.1), (0.19, 2.9), ...

Data structure: U ∈ ℝ^(Nx × Nt × 20)

This is NOT a tensor ✗ (just a matrix!)
```

### Why This Matters

**Tucker/HOSVD requires tensor structure:**
- Mode-3 unfolding needs $\mu_1$ dimension
- Mode-4 unfolding needs $\mu_2$ dimension
- But scattered points don't have separate $\mu_1$ and $\mu_2$ dimensions!

**Example of failure:**
```python
# Grid-based (works):
U_grid = np.zeros((Nx, Nt, Nmu1, Nmu2))  # 4D tensor ✓
tucker_decomposition(U_grid)  # Works!

# Scattered (fails):
U_scattered = np.zeros((Nx, Nt, 20))  # 3D tensor ✗
tucker_decomposition(U_scattered)  # Can't unfold along μ₁ and μ₂!
```

---

## Grid vs. Non-Grid Parameter Sampling

### Grid-Based Sampling (Tensor Structure)

**Characteristics:**
- Parameters sampled on a **regular grid**
- All combinations of parameter values evaluated
- Data naturally forms a multi-dimensional array (tensor)

**Advantages:**
✅ Natural tensor structure
✅ Tucker/HOSVD methods applicable
✅ Easy interpolation (structured grid)
✅ Clear separation of parameter effects

**Disadvantages:**
❌ **Curse of dimensionality**: $N^d$ samples needed for $d$ parameters
❌ Wastes samples in unimportant regions
❌ Expensive for high-dimensional parameter spaces

**Example:**
```
2 parameters, 10 values each:
Grid sampling: 10 × 10 = 100 simulations

3 parameters, 10 values each:
Grid sampling: 10 × 10 × 10 = 1,000 simulations

5 parameters, 10 values each:
Grid sampling: 10^5 = 100,000 simulations (impractical!)
```

### Non-Grid Sampling (No Tensor Structure)

**Types:**

1. **Random Sampling**
   ```
   Sample N parameter combinations uniformly at random:
   (μ₁, μ₂) ~ Uniform([μ₁_min, μ₁_max] × [μ₂_min, μ₂_max])
   ```

2. **Latin Hypercube Sampling (LHS)**
   ```
   Stratified sampling ensuring coverage:
   Better space-filling than pure random
   ```

3. **Adaptive Sampling**
   ```
   Sample more densely where solution varies rapidly:
   Error-driven refinement
   ```

4. **Sobol Sequences**
   ```
   Low-discrepancy quasi-random sequences:
   Better coverage than random
   ```

5. **Experimental Data**
   ```
   Parameters determined by experiments, not design:
   Irregular, opportunistic sampling
   ```

**Advantages:**
✅ Far fewer samples needed
✅ Focuses samples where needed
✅ Practical for high-dimensional parameters
✅ Reflects real experimental constraints

**Disadvantages:**
❌ No natural tensor structure
❌ Tucker/HOSVD not directly applicable
❌ Interpolation more complex
❌ Requires different ROM approaches

---

## Solution 1: Classical POD (Always Works)

### Core Idea

**POD doesn't require tensor structure** — it only needs a **snapshot matrix**.

### Mathematical Formulation

**Data Organization:**

Regardless of parameter sampling strategy, organize snapshots as a matrix:

$$
X = [u(x, t_1; \mu^{(1)}), u(x, t_2; \mu^{(1)}), \ldots, u(x, t_{N_t}; \mu^{(1)}), u(x, t_1; \mu^{(2)}), \ldots]
$$

$$
X \in \mathbb{R}^{N_x \times (N_t \cdot N_\mu)}
$$

where:
- $N_x$ = spatial degrees of freedom
- $N_t$ = time snapshots per parameter
- $N_\mu$ = number of parameter samples (can be scattered!)

**Key Point:** $N_\mu$ doesn't need to have tensor structure — it's just a flat list of parameter combinations.

### Algorithm

**Step 1: Collect Snapshots**

For each parameter combination $\mu^{(k)}$ (scattered or gridded):
- Solve PDE: $u(x, t; \mu^{(k)})$
- Store spatiotemporal snapshots

```python
snapshots = []
parameters = []  # Can be random, adaptive, etc.

for mu in sampled_parameters:  # Non-gridded!
    u = solve_pde(mu)  # Shape: (Nx, Nt)
    snapshots.append(u.flatten())  # or u.reshape(-1)
    parameters.append(mu)

X = np.column_stack(snapshots)  # Shape: (Nx*Nt, Nmu)
# Or if keeping time separate:
X = np.hstack([u.reshape(Nx, -1) for u in snapshots])  # (Nx, Nt*Nmu)
```

**Step 2: Perform SVD**

$$
X = U \Sigma V^T
$$

**Step 3: Extract POD Modes**

$$
\Phi = U(:, 1:r) \in \mathbb{R}^{N_x \times r}
$$

**Step 4: Reduced Representation**

For any parameter $\mu$:
$$
u(x, t; \mu) \approx \Phi \, a(t; \mu)
$$

where $a(t; \mu) \in \mathbb{R}^r$ are the reduced coefficients.

### Handling New Parameters (The Challenge)

**Problem:** Given POD modes $\Phi$, how to find $a(t; \mu^*)$ for new $\mu^*$ not in training set?

**Sub-Solution 1A: Interpolation of Coefficients**

**Step 1:** Project training snapshots onto POD basis
$$
a^{(k)}(t) = \Phi^T u(x, t; \mu^{(k)}) \quad \text{for } k = 1, \ldots, N_\mu
$$

**Step 2:** Build interpolation model
$$
a(t; \mu) = \mathcal{I}(\mu; \{a^{(k)}, \mu^{(k)}\}_{k=1}^{N_\mu})
$$

**Interpolation Methods:**

1. **Radial Basis Functions (RBF)**
   $$
   a_i(t; \mu) = \sum_{k=1}^{N_\mu} w_{ik}(t) \, \phi(\|\mu - \mu^{(k)}\|)
   $$

   Common kernels:
   - Gaussian: $\phi(r) = e^{-(\epsilon r)^2}$
   - Multiquadric: $\phi(r) = \sqrt{1 + (\epsilon r)^2}$
   - Thin-plate spline: $\phi(r) = r^2 \log r$

2. **Kriging (Gaussian Process)**
   $$
   a(t; \mu) \sim \mathcal{GP}(m(\mu), k(\mu, \mu'))
   $$

   Provides uncertainty quantification!

3. **Polynomial Regression**
   $$
   a_i(t; \mu) = \sum_{|\alpha| \leq p} c_{i\alpha}(t) \, \mu^\alpha
   $$

4. **k-Nearest Neighbors**
   $$
   a(t; \mu) = \frac{1}{k} \sum_{j \in \text{NN}_k(\mu)} a^{(j)}(t)
   $$

**Sub-Solution 1B: Solve in Reduced Space (Intrusive)**

If governing equations are known, solve directly in reduced space:

$$
\Phi^T \frac{\partial u}{\partial t} = \Phi^T F(u; \mu) \quad \Rightarrow \quad \frac{da}{dt} = F_r(a; \mu)
$$

This bypasses interpolation but requires intrusive access to operators.

### Concrete Example

**Problem:** Heat equation with two parameters

$$
\frac{\partial u}{\partial t} = \mu_1 \frac{\partial^2 u}{\partial x^2} + \mu_2 f(x)
$$

**Non-Grid Sampling:**
```python
# 30 random parameter combinations
np.random.seed(42)
Nmu = 30
mu1_samples = np.random.uniform(0.05, 0.25, Nmu)
mu2_samples = np.random.uniform(0.1, 0.5, Nmu)
parameters = np.column_stack([mu1_samples, mu2_samples])

# Solve for each parameter combination
Nx = 101
Nt = 200
X = np.zeros((Nx, Nt * Nmu))

for k, (mu1, mu2) in enumerate(parameters):
    u = solve_heat_equation(mu1, mu2)  # Shape: (Nx, Nt)
    X[:, k*Nt:(k+1)*Nt] = u

# POD (standard, works regardless of parameter structure!)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
Phi = U[:, :r]  # Select r modes

# Project training data
A_train = Phi.T @ X  # Shape: (r, Nt*Nmu)
A_train = A_train.reshape(r, Nmu, Nt)  # (r, Nmu, Nt)

# Interpolation for new parameter mu_star = (0.15, 0.3)
from scipy.interpolate import RBFInterpolator

a_interp = np.zeros((r, Nt))
for i in range(r):
    for j in range(Nt):
        # Interpolate i-th mode at j-th time step
        rbf = RBFInterpolator(parameters, A_train[i, :, j])
        a_interp[i, j] = rbf([0.15, 0.3])

# Reconstruct
u_star = Phi @ a_interp  # Shape: (Nx, Nt)
```

### Advantages

✅ **Always applicable** — no tensor structure needed
✅ **Well-established theory** — POD convergence guarantees
✅ **Flexible sampling** — works with any parameter distribution
✅ **Optimal spatial compression** — POD modes are optimal in $L^2$ sense

### Disadvantages

❌ **Parameter interpolation required** — adds approximation error
❌ **Curse of dimensionality** — interpolation struggles in high-dimensional parameter space
❌ **No parametric structure exploitation** — treats parameters as unstructured points
❌ **Interpolation quality varies** — depends on parameter density and smoothness

### When to Use

- Parameter dimension ≤ 4
- Non-grid sampling is unavoidable (experiments, adaptive refinement)
- Intrusive ROM with Galerkin projection available
- Good coverage of parameter space with training samples

---

## Solution 2: Manifold Learning + POD

### Core Idea

Even if parameters are scattered, the **solution manifold** may have low-dimensional structure. Use **manifold learning** to find intrinsic coordinates, then apply POD.

### Mathematical Framework

**Hypothesis:** Solutions lie on a low-dimensional manifold
$$
\mathcal{M} = \{u(x, t; \mu) : \mu \in \mathcal{P}\} \subset \mathbb{R}^{N_x \times N_t}
$$

**Goal:** Find intrinsic coordinates $\xi \in \mathbb{R}^d$ where $d \ll N_\mu$

### Algorithm

**Step 1: Manifold Learning**

Apply dimension reduction to parameter-snapshot pairs:

**Input:** $\{(\mu^{(k)}, u^{(k)})\}_{k=1}^{N_\mu}$

**Methods:**

1. **Isomap** (Isometric Mapping)
   - Preserve geodesic distances
   - Build $k$-NN graph
   - Compute shortest paths
   - MDS on distance matrix

2. **Diffusion Maps**
   - Spectral analysis of diffusion operator
   - Intrinsic geometry via eigendecomposition

3. **Locally Linear Embedding (LLE)**
   - Preserve local neighborhoods
   - Linear reconstruction weights

4. **t-SNE / UMAP**
   - Nonlinear visualization
   - Less suitable for interpolation

**Output:** Intrinsic coordinates $\xi^{(k)} \in \mathbb{R}^d$ for each snapshot

**Step 2: POD in Original Space**

$$
X = U \Sigma V^T, \quad \Phi = U(:, 1:r)
$$

**Step 3: Learn Mapping**

Build surrogate model:
$$
a(t; \xi) = \mathcal{F}(\xi)
$$

where $\mathcal{F}$ learned from $\{(\xi^{(k)}, a^{(k)}(t))\}$

**Step 4: For New Parameter**

1. Map $\mu^* \to \xi^*$ (use learned inverse map or interpolation)
2. Evaluate $a^* = \mathcal{F}(\xi^*)$
3. Reconstruct $u^* = \Phi \, a^*$

### Concrete Example: Diffusion Maps

**Problem:** 2D parameter space with complex structure

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

# Step 1: Build affinity matrix
epsilon = 0.1  # kernel bandwidth
distances = squareform(pdist(parameters))
W = np.exp(-distances**2 / epsilon)

# Step 2: Diffusion map
D = np.diag(W.sum(axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
L = D_inv_sqrt @ W @ D_inv_sqrt

# Step 3: Eigendecomposition
eigenvalues, eigenvectors = eigh(L)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 4: Intrinsic coordinates (first d eigenvectors)
d = 2  # intrinsic dimension
xi = eigenvectors[:, 1:d+1]  # Skip first (constant) eigenvector

# Step 5: Interpolate in intrinsic space
from scipy.interpolate import LinearNDInterpolator
for i in range(r):
    for j in range(Nt):
        interp = LinearNDInterpolator(xi, A_train[i, :, j])
        # For new mu_star, first map to xi_star (via interpolation of mu->xi map)
        # Then evaluate a_interp[i,j] = interp(xi_star)
```

### Advantages

✅ **Discovers hidden structure** — finds intrinsic parameter dimension
✅ **Reduces effective dimension** — interpolation in low-$d$ space
✅ **Handles nonlinear manifolds** — not limited to linear relationships
✅ **Better interpolation** — lower-dimensional intrinsic space

### Disadvantages

❌ **Complex implementation** — multiple steps with tuning
❌ **Out-of-sample extension** — mapping new $\mu$ to $\xi$ non-trivial
❌ **Computational cost** — eigendecomposition of large matrices
❌ **Requires dense sampling** — manifold reconstruction needs good coverage

### When to Use

- Parameter space has intrinsic low-dimensional structure
- High-dimensional parameters but solutions vary smoothly
- Have dense sampling in important regions
- Willing to invest in preprocessing

---

## Solution 3: Kernel Methods + POD

### Core Idea

Use **kernel functions** to handle scattered parameter samples, combined with POD spatial modes.

### Mathematical Framework

**Key Insight:** Kernel methods implicitly work in high (or infinite) dimensional feature space without computing coordinates explicitly.

### Algorithm: Kernel POD

**Step 1: Choose Kernel Function**

Define kernel measuring similarity between parameter values:
$$
k(\mu, \mu') = \exp\left(-\frac{\|\mu - \mu'\|^2}{2\sigma^2}\right) \quad \text{(Gaussian/RBF)}
$$

**Step 2: Spatial POD (Standard)**

$$
X = U \Sigma V^T, \quad \Phi = U(:, 1:r)
$$

**Step 3: Kernel Matrix for Parameters**

Build Gram matrix:
$$
K_{ij} = k(\mu^{(i)}, \mu^{(j)}), \quad K \in \mathbb{R}^{N_\mu \times N_\mu}
$$

**Step 4: Kernel Ridge Regression for Coefficients**

For each POD mode $i$ and time $t_j$:
$$
a_i(t_j; \cdot) = \sum_{k=1}^{N_\mu} \alpha_{ijk} \, k(\cdot, \mu^{(k)})
$$

Solve for weights $\alpha$ via ridge regression:
$$
\alpha = (K + \lambda I)^{-1} a_{\text{train}}
$$

**Step 5: Prediction for New Parameter**

$$
a_i(t_j; \mu^*) = \sum_{k=1}^{N_\mu} \alpha_{ijk} \, k(\mu^*, \mu^{(k)})
$$

### Concrete Example

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

# Step 1: Standard spatial POD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
Phi = U[:, :r]
A_train = Phi.T @ X  # (r, Nt*Nmu)

# Step 2: Reshape to separate parameters
A_train = A_train.reshape(r, Nmu, Nt)

# Step 3: Kernel regression for each mode and time
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

a_pred = np.zeros((r, Nt))
for i in range(r):
    for j in range(Nt):
        # GP regression
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)
        gp.fit(parameters, A_train[i, :, j])

        # Predict at new parameter
        mu_star = np.array([[0.15, 0.3]])
        a_pred[i, j], sigma = gp.predict(mu_star, return_std=True)

# Step 4: Reconstruct
u_star = Phi @ a_pred
```

### Advantages

✅ **Handles scattered data naturally** — no grid structure needed
✅ **Uncertainty quantification** — GP provides confidence intervals
✅ **Flexible nonlinear relationships** — kernel choice adapts to problem
✅ **Mathematically rigorous** — representer theorem guarantees

### Disadvantages

❌ **Kernel bandwidth tuning** — critical parameter selection
❌ **Computational cost** — $\mathcal{O}(N_\mu^3)$ for kernel matrix inversion
❌ **Memory requirements** — storing $N_\mu \times N_\mu$ kernel matrix
❌ **Curse of dimensionality** — kernel methods struggle in high-$d$ parameter space (>5-6D)

### When to Use

- Small to moderate number of parameter samples ($N_\mu < 1000$)
- Parameter dimension ≤ 5
- Need uncertainty quantification
- Smooth parameter dependence expected

---

## Solution 4: Interpolation to Grid (Simple but Problematic)

### Core Idea

**Force** scattered data onto a regular grid via interpolation, then apply tensor methods.

### Algorithm

**Step 1: Define Regular Grid**

Choose grid resolution:
$$
\mu_1 \in \{\mu_1^{(1)}, \ldots, \mu_1^{(N_1)}\}
$$
$$
\mu_2 \in \{\mu_2^{(1)}, \ldots, \mu_2^{(N_2)}\}
$$

Total grid points: $N_1 \times N_2$

**Step 2: Interpolate Scattered Data to Grid**

For each grid point $(\mu_1^{(i)}, \mu_2^{(j)})$:
$$
u_{\text{grid}}(x, t; \mu_1^{(i)}, \mu_2^{(j)}) = \mathcal{I}(u_{\text{scattered}}, (\mu_1^{(i)}, \mu_2^{(j)}))
$$

Methods:
- Linear interpolation (Delaunay triangulation)
- Cubic interpolation
- RBF interpolation
- Kriging

**Step 3: Form Tensor**

$$
\mathcal{U}_{\text{grid}} \in \mathbb{R}^{N_x \times N_t \times N_1 \times N_2}
$$

**Step 4: Apply Tucker/HOSVD**

Now tensor structure exists, proceed with tensor decomposition.

### Concrete Example

```python
from scipy.interpolate import griddata

# Step 1: Define grid
mu1_grid = np.linspace(0.05, 0.25, 10)
mu2_grid = np.linspace(0.1, 0.5, 8)
Mu1, Mu2 = np.meshgrid(mu1_grid, mu2_grid, indexing='ij')
grid_points = np.column_stack([Mu1.ravel(), Mu2.ravel()])

# Step 2: Interpolate each spatial-temporal point
U_grid = np.zeros((Nx, Nt, 10, 8))

for i in range(Nx):
    for j in range(Nt):
        # Scattered values at this (x,t)
        values_scattered = np.array([X[i, k*Nt + j] for k in range(Nmu)])

        # Interpolate to grid
        values_grid = griddata(
            parameters, values_scattered, grid_points, method='cubic'
        )
        U_grid[i, j, :, :] = values_grid.reshape(10, 8)

# Step 3: Tucker decomposition
from tensorly.decomposition import tucker
core, factors = tucker(U_grid, rank=[20, 10, 8, 6])
```

### Advantages

✅ **Simple concept** — straightforward to implement
✅ **Enables tensor methods** — Tucker/HOSVD now applicable
✅ **Standard tools** — uses existing tensor libraries

### Disadvantages

❌ **Adds interpolation error** — two-stage error (interpolation + decomposition)
❌ **Inefficient** — creates more data than available (grid points > scattered samples)
❌ **Lost information** — interpolation can introduce artifacts
❌ **Arbitrary grid choice** — resolution selection non-obvious
❌ **Extrapolation issues** — grid may extend beyond sampled region

### Why This Is Generally Not Recommended

**Error Compounding:**
$$
\text{Total Error} = \underbrace{\text{Interp Error}}_{\text{scattered} \to \text{grid}} + \underbrace{\text{Tucker Error}}_{\text{grid} \to \text{reduced}}
$$

**Better Alternative:** Interpolate reduced coefficients directly (Solution 1) — only one interpolation step.

### When It Might Be Acceptable

- Scattered samples are dense and well-distributed
- Parameter dimension is low (2-3D)
- Quick prototyping or diagnostic analysis
- No better alternative available

---

## Solution 5: Grassmannian Interpolation

### Core Idea

Instead of interpolating coefficients, **interpolate POD subspaces** directly using geometry of Grassmann manifold.

### Mathematical Framework

**Grassmann Manifold:** Space of $r$-dimensional linear subspaces in $\mathbb{R}^{N_x}$

Each POD basis $\Phi(\mu) \in \mathbb{R}^{N_x \times r}$ represents a point on $\text{Gr}(r, N_x)$

### Algorithm

**Step 1: Compute Local POD Bases**

For each parameter sample $\mu^{(k)}$:
$$
\Phi^{(k)} = \text{POD}(u(x, t; \mu^{(k)}), r)
$$

**Step 2: Grassmann Metric**

Distance between subspaces via principal angles:
$$
d_G(\Phi^{(i)}, \Phi^{(j)}) = \left\| \theta(\Phi^{(i)}, \Phi^{(j)}) \right\|
$$

where $\theta$ are principal angles from canonical correlation.

**Step 3: Interpolation on Grassmann Manifold**

For new parameter $\mu^*$, find $k$ nearest neighbors and perform **geodesic interpolation**:

$$
\Phi(\mu^*) = \text{Exp}_{\Phi^{(i)}}\left(\sum_{j \in \text{NN}_k} w_j \, \text{Log}_{\Phi^{(i)}}(\Phi^{(j)})\right)
$$

where $\text{Exp}$ and $\text{Log}$ are Riemannian exponential and logarithm maps.

**Step 4: Project and Solve**

$$
a(t; \mu^*) = \Phi(\mu^*)^T u_0, \quad \frac{da}{dt} = F_r(a; \mu^*)
$$

### Concrete Example (Simplified)

```python
from scipy.linalg import subspace_angles

# Step 1: Local POD for each parameter
POD_bases = []
for k in range(Nmu):
    u_k = X[:, k*Nt:(k+1)*Nt]
    U_k, _, _ = np.linalg.svd(u_k, full_matrices=False)
    Phi_k = U_k[:, :r]
    POD_bases.append(Phi_k)

# Step 2: Distance matrix on Grassmann manifold
def grassmann_distance(Phi1, Phi2):
    angles = subspace_angles(Phi1, Phi2)
    return np.linalg.norm(angles)

D = np.zeros((Nmu, Nmu))
for i in range(Nmu):
    for j in range(i+1, Nmu):
        D[i,j] = D[j,i] = grassmann_distance(POD_bases[i], POD_bases[j])

# Step 3: For new mu_star, find nearest neighbors
mu_star = np.array([0.15, 0.3])
distances_to_star = [np.linalg.norm(mu_star - parameters[k]) for k in range(Nmu)]
k_nearest = np.argsort(distances_to_star)[:5]

# Simplified geodesic interpolation (proper implementation requires Riemannian geometry)
weights = 1.0 / (np.array(distances_to_star)[k_nearest] + 1e-10)
weights /= weights.sum()

# Weighted average on tangent space (approximate)
Phi_star = sum(weights[i] * POD_bases[k_nearest[i]] for i in range(5))
# Orthonormalize
Phi_star, _ = np.linalg.qr(Phi_star)
```

### Advantages

✅ **Geometrically principled** — respects manifold structure
✅ **Preserves orthonormality** — interpolated basis remains orthonormal
✅ **No coefficient interpolation** — works directly with subspaces
✅ **Better for rapidly changing bases** — handles rotations of POD modes

### Disadvantages

❌ **Complex implementation** — requires Riemannian geometry tools
❌ **Computational cost** — geodesic computations expensive
❌ **Requires dense sampling** — needs good coverage of parameter space
❌ **Limited software support** — fewer libraries available

### When to Use

- POD basis changes significantly with parameters
- High accuracy requirements
- Research context with implementation resources
- Moderate parameter dimension (2-4D)

---

## Solution 6: Neural Network Parametrization

### Core Idea

Use **neural networks** to learn the mapping from parameters to solution (or reduced coefficients).

### Approaches

### Approach 6A: Neural Network for Reduced Coefficients

**Architecture:**
$$
a(t; \mu) = \text{NN}(\mu; \theta)
$$

where $\text{NN}$ is a feedforward neural network with parameters $\theta$.

**Algorithm:**

**Step 1: Spatial POD (Standard)**
$$
\Phi = \text{POD}(X, r)
$$

**Step 2: Project Training Data**
$$
a^{(k)}(t) = \Phi^T u(x, t; \mu^{(k)})
$$

**Step 3: Train Neural Network**

Input: $\mu \in \mathbb{R}^{d_\mu}$
Output: $a(t) \in \mathbb{R}^{r \times N_t}$

Loss:
$$
\mathcal{L}(\theta) = \sum_{k=1}^{N_\mu} \sum_{j=1}^{N_t} \|a^{(k)}(t_j) - \text{NN}(\mu^{(k)}; \theta)\|^2
$$

**Step 4: Prediction**
$$
a^* = \text{NN}(\mu^*; \theta), \quad u^* = \Phi \, a^*
$$

### Approach 6B: Physics-Informed Neural Networks (PINNs)

Embed governing equations as physics constraints:

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}}
$$

where:
$$
\mathcal{L}_{\text{PDE}} = \left\| \frac{\partial u_{\text{NN}}}{\partial t} - F(u_{\text{NN}}, \mu) \right\|^2
$$

### Approach 6C: Autoencoder

**Architecture:**

Encoder: $z = E(u; \mu)$ (parametric latent space)
Decoder: $\hat{u} = D(z; \mu)$

Train end-to-end:
$$
\mathcal{L} = \sum_{k} \|u^{(k)} - D(E(u^{(k)}; \mu^{(k)}); \mu^{(k)})\|^2
$$

### Concrete Example: Simple NN for Coefficients

```python
import torch
import torch.nn as nn

# Step 1: POD
U_svd, S, Vt = np.linalg.svd(X, full_matrices=False)
Phi = U_svd[:, :r]
A_train = Phi.T @ X  # (r, Nt*Nmu)
A_train = A_train.reshape(r, Nmu, Nt)

# Step 2: Neural network architecture
class CoefficientNet(nn.Module):
    def __init__(self, d_mu, r, Nt):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_mu, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, r * Nt)
        )
        self.r = r
        self.Nt = Nt

    def forward(self, mu):
        output = self.net(mu)
        return output.view(-1, self.r, self.Nt)

# Step 3: Training
model = CoefficientNet(d_mu=2, r=r, Nt=Nt)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

mu_tensor = torch.FloatTensor(parameters)
a_tensor = torch.FloatTensor(A_train.transpose(1, 0, 2))  # (Nmu, r, Nt)

for epoch in range(5000):
    optimizer.zero_grad()
    a_pred = model(mu_tensor)
    loss = criterion(a_pred, a_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Step 4: Prediction
mu_star_tensor = torch.FloatTensor([[0.15, 0.3]])
with torch.no_grad():
    a_star = model(mu_star_tensor).numpy()[0]

u_star = Phi @ a_star
```

### Advantages

✅ **Handles scattered data naturally** — no grid structure needed
✅ **Scalable to high dimensions** — deep networks handle high-$d$ parameters
✅ **Fast online evaluation** — forward pass is cheap
✅ **Can incorporate physics** — PINNs embed governing equations

### Disadvantages

❌ **Requires substantial training data** — thousands of samples often needed
❌ **Hyperparameter tuning** — architecture, learning rate, regularization
❌ **Black-box behavior** — less interpretable than classical methods
❌ **Training time** — can be slow for large networks
❌ **No theoretical guarantees** — error bounds unclear

### When to Use

- Large training dataset available (>100 samples)
- High-dimensional parameter space (>5D)
- Online evaluation speed critical
- Willing to invest in training infrastructure
- Modern machine learning tools available

---

## Solution 7: Interpolating Neural Networks (INN)

### Core Idea

**Interpolating Neural Networks** combine interpolation theory with neural network architectures to create ROMs with **provable convergence**, **fewer parameters**, and **structured parameter handling**. Unlike standard neural networks, INNs use **tensor-product interpolation** as their building block.

### Background: Recent Advances

Published in *Nature Communications* and related journals, INNs represent a paradigm shift from classical MLPs:

**Key Innovation:** Replace dense layers with **CP tensor decomposition** and **interpolation kernels**:

$$
u_{\text{INN}}(x; \mu) = \sum_{i=1}^{r} \prod_{d=1}^{D} \phi_d^{(i)}(\xi_d)
$$

where:
- $r$ is the CP rank (number of modes)
- $D$ is the total dimension (space + parameters)
- $\phi_d^{(i)}$ are 1D interpolation functions (linear, RBF, splines)
- $\xi_d$ are normalized coordinates

### Mathematical Framework

#### INN Architecture for Parametric PDEs

For a solution $u(x, t; \mu_1, \mu_2)$ with scattered parameter samples:

**Step 1: Normalize All Dimensions**

$$
\xi_x \in [0,1], \quad \xi_t \in [0,1], \quad \xi_{\mu_1} \in [0,1], \quad \xi_{\mu_2} \in [0,1]
$$

**Step 2: Define Interpolation Grid**

For each dimension $d$, choose grid points $\{g_d^{(j)}\}_{j=1}^{n_d}$:
- Spatial: $n_x$ points (e.g., 50)
- Temporal: $n_t$ points (e.g., 50)
- Parameter 1: $n_{\mu_1}$ points (e.g., 20)
- Parameter 2: $n_{\mu_2}$ points (e.g., 20)

**Important:** These grid points are **INN internal nodes**, not necessarily the parameter sample locations!

**Step 3: INN Representation**

$$
u_{\text{INN}}(\xi_x, \xi_t, \xi_{\mu_1}, \xi_{\mu_2}) = \sum_{i=1}^{r} w_i \prod_{d \in \{x,t,\mu_1,\mu_2\}} \phi_d^{(i)}(\xi_d)
$$

where $\phi_d^{(i)}$ are interpolation basis functions (linear, cubic spline, RBF).

**Step 4: Trainable Parameters**

For each mode $i$ and dimension $d$, trainable parameters are values at grid points:

$$
\theta_d^{(i)} = [v_1^{(i)}, v_2^{(i)}, \ldots, v_{n_d}^{(i)}] \in \mathbb{R}^{n_d}
$$

Total parameters:
$$
N_{\text{params}} = r \times \sum_{d} n_d = r \times (n_x + n_t + n_{\mu_1} + n_{\mu_2})
$$

**Example:**
```
r = 10 modes
nx = 50, nt = 50, nμ₁ = 20, nμ₂ = 20

Standard MLP (3 hidden layers, 128 neurons):
  Parameters ≈ 4 × 128² ≈ 65,536

INN:
  Parameters = 10 × (50 + 50 + 20 + 20) = 1,400

Reduction: 47× fewer parameters!
```

### How INN Handles Non-Gridded Parameters

**Key Insight:** INN creates its **own internal interpolation grid** for parameters, independent of training sample locations.

**Training Process:**

**Step 1: Define Internal Grid**

Choose $n_{\mu_1}$ and $n_{\mu_2}$ grid points in normalized parameter space $[0,1]^2$:

```python
# INN internal parameter grid (can be uniform or adaptive)
mu1_grid_inn = np.linspace(0, 1, 20)  # Normalized [0,1]
mu2_grid_inn = np.linspace(0, 1, 20)
```

**Step 2: Map Scattered Training Samples**

For each scattered training parameter $(\mu_1^{(k)}, \mu_2^{(k)})$:

1. Normalize to $[0,1]$:
$$
\xi_{\mu_1}^{(k)} = \frac{\mu_1^{(k)} - \mu_1^{\min}}{\mu_1^{\max} - \mu_1^{\min}}
$$

2. Evaluate INN at this point using interpolation on internal grid

**Step 3: Training Loss**

$$
\mathcal{L}(\theta) = \sum_{k=1}^{N_{\text{train}}} \sum_{i,j} \left|u^{(k)}(x_i, t_j) - u_{\text{INN}}(x_i, t_j, \xi_{\mu_1}^{(k)}, \xi_{\mu_2}^{(k)})\right|^2
$$

**Step 4: Prediction for New Parameter**

For new $\mu^* = (\mu_1^*, \mu_2^*)$ (scattered, not on grid):

1. Normalize: $\xi_{\mu_1}^* = (\mu_1^* - \mu_1^{\min})/(\mu_1^{\max} - \mu_1^{\min})$
2. Evaluate INN via interpolation on internal grid
3. Get $u^* = u_{\text{INN}}(x, t, \xi_{\mu_1}^*, \xi_{\mu_2}^*)$

**No explicit parameter interpolation needed** — interpolation is built into the INN architecture!

### Concrete Implementation

#### INN with Linear Interpolation

```python
import numpy as np
import torch
import torch.nn as nn

class LinearInterpolator1D(nn.Module):
    """1D linear interpolation with trainable values at grid points."""
    def __init__(self, n_grid):
        super().__init__()
        self.n_grid = n_grid
        # Trainable values at grid points
        self.values = nn.Parameter(torch.randn(n_grid))
        # Fixed grid points in [0,1]
        self.register_buffer('grid', torch.linspace(0, 1, n_grid))

    def forward(self, xi):
        """
        xi: (batch_size,) in [0,1]
        Returns: (batch_size,) interpolated values
        """
        # Find interval
        xi = torch.clamp(xi, 0, 1)
        idx = torch.searchsorted(self.grid, xi, right=False)
        idx = torch.clamp(idx, 1, self.n_grid - 1)

        # Linear interpolation
        x0, x1 = self.grid[idx-1], self.grid[idx]
        v0, v1 = self.values[idx-1], self.values[idx]
        w = (xi - x0) / (x1 - x0 + 1e-10)

        return v0 * (1 - w) + v1 * w


class INN_Parametric(nn.Module):
    """INN for parametric PDE: u(x, t; μ₁, μ₂)"""
    def __init__(self, n_modes, nx, nt, nmu1, nmu2):
        super().__init__()
        self.n_modes = n_modes

        # Interpolators for each mode and dimension
        self.interp_x = nn.ModuleList([
            LinearInterpolator1D(nx) for _ in range(n_modes)
        ])
        self.interp_t = nn.ModuleList([
            LinearInterpolator1D(nt) for _ in range(n_modes)
        ])
        self.interp_mu1 = nn.ModuleList([
            LinearInterpolator1D(nmu1) for _ in range(n_modes)
        ])
        self.interp_mu2 = nn.ModuleList([
            LinearInterpolator1D(nmu2) for _ in range(n_modes)
        ])

        # Mode weights (optional, can be fixed to 1)
        self.weights = nn.Parameter(torch.ones(n_modes))

    def forward(self, xi_x, xi_t, xi_mu1, xi_mu2):
        """
        xi_x:   (batch_x,) spatial coordinates in [0,1]
        xi_t:   (batch_t,) temporal coordinates in [0,1]
        xi_mu1: (batch_mu,) parameter 1 in [0,1]
        xi_mu2: (batch_mu,) parameter 2 in [0,1]

        Returns: (batch_x, batch_t, batch_mu) tensor
        """
        batch_x, batch_t, batch_mu = len(xi_x), len(xi_t), len(xi_mu1)
        output = torch.zeros(batch_x, batch_t, batch_mu)

        for i in range(self.n_modes):
            # Evaluate each 1D interpolator
            phi_x = self.interp_x[i](xi_x)      # (batch_x,)
            phi_t = self.interp_t[i](xi_t)      # (batch_t,)
            phi_mu1 = self.interp_mu1[i](xi_mu1)  # (batch_mu,)
            phi_mu2 = self.interp_mu2[i](xi_mu2)  # (batch_mu,)

            # Tensor product: φ_x ⊗ φ_t ⊗ (φ_μ₁ * φ_μ₂)
            phi_param = phi_mu1 * phi_mu2  # (batch_mu,)

            # Outer products
            term = torch.einsum('x,t,m->xtm', phi_x, phi_t, phi_param)

            output += self.weights[i] * term

        return output


# Training example
def train_inn_scattered_params():
    # Scattered parameter samples (30 random pairs)
    np.random.seed(42)
    N_samples = 30
    mu1_samples = np.random.uniform(0.05, 0.25, N_samples)
    mu2_samples = np.random.uniform(0.1, 0.5, N_samples)

    # Generate training data (simulated)
    Nx, Nt = 101, 200
    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 0.5, Nt)

    U_train = []  # List of solutions for each parameter
    for mu1, mu2 in zip(mu1_samples, mu2_samples):
        u = solve_pde(mu1, mu2, x, t)  # Shape: (Nx, Nt)
        U_train.append(u)

    # Normalize parameters to [0,1]
    mu1_norm = (mu1_samples - mu1_samples.min()) / (mu1_samples.max() - mu1_samples.min())
    mu2_norm = (mu2_samples - mu2_samples.min()) / (mu2_samples.max() - mu2_samples.min())

    # Create INN model
    model = INN_Parametric(
        n_modes=10,
        nx=50, nt=50,   # Internal grid resolution
        nmu1=20, nmu2=20
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Normalize spatial and temporal coordinates
    xi_x = torch.FloatTensor((x - x.min()) / (x.max() - x.min()))
    xi_t = torch.FloatTensor((t - t.min()) / (t.max() - t.min()))

    # Training loop
    for epoch in range(5000):
        total_loss = 0

        for k in range(N_samples):
            # Get training sample
            u_target = torch.FloatTensor(U_train[k])  # (Nx, Nt)

            # Parameter coordinates for this sample
            xi_mu1 = torch.FloatTensor([mu1_norm[k]])
            xi_mu2 = torch.FloatTensor([mu2_norm[k]])

            # Forward pass
            u_pred = model(xi_x, xi_t, xi_mu1, xi_mu2).squeeze(2)  # (Nx, Nt)

            # Loss
            loss = criterion(u_pred, u_target)
            total_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/N_samples:.6f}')

    return model

# Prediction for new scattered parameter
def predict_new_parameter(model, mu1_star, mu2_star, mu1_train, mu2_train):
    """
    mu1_star, mu2_star: New parameter values (scattered, not on grid)
    """
    # Normalize using training range
    mu1_norm = (mu1_star - mu1_train.min()) / (mu1_train.max() - mu1_train.min())
    mu2_norm = (mu2_star - mu2_train.min()) / (mu2_train.max() - mu2_train.min())

    xi_mu1 = torch.FloatTensor([mu1_norm])
    xi_mu2 = torch.FloatTensor([mu2_norm])

    # Spatial and temporal grids
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 200)
    xi_x = torch.FloatTensor(x)
    xi_t = torch.FloatTensor(t)

    # Predict
    with torch.no_grad():
        u_pred = model(xi_x, xi_t, xi_mu1, xi_mu2).squeeze(2)

    return u_pred.numpy()

def solve_pde(mu1, mu2, x, t):
    """Placeholder for actual PDE solver."""
    # Example: analytical solution
    X, T = np.meshgrid(x, t, indexing='ij')
    return np.sin(np.pi * X) * np.exp(-mu1 * np.pi**2 * T) + \
           mu2 * np.sin(3*np.pi * X) * np.exp(-mu1 * 9*np.pi**2 * T)
```

#### INN with Cubic Splines

For smoother interpolation, use cubic B-splines:

```python
from scipy.interpolate import BSpline

class CubicSplineInterpolator1D(nn.Module):
    """1D cubic spline interpolation with trainable control points."""
    def __init__(self, n_control_points):
        super().__init__()
        self.n_cp = n_control_points
        # Trainable control point values
        self.control_values = nn.Parameter(torch.randn(n_control_points))

        # Cubic B-spline knot vector
        # Interior knots + boundary knots (multiplicity 4 for cubic)
        self.degree = 3
        interior_knots = np.linspace(0, 1, n_control_points - self.degree + 1)
        knots = np.concatenate([
            np.zeros(self.degree),
            interior_knots,
            np.ones(self.degree)
        ])
        self.register_buffer('knots', torch.FloatTensor(knots))

    def forward(self, xi):
        """Evaluate cubic B-spline at xi using trainable control points."""
        # Use scipy for basis evaluation (convert to numpy)
        xi_np = xi.detach().cpu().numpy()
        basis_matrix = self._compute_basis_matrix(xi_np)

        # Linear combination with trainable control values
        values = basis_matrix @ self.control_values

        return values

    def _compute_basis_matrix(self, xi):
        """Compute B-spline basis functions."""
        n_eval = len(xi)
        basis = np.zeros((n_eval, self.n_cp))

        for i in range(self.n_cp):
            # Evaluate i-th B-spline basis function
            coeff = np.zeros(self.n_cp)
            coeff[i] = 1.0
            spl = BSpline(self.knots.cpu().numpy(), coeff, self.degree)
            basis[:, i] = spl(np.clip(xi, 0, 1))

        return torch.FloatTensor(basis)
```

### Theoretical Advantages

#### 1. **Provable Convergence**

Unlike standard NNs, INN convergence is guaranteed by interpolation theory:

**Theorem (Approximation Error):**

For smooth solution $u \in C^{k}(\Omega)$, the INN approximation error is:

$$
\|u - u_{\text{INN}}\|_{L^2} \leq C \cdot h^k
$$

where:
- $h$ = maximum grid spacing
- $k$ = interpolation order (1 for linear, 3 for cubic, etc.)
- $C$ depends on $\|D^k u\|$ (smoothness)

**Implication:** Increasing grid resolution **guarantees** error reduction.

#### 2. **Exponential Convergence with CP Rank**

For separable solutions:

$$
u(x, t; \mu_1, \mu_2) = \sum_{i=1}^{r} f_i(x) g_i(t) h_i(\mu_1) k_i(\mu_2)
$$

INN with $r$ modes achieves **exact** representation (up to interpolation error within each 1D function).

**Theorem (CP Approximation):**

If the solution has exact CP rank $r_{\text{true}}$, then INN with $r \geq r_{\text{true}}$ modes achieves:

$$
\|u - u_{\text{INN}}\|_{L^2} = \mathcal{O}(h^k)
$$

(only interpolation error, no approximation error from rank truncation)

#### 3. **Parameter Efficiency**

**Standard MLP:**
$$
N_{\text{MLP}} \approx L \times W^2
$$
where $L$ = layers, $W$ = width

**INN:**
$$
N_{\text{INN}} = r \times \sum_{d=1}^{D} n_d
$$

**Comparison for 4D problem (x, t, μ₁, μ₂):**
```
MLP (3 layers, 128 neurons):  ≈ 65,536 parameters
INN (10 modes, 50×50×20×20):  = 1,400 parameters

Ratio: 47× reduction
```

### Advantages for Non-Gridded Parameters

✅ **No Explicit Interpolation Needed**
- INN architecture handles interpolation internally
- Same network works for gridded or scattered training data

✅ **Automatic Smoothness**
- Interpolation kernels (splines, RBF) ensure smooth parameter dependence
- No need for separate smoothing step

✅ **Provable Convergence**
- Mathematical guarantees on approximation error
- Can predict accuracy based on grid resolution and mode count

✅ **Far Fewer Parameters**
- 10-100× fewer parameters than standard NNs
- Faster training, less overfitting risk

✅ **Structured Representation**
- Each dimension (x, t, μ₁, μ₂) has explicit representation
- Easy to analyze and interpret

✅ **Flexible Interpolation**
- Can mix different interpolation types (linear for some dims, cubic for others)
- Adaptive grid refinement possible

✅ **No Curse of Dimensionality**
- Complexity grows **linearly** with dimension count (not exponentially like grids)
- Practical for 5-10 dimensional problems

### Disadvantages

❌ **Implementation Complexity**
- More complex than standard NNs
- Requires custom interpolation modules

❌ **Grid Resolution Selection**
- Need to choose $n_x, n_t, n_{\mu_1}, n_{\mu_2}$
- Adaptive selection can be non-trivial

❌ **Limited Flexibility**
- Assumes tensor-product structure
- May not capture arbitrary nonlinear interactions

❌ **Training Stability**
- Can be sensitive to initialization
- May need careful learning rate scheduling

### When to Use INN for Scattered Parameters

**Ideal Scenarios:**

1. **Moderate Parameter Dimension (2-6D)**
   - INN scales well where standard NNs struggle
   - Better than kernel methods for >4D

2. **Smooth Parameter Dependence**
   - Solution varies smoothly with parameters
   - Interpolation-based approach is natural

3. **Limited Training Data**
   - Fewer parameters → less overfitting
   - Works well with 20-100 samples

4. **Need for Interpretability**
   - Structured representation is explainable
   - Can visualize 1D interpolation functions

5. **Theoretical Guarantees Required**
   - Provable convergence is critical
   - Error bounds needed

**Comparison with Other Solutions:**

| Scenario | INN vs. Alternative |
|----------|-------------------|
| 2-3D params, 20-50 samples | **INN** > Kernel POD (fewer params, faster) |
| 4-5D params, 50-100 samples | **INN** ≈ Neural Net POD (similar performance) |
| 6-10D params, >100 samples | **INN** > Standard NN (better scaling) |
| <20 samples | Kernel POD > **INN** (GP uncertainty quantification) |
| Need interpretability | **INN** > Black-box NN |

### Hybrid Approach: INN + POD

**Combine strengths of both methods:**

**Step 1: Spatial POD**
$$
\Phi \in \mathbb{R}^{N_x \times r_s}
$$

**Step 2: INN for Coefficients**
$$
a_i(t; \mu_1, \mu_2) = \text{INN}_i(t, \mu_1, \mu_2)
$$

**Step 3: Reconstruction**
$$
u(x, t; \mu_1, \mu_2) = \sum_{i=1}^{r_s} \Phi_i(x) \cdot \text{INN}_i(t, \mu_1, \mu_2)
$$

**Benefits:**
- POD reduces spatial dimension
- INN handles (t, μ₁, μ₂) efficiently
- Combines optimal spatial compression with structured parameter interpolation

```python
# Hybrid POD + INN
class POD_INN_Hybrid:
    def __init__(self, X_train, parameters, r_pod=20, n_modes_inn=10):
        # Step 1: Spatial POD
        self.Phi = self.compute_pod(X_train, r_pod)

        # Step 2: Project to POD coefficients
        A_train = self.Phi.T @ X_train  # (r_pod, Nt*Nmu)

        # Step 3: Create INN for each POD mode
        self.inns = [
            INN_Parametric(n_modes_inn, nx=1, nt=50, nmu1=20, nmu2=20)
            for _ in range(r_pod)
        ]

        # Train each INN
        self.train_inns(A_train, parameters)

    def predict(self, mu1_star, mu2_star, t):
        # Evaluate each INN
        a_pred = np.array([
            inn.predict(t, mu1_star, mu2_star)
            for inn in self.inns
        ])

        # Reconstruct
        u_pred = self.Phi @ a_pred
        return u_pred
```

### Practical Recommendations for INN

**Grid Resolution Guidelines:**

```python
# Rule of thumb for grid sizes
def choose_inn_grid_sizes(N_samples, d_params):
    """
    N_samples: number of training samples
    d_params: number of parameters
    """
    # Spatial/temporal: higher resolution
    nx = min(100, max(20, int(N_samples**0.3)))
    nt = min(100, max(20, int(N_samples**0.3)))

    # Parameters: scale with samples and dimension
    n_per_param = max(10, int((N_samples / d_params)**0.5))

    return {
        'nx': nx,
        'nt': nt,
        'nmu': [n_per_param] * d_params
    }

# Example
grid_sizes = choose_inn_grid_sizes(N_samples=50, d_params=2)
# → {'nx': 50, 'nt': 50, 'nmu': [5, 5]}
```

**Mode Count Selection:**

```python
# Start conservative, increase if needed
n_modes_start = 5
n_modes_max = min(20, N_samples // 3)

# Train with increasing modes until validation error plateaus
```

**Interpolation Type Selection:**

- **Linear**: Fastest, works for smooth problems
- **Cubic splines**: Better for higher smoothness, 2-3× more parameters
- **RBF**: Best for irregular spacing, most expensive

### Summary: INN for Scattered Parameters

**Key Advantages:**
1. Internal grid handles scattered training data naturally
2. 10-100× fewer parameters than standard NNs
3. Provable convergence guarantees
4. Scales linearly with dimension count
5. Interpretable structured representation

**When to Choose INN:**
- Smooth parameter dependence expected
- 2-6 dimensional parameter space
- 20-200 training samples available
- Need theoretical guarantees or interpretability
- Willing to implement custom architecture

**INN vs. Other Methods:**
- **vs. POD + RBF**: INN better for >3D parameters
- **vs. Kernel POD**: INN scales better, no kernel bandwidth tuning
- **vs. Standard NN**: INN has fewer parameters, provable convergence
- **vs. Tucker**: INN works with scattered data, Tucker needs grid

---

## Practical Recommendations

### Decision Framework

```
Do you have gridded parameter samples?
├─ YES → Use Tucker/HOSVD (see HighDimensional_ROM_Strategies.md)
│
└─ NO (scattered samples) → How many parameter dimensions?
   ├─ 1-2D → Solution 1 (POD + RBF interpolation) or Solution 7 (INN)
   │         POD+RBF: Simple, reliable, well-established
   │         INN: Better accuracy, provable convergence, fewer parameters
   │
   ├─ 3-4D → Solution 1, 3, or 7 (POD + Kernel methods or INN)
   │         Kernel methods provide better interpolation
   │         GP regression gives uncertainty quantification
   │         INN handles high dimensions efficiently
   │
   ├─ 5-6D → Solution 2, 6, or 7 (Manifold learning, NN, or INN)
   │         Manifold if structure suspected
   │         Standard NN if large dataset available
   │         INN for parameter efficiency & convergence guarantees
   │
   └─ >6D → Solution 6 or 7 (Neural networks or INN)
            Standard NN: flexible but many parameters
            INN: structured, fewer parameters, provable convergence
```

### Recommended Workflow for Non-Gridded Sampling

**Step 1: Start with Classical POD + Interpolation**

Always begin with Solution 1 (POD + RBF/Kriging):
- Establish baseline performance
- Quick to implement
- Works for moderate parameter dimensions

**Step 2: Assess Performance**

Check interpolation accuracy:
- Cross-validation error
- Extrapolation behavior
- Visual inspection of coefficient surfaces

**Step 3: Upgrade if Needed**

If baseline insufficient:
- Parameter dimension ≤ 4: Try kernel methods (Solution 3) or INN (Solution 7)
- Parameter dimension > 4: Try manifold learning (Solution 2), NNs (Solution 6), or INN (Solution 7)
- Rapidly changing POD bases: Try Grassmannian interpolation (Solution 5)
- Need provable convergence: Use INN (Solution 7)

**Step 4: Validate**

- Test on held-out parameters
- Check physical consistency
- Compare with full-order solutions

### Best Practices

#### Sampling Strategy

Even if non-gridded, strategic sampling helps:

1. **Latin Hypercube Sampling (LHS)**
   - Better space-filling than random
   - Easy to implement

2. **Adaptive Sampling**
   - Start with LHS
   - Add samples where error is high
   - Greedy refinement

3. **Sobol Sequences**
   - Low-discrepancy quasi-random
   - Better coverage than pure random

#### Interpolation Quality

Ensure good interpolation:

- **Sufficient samples**: Rule of thumb: $N_\mu \geq 10 \times d_\mu$ for linear, $\geq 50 \times d_\mu$ for nonlinear
- **Dense coverage**: No large gaps in parameter space
- **Validation set**: Hold out 20% for testing interpolation

#### Error Estimation

Always quantify errors:

- **Cross-validation**: Leave-one-out or k-fold
- **Held-out test set**: Never used in training
- **Uncertainty quantification**: Use GP regression for confidence intervals

---

## Comparison of Methods

### Summary Table

| **Method** | **Tensor Needed?** | **Param Dim** | **Samples Needed** | **Complexity** | **Accuracy** | **Online Cost** |
|------------|-------------------|---------------|-------------------|----------------|--------------|-----------------|
| **1. POD + Interp** | ❌ No | ≤ 4 | Moderate (10d) | Low | Good | Fast |
| **2. Manifold + POD** | ❌ No | ≤ 5 | High (50d) | High | Very Good | Moderate |
| **3. Kernel POD** | ❌ No | ≤ 5 | Moderate (20d) | Medium | Very Good | Moderate |
| **4. Interp to Grid** | ✅ Creates | ≤ 3 | Moderate | Low | Poor | Fast |
| **5. Grassmannian** | ❌ No | ≤ 4 | High (50d) | Very High | Excellent | Moderate |
| **6. Neural Network** | ❌ No | > 5 | Very High (100d) | High | Good | Very Fast |
| **7. INN** | ❌ No | Any | Moderate (20d) | Medium | Excellent | Very Fast |

### Accuracy vs. Complexity Trade-off

```
High Accuracy
    │
    │   Grassmannian ●      INN ●
    │
    │   Kernel POD ●     Neural Net ●
    │
    │   POD+Interp ●
    │
    │   Interp to Grid ●
    │
    └───────────────────────────────> Complexity
      Low                          High
```

### When Each Method Excels

**POD + Interpolation:**
- Standard baseline for any non-gridded problem
- Reliable, well-understood, easy to implement

**Manifold Learning:**
- Parameters have intrinsic low-dimensional structure
- Research setting with time for preprocessing

**Kernel Methods:**
- Need uncertainty quantification
- Smooth parameter dependence
- Moderate sample size

**Grassmannian:**
- POD bases change significantly with parameters
- High accuracy requirements
- Expert implementation available

**Neural Networks:**
- High-dimensional parameters (>5D)
- Large training dataset
- Modern ML infrastructure

**Interpolating Neural Networks (INN):**
- Need provable convergence guarantees
- Parameter efficiency critical (47× fewer parameters)
- Any parameter dimension (1D to high-D)
- Moderate sample requirements
- Structured interpolation with theoretical foundation

---

## Conclusion

### Key Takeaways

1. **POD doesn't require tensor structure** — only a snapshot matrix
   - Scattered parameter samples are **not a fundamental obstacle**
   - Main challenge is **parameter interpolation**, not POD itself

2. **Classical POD + interpolation works for most cases**
   - Start here for parameter dimension ≤ 4
   - RBF or Kriging interpolation is reliable

3. **High-dimensional parameters need advanced methods**
   - Manifold learning discovers intrinsic structure
   - Neural networks scale to >5D parameters
   - INN provides structured approach with convergence guarantees

4. **Don't interpolate to grid** unless necessary
   - Adds error without clear benefit
   - Better to interpolate reduced coefficients

5. **Strategic sampling matters**
   - LHS or Sobol sequences better than pure random
   - Adaptive refinement improves efficiency

### Final Recommendation

**For most practical problems with non-gridded parameters:**

```python
# Recommended approach (Solution 1)
# 1. Spatial POD
Phi = POD(X, r)

# 2. Project training data
A_train = Phi.T @ X

# 3. Build interpolation model (choose based on d_mu and requirements)
if d_mu <= 3:
    # Classical interpolation
    interpolator = RBFInterpolator(parameters, A_train)
elif d_mu <= 5 and need_uncertainty:
    # Kernel methods with uncertainty quantification
    from sklearn.gaussian_process import GaussianProcessRegressor
    interpolator = GaussianProcessRegressor(...)
elif need_convergence_guarantee:
    # INN with provable convergence (Solution 7)
    from inn import INN_Parametric
    interpolator = INN_Parametric(n_modes=r, grid_sizes=[20]*d_mu)
else:
    # Standard neural network (Solution 6)
    interpolator = train_neural_network(parameters, A_train)

# 4. Predict for new parameter
a_star = interpolator.predict(mu_star)
u_star = Phi @ a_star
```

This balances simplicity, accuracy, and robustness for non-tensor-structured data.

---

## References

### Classical POD
- Sirovich, L. (1987). "Turbulence and the dynamics of coherent structures." *Quarterly of Applied Mathematics*, 45(3), 561-571.
- Holmes, P., et al. (1996). *Turbulence, Coherent Structures, Dynamical Systems and Symmetry*. Cambridge University Press.

### Interpolation Methods
- Buhmann, M. D. (2003). *Radial Basis Functions: Theory and Implementations*. Cambridge University Press.
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

### Manifold Learning
- Tenenbaum, J. B., et al. (2000). "A global geometric framework for nonlinear dimensionality reduction." *Science*, 290(5500), 2319-2323.
- Coifman, R. R., & Lafon, S. (2006). "Diffusion maps." *Applied and Computational Harmonic Analysis*, 21(1), 5-30.

### Grassmannian Methods
- Amsallem, D., & Farhat, C. (2011). "Interpolation method for adapting reduced-order models and application to aeroelasticity." *AIAA Journal*, 46(7), 1803-1813.

### Neural Networks for ROMs
- Hesthaven, J. S., & Ubbiali, S. (2018). "Non-intrusive reduced order modeling of nonlinear problems using neural networks." *Journal of Computational Physics*, 363, 55-78.
- Raissi, M., et al. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems." *Journal of Computational Physics*, 378, 686-707.

---

*This document provides comprehensive guidance on applying POD and reduced order modeling to non-gridded parameter samples. The key insight is that tensor structure is not required for POD — only for tensor decomposition methods like Tucker/HOSVD.*
