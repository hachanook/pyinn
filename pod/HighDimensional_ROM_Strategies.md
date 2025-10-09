# Dimensional Reduction Strategies for High-Dimensional Problems

## Question Addressed

**For high-dimensional problems, can we reduce only one dimension among others?**

For instance, if we have a 6-dimensional problem $\mathcal{U} \in \mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}$, should we select one dimension (e.g., x-direction) and flatten the snapshot tensor accordingly?

---

## Table of Contents

1. [Short Answer](#short-answer)
2. [Strategy 1: Full Spatial Flattening (Classical POD-Galerkin)](#strategy-1-full-spatial-flattening-classical-pod-galerkin)
3. [Strategy 2: Full Tensor Decomposition (Tucker/CP/Tensor-Train)](#strategy-2-full-tensor-decomposition-tuckercptensor-train)
4. [Strategy 3: Hybrid Grouping (Practical Compromise)](#strategy-3-hybrid-grouping-practical-compromise)
5. [When to Use Which Strategy](#when-to-use-which-strategy)
6. [Answering the Core Question: Single Dimension Reduction](#answering-the-core-question-single-dimension-reduction)
7. [Practical Recommendation for 6D Problems](#practical-recommendation-for-6d-problems)
8. [Mathematical Comparison](#mathematical-comparison)
9. [Compression Efficiency Analysis](#compression-efficiency-analysis)

---

## Short Answer

**No, you typically don't reduce only one dimension for high-dimensional problems.** For a 6D problem like $\mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}$, you have **three main strategies**:

1. **Full Flattening (POD approach)** - Flatten ALL spatial dimensions, keep time/parameters separate
2. **Tensor Decomposition (Tucker/CP approach)** - Reduce EACH dimension independently
3. **Hybrid Approach** - Group related dimensions strategically

Reducing only one dimension (e.g., just x) would waste the compression potential of the other dimensions.

---

## Strategy 1: Full Spatial Flattening (Classical POD-Galerkin)

### Core Concept

Flatten **ALL spatial dimensions** $(N_x, N_y, N_z)$ into a single "space" dimension, treat time and parameters separately.

### Mathematical Formulation

**Original Tensor:**
$$
\mathcal{U} \in \mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 1: Reshape to Group Spatial Dimensions**
$$
\mathcal{U} \in \mathbb{R}^{\underbrace{(N_x \cdot N_y \cdot N_z)}_{\text{spatial DOFs}} \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 2: Flatten for Snapshot POD**
$$
X = \text{reshape}(\mathcal{U}) \in \mathbb{R}^{(N_x \cdot N_y \cdot N_z) \times (N_t \cdot N_{p_1} \cdot N_{p_2})}
$$

**Step 3: Perform SVD**
$$
X = U \Sigma V^T
$$

**Step 4: Extract Spatial POD Modes**
$$
\Phi \in \mathbb{R}^{(N_x \cdot N_y \cdot N_z) \times r}
$$

where $r \ll N_x \cdot N_y \cdot N_z$ is the truncation rank.

**Step 5: Reduced Representation**
$$
u_r(x,y,z,t;\mu_1,\mu_2) = \Phi \, a(t;\mu_1,\mu_2)
$$

where $a \in \mathbb{R}^r$ are the reduced coefficients.

### Concrete Example: 3D Heat Equation

**Problem Setup:**
```
Solution: u(x,y,z,t; μ₁, μ₂) ∈ ℝ^(50×50×50×200×5×5)

Dimensions:
- Nx = 50 spatial points in x
- Ny = 50 spatial points in y
- Nz = 50 spatial points in z
- Nt = 200 time steps
- Np₁ = 5 values of parameter μ₁
- Np₂ = 5 values of parameter μ₂

Total: 50 × 50 × 50 × 200 × 5 × 5 = 62,500,000 values
```

**Flattening Process:**
```
Spatial flattening:
X = [u(x₁,y₁,z₁,·), u(x₂,y₁,z₁,·), ..., u(x₅₀,y₅₀,z₅₀,·)]ᵀ
  ∈ ℝ^(125,000 × 5,000)

Explanation:
- Rows: 50×50×50 = 125,000 spatial degrees of freedom
- Cols: 200×5×5 = 5,000 snapshots (all time-parameter combinations)
```

**POD Compression:**
```
SVD: X = U Σ Vᵀ
Select r = 20 dominant modes

Result:
- POD modes: Φ ∈ ℝ^(125,000 × 20)
- Reduced state: a ∈ ℝ^20
- Compression: 125,000 → 20 (6,250× reduction in spatial dimension)
```

### Advantages

✅ **Standard POD Theory**: Well-established mathematical framework applies directly

✅ **Galerkin Projection Compatible**: Can project spatial operators (Laplacian, gradient, etc.) onto reduced space
$$
L_r = \Phi^T L \Phi \in \mathbb{R}^{r \times r}
$$

✅ **Physics-Informed**: Respects spatial structure and governing equation operators

✅ **Mature Implementations**: Robust algorithms and software libraries available

✅ **Error Bounds**: Clear theoretical guarantees on approximation error
$$
\|u - u_r\|^2 \leq \sum_{i>r} \sigma_i^2
$$

### Disadvantages

❌ **Ignores Tensor Structure**: Treats 3D spatial field as flat vector, losing geometric information

❌ **Curse of Dimensionality**: For high spatial resolution, $N_x \cdot N_y \cdot N_z$ becomes enormous
- Example: $100 \times 100 \times 100 = 10^6$ spatial DOFs

❌ **No Directional Separability**: Doesn't exploit potential independence in x, y, z directions

❌ **Large SVD Cost**: Computing SVD on $125k \times 5k$ matrix is expensive
- Complexity: $\mathcal{O}(\min(m^2 n, m n^2))$ where $m=125k$, $n=5k$

❌ **Memory Requirements**: Storing full snapshot matrix $X$ can exceed available RAM

### When to Use

- Spatial dimension is moderate: $N_x \cdot N_y \cdot N_z < 10^6$
- Intrusive ROM with Galerkin projection is required
- Governing equation operators must be preserved
- Physics-based accuracy is critical
- You have computational resources for large SVD

---

## Strategy 2: Full Tensor Decomposition (Tucker/CP/Tensor-Train)

### Core Concept

Treat the problem as a **true 6D tensor** and reduce **each dimension independently** using tensor decomposition methods.

### Tucker Decomposition Formulation

**Decomposition:**
$$
\mathcal{U} \approx \mathcal{G} \times_1 \Phi_x \times_2 \Phi_y \times_3 \Phi_z \times_4 \Phi_t \times_5 \Phi_{p_1} \times_6 \Phi_{p_2}
$$

**Component Definitions:**
- $\Phi_x \in \mathbb{R}^{N_x \times r_x}$ — Spatial modes in x-direction
- $\Phi_y \in \mathbb{R}^{N_y \times r_y}$ — Spatial modes in y-direction
- $\Phi_z \in \mathbb{R}^{N_z \times r_z}$ — Spatial modes in z-direction
- $\Phi_t \in \mathbb{R}^{N_t \times r_t}$ — Temporal modes
- $\Phi_{p_1} \in \mathbb{R}^{N_{p_1} \times r_{p_1}}$ — Parameter 1 modes
- $\Phi_{p_2} \in \mathbb{R}^{N_{p_2} \times r_{p_2}}$ — Parameter 2 modes
- $\mathcal{G} \in \mathbb{R}^{r_x \times r_y \times r_z \times r_t \times r_{p_1} \times r_{p_2}}$ — Core tensor

**Explicit Form:**
$$
\mathcal{U}(i,j,k,\ell,m,n) \approx \sum_{\alpha=1}^{r_x} \sum_{\beta=1}^{r_y} \sum_{\gamma=1}^{r_z} \sum_{\tau=1}^{r_t} \sum_{\pi=1}^{r_{p_1}} \sum_{\rho=1}^{r_{p_2}}
\mathcal{G}(\alpha,\beta,\gamma,\tau,\pi,\rho) \, \Phi_x(i,\alpha) \, \Phi_y(j,\beta) \, \Phi_z(k,\gamma) \, \Phi_t(\ell,\tau) \, \Phi_{p_1}(m,\pi) \, \Phi_{p_2}(n,\rho)
$$

### HOSVD Algorithm (Higher-Order SVD)

**Step 1: Mode-1 Unfolding (x-direction)**
$$
U^{(1)} = \mathcal{U}_{(1)} \in \mathbb{R}^{N_x \times (N_y \cdot N_z \cdot N_t \cdot N_{p_1} \cdot N_{p_2})}
$$
Perform SVD: $U^{(1)} = U_1 \Sigma_1 V_1^T$

Extract: $\Phi_x = U_1(:, 1:r_x)$

**Step 2-6: Repeat for Other Modes**
Similarly unfold along dimensions 2-6 to obtain $\Phi_y, \Phi_z, \Phi_t, \Phi_{p_1}, \Phi_{p_2}$

**Step 7: Compute Core Tensor**
$$
\mathcal{G} = \mathcal{U} \times_1 \Phi_x^T \times_2 \Phi_y^T \times_3 \Phi_z^T \times_4 \Phi_t^T \times_5 \Phi_{p_1}^T \times_6 \Phi_{p_2}^T
$$

### Concrete Example with Compression Analysis

**Original Tensor:**
```
Dimensions: ℝ^(50×50×50×200×5×5)
Total parameters: 62,500,000
```

**Tucker Decomposition with Ranks:**
```
Chosen ranks: (rx, ry, rz, rt, rp₁, rp₂) = (8, 8, 8, 10, 4, 4)

Component storage:
  Φₓ:  50 × 8     =      400 values
  Φᵧ:  50 × 8     =      400 values
  Φᵧ:  50 × 8     =      400 values
  Φₜ:  200 × 10   =    2,000 values
  Φₚ₁:  5 × 4     =       20 values
  Φₚ₂:  5 × 4     =       20 values
  Core: 8×8×8×10×4×4 = 163,840 values
  ----------------------------------------
  Total:             167,080 values

Compression ratio: 62,500,000 / 167,080 = 374×
Percentage: 0.267% of original size
```

### Parameter Interpolation for New Values

For a new parameter combination $(\mu_1^*, \mu_2^*)$ not in training set:

**Step 1: Interpolate in Parameter Space**
$$
\phi_{p_1}^* = \text{interp}(\Phi_{p_1}, \mu_1^*) \in \mathbb{R}^{r_{p_1}}
$$
$$
\phi_{p_2}^* = \text{interp}(\Phi_{p_2}, \mu_2^*) \in \mathbb{R}^{r_{p_2}}
$$

**Step 2: Contract Core Tensor**
$$
H = \mathcal{G} \times_5 \phi_{p_1}^* \times_6 \phi_{p_2}^* \in \mathbb{R}^{r_x \times r_y \times r_z \times r_t}
$$

**Step 3: Reconstruct Solution**
$$
\hat{u}(x,y,z,t;\mu_1^*,\mu_2^*) = H \times_1 \Phi_x \times_2 \Phi_y \times_3 \Phi_z \times_4 \Phi_t
$$

### Advantages

✅ **Exploits Structure in Each Dimension**: Reduces independently in x, y, z, t, and parameters

✅ **Exponential Compression for High-D**: Compression ratio grows exponentially with dimension count

✅ **Separable Representation**: Each direction treated with appropriate resolution
$$
u \approx \sum_{\text{modes}} c_{\alpha\beta\gamma\tau\pi\rho} \, \phi_x^\alpha \phi_y^\beta \phi_z^\gamma \phi_t^\tau \phi_{p_1}^\pi \phi_{p_2}^\rho
$$

✅ **Anisotropic Rank Selection**: Can use different ranks per dimension
- High resolution in critical directions
- Low resolution in homogeneous directions

✅ **Avoids Curse of Dimensionality**: Complexity grows linearly (not exponentially) with number of dimensions

✅ **Memory Efficient**: Never need to store full tensor in memory

### Disadvantages

❌ **Non-Intrusive Only**: Difficult to perform Galerkin projection on 6D core tensor

❌ **No Direct Operator Projection**: Can't easily compute $\Phi^T L \Phi$ for 6D decomposition

❌ **Implementation Complexity**: More sophisticated algorithms and data structures required

❌ **Core Tensor Size**: For many dimensions, core can still be large
- Example: $10^6$ elements for $10 \times 10 \times 10 \times 10 \times 10 \times 10$ core

❌ **Interpolation Required**: New parameter values require interpolation (adds error)

❌ **Less Mature Theory**: Fewer error bounds and convergence guarantees than POD

### When to Use

- Problem has ≥ 5 dimensions
- Each dimension exhibits structure that can be exploited
- Non-intrusive approach is acceptable
- Memory/storage is critical constraint
- Solution exhibits separability in different directions
- Pure data-driven approach preferred

---

## Strategy 3: Hybrid Grouping (Practical Compromise)

### Core Concept

**Strategically group related dimensions** based on physical or mathematical structure, combining benefits of POD and tensor decomposition.

### Option 3A: Group Spatial, Separate Time/Parameters

**Approach:**
$$
\mathcal{U} \in \mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 1: Flatten Spatial Dimensions**
$$
\mathcal{U}_{\text{flat}} \in \mathbb{R}^{(N_x \cdot N_y \cdot N_z) \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 2: Spatial POD**
$$
X_{\text{space}} \in \mathbb{R}^{(N_x N_y N_z) \times (N_t \cdot N_{p_1} \cdot N_{p_2})}
$$
SVD → $\Phi_{\text{space}} \in \mathbb{R}^{(N_x N_y N_z) \times r_s}$

**Step 3: Project to Reduced Spatial Basis**
$$
\mathcal{U}_r = \Phi_{\text{space}}^T \mathcal{U}_{\text{flat}} \in \mathbb{R}^{r_s \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 4: Tucker on Remaining 4D Tensor**
$$
\mathcal{U}_r \approx \mathcal{G} \times_2 \Phi_t \times_3 \Phi_{p_1} \times_4 \Phi_{p_2}
$$

where $\mathcal{G} \in \mathbb{R}^{r_s \times r_t \times r_{p_1} \times r_{p_2}}$

**Final Representation:**
$$
u(x,y,z,t;\mu_1,\mu_2) \approx \Phi_{\text{space}} \left[ \mathcal{G} \times_2 \Phi_t \times_3 \Phi_{p_1} \times_4 \Phi_{p_2} \right]
$$

### Concrete Example

**Problem:**
```
Original: ℝ^(50×50×50×200×5×5) = 62,500,000 parameters
```

**Hybrid Decomposition:**
```
Ranks: rₛ=20, rₜ=10, rₚ₁=4, rₚ₂=4

Components:
  Φₛₚₐ꜀ₑ: 125,000 × 20   = 2,500,000
  Φₜ:        200 × 10     =     2,000
  Φₚ₁:         5 × 4      =        20
  Φₚ₂:         5 × 4      =        20
  Core:   20×10×4×4       =     3,200
  --------------------------------------
  Total:                    2,505,240

Compression ratio: 62,500,000 / 2,505,240 = 25×
Percentage: 4% of original size
```

### Option 3B: Separate Spatial, Group Parameters

**Alternative Grouping:**
$$
\mathcal{U} \approx \mathcal{G} \times_1 \Phi_x \times_2 \Phi_y \times_3 \Phi_z \times_4 \Phi_t \times_5 \Phi_{\text{params}}
$$

where $\Phi_{\text{params}} \in \mathbb{R}^{(N_{p_1} \cdot N_{p_2}) \times r_p}$ combines both parameters.

**When Useful:**
- Parameters are correlated or coupled
- Want to preserve spatial tensor structure
- Parameter space is low-dimensional

### Advantages

✅ **Balanced Complexity**: More sophisticated than pure POD, simpler than full tensor

✅ **Galerkin Capability**: Can still project spatial operators (on flattened space)

✅ **Exploits Temporal/Parametric Structure**: Uses tensor decomposition where most beneficial

✅ **Manageable Core Tensor**: 4D core much smaller than 6D core

✅ **Flexible Trade-offs**: Can adjust which dimensions to group

✅ **Good Compression**: Better than pure POD for high-dimensional parameter spaces

### Disadvantages

❌ **Doesn't Fully Exploit Spatial Separability**: Still treats x,y,z as single flattened dimension

❌ **Requires Strategic Decisions**: Must choose grouping based on problem structure

❌ **Intermediate Complexity**: More complex than POD, less compression than full tensor

### When to Use

- Want intrusive ROM capability with some tensor compression
- Moderate dimension count (4-5D)
- Some dimensions naturally group (spatial coordinates, related parameters)
- Need balance between compression and implementation complexity
- Hybrid physics-informed and data-driven approach desired

---

## When to Use Which Strategy?

### Decision Tree

```
Is intrusive ROM with Galerkin projection required?
├─ YES → Is spatial dimension manageable (< 10⁶ DOFs)?
│        ├─ YES → Strategy 1: Full Spatial Flattening (POD-Galerkin)
│        └─ NO  → Strategy 3A: Hybrid (spatial POD + temporal/param tensor)
│
└─ NO  → How many dimensions?
         ├─ ≤ 4D → Strategy 1 or 3 (depending on problem structure)
         └─ ≥ 5D → Strategy 2: Full Tensor Decomposition
```

### Detailed Selection Criteria

#### Use **Strategy 1: Full Spatial Flattening** When:

🎯 **Intrusive ROM Required**
- Need Galerkin projection of governing equations
- Must preserve spatial operators (Laplacian, gradient, divergence)
- Physics-based accuracy is critical

🎯 **Spatial Dimension Moderate**
- $N_x \cdot N_y \cdot N_z < 10^6$ (manageable matrix size)
- Can afford SVD on large spatial matrix
- Memory available for snapshot matrix

🎯 **Few Parameters**
- Time and parameter variation is limited
- Tensor decomposition overhead not justified

🎯 **Examples:**
- 2D/3D Navier-Stokes with 1-2 parameters
- 3D heat equation with moderate resolution
- Structural dynamics with few loading scenarios

#### Use **Strategy 2: Full Tensor Decomposition** When:

🎯 **High Dimension Count**
- Problem has ≥ 5 dimensions
- Each dimension has significant size ($N_i > 10$)

🎯 **Separability Expected**
- Solution exhibits product structure: $u \approx f(x) g(y) h(z) k(t) \ell(\mu)$
- Spatial directions are weakly coupled
- Parameters affect solution in separable ways

🎯 **Memory Constraints**
- Cannot store full tensor in memory
- Need extreme compression ratios

🎯 **Non-Intrusive Acceptable**
- Data-driven approach preferred
- Don't need Galerkin projection
- Can tolerate interpolation errors

🎯 **Examples:**
- 6D Boltzmann equation (3D space + 3D velocity)
- Multi-parameter optimization (many design variables)
- Uncertainty quantification with high-dimensional stochastic space
- High-resolution multi-physics simulations

#### Use **Strategy 3: Hybrid Approaches** When:

🎯 **Moderate Complexity**
- 4-5 dimensional problems
- Want benefits of both POD and tensor methods

🎯 **Natural Groupings**
- Some dimensions are physically related (x,y,z coordinates)
- Others are independent (time, parameters)

🎯 **Partial Intrusive Capability**
- Need some Galerkin projection (spatial operators)
- But also want parametric compression

🎯 **Resource Constraints**
- More compression than pure POD needed
- But full tensor decomposition too complex

🎯 **Examples:**
- 3D turbulent flow with 2-3 physical parameters
- Multi-scale problems (space + time + scale parameter)
- Design optimization with moderate parameter count

---

## Answering the Core Question: Single Dimension Reduction

### Can We Reduce Only One Dimension (e.g., Only x)?

**Short Answer: Technically yes, but it's almost never optimal.**

### Mathematical Formulation

**Original Tensor:**
$$
\mathcal{U} \in \mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Mode-1 Unfolding (Only x-direction):**
$$
U^{(1)} = \mathcal{U}_{(1)} \in \mathbb{R}^{N_x \times (N_y \cdot N_z \cdot N_t \cdot N_{p_1} \cdot N_{p_2})}
$$

**SVD on Mode-1:**
$$
U^{(1)} = U_1 \Sigma_1 V_1^T
$$

**Extract x-direction Modes:**
$$
\Phi_x \in \mathbb{R}^{N_x \times r_x}
$$

**Result:**
$$
\mathcal{U} \approx \Phi_x \times_1 \mathcal{C}
$$

where $\mathcal{C} \in \mathbb{R}^{r_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}$ is the coefficient tensor.

### Compression Analysis

**Original Size:**
$$
N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}
$$

**After x-only Reduction:**
$$
\underbrace{N_x \times r_x}_{\Phi_x} + \underbrace{r_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}_{\mathcal{C}}
$$

**Concrete Example:**
```
Original: 50 × 50 × 50 × 200 × 5 × 5 = 62,500,000

Only x reduced (rx = 8):
  Φₓ:  50 × 8                          =        400
  C:    8 × 50 × 50 × 200 × 5 × 5      = 10,000,000
  -------------------------------------------------
  Total:                                 10,000,400

Compression: 62.5M → 10M (only 6.25× reduction!)
```

### Why This Is Rarely Optimal

#### 1. **Severely Limited Compression**

Only one dimension reduced means most of the data remains:

$$
\text{Compression ratio} = \frac{N_x}{r_x} \times \frac{N_x + r_x \cdot N_y N_z N_t N_{p_1} N_{p_2}}{N_x N_y N_z N_t N_{p_1} N_{p_2}}
$$

For typical values, this is minimal (6-10× at best).

**Comparison:**
```
Strategy                      Compression Ratio
--------------------------------------------
Only x reduced (rx=8)                6.25×
All spatial (rs=20)                   25×
Full tensor (8,8,8,10,4,4)           374×
```

#### 2. **Ignores Structure in Other Dimensions**

If the solution has structure in x, it likely has similar structure in y and z:

- **Physical isotropy**: Most physical systems don't privilege one spatial direction
- **Wasted opportunity**: Y and Z directions could be compressed just as well
- **No mathematical justification**: Why would x be special?

#### 3. **Computational Cost Not Reduced Much**

Even with x reduced, you still have:
- $N_y \times N_z$ spatial DOFs in other directions
- $N_t$ time steps to simulate
- $N_{p_1} \times N_{p_2}$ parameter evaluations

**Example**: With only x reduced (8 modes), still need to handle:
$$
8 \times 50 \times 50 \times 200 = 4,000,000 \text{ values per parameter combination}
$$

#### 4. **Missing Parametric Compression**

Parameters often have the most structure to exploit:

- Parameter sweeps create smooth manifolds
- Often high correlation between parameter values
- Tensor decomposition excels at finding these patterns

**Example**: For parameters on a grid:
- $N_{p_1} \times N_{p_2} = 5 \times 5 = 25$ combinations
- Tensor decomposition → $r_{p_1} \times r_{p_2} = 4 \times 4 = 16$ modes
- But single-dimension reduction doesn't touch this!

### When Single-Dimension Reduction Might Be Justified

#### Scenario 1: Strong Physical Anisotropy

**Example: Boundary Layer Flow**
```
Channel flow with wall-normal coordinate y:

- x (streamwise):  homogeneous, periodic → reduce heavily (rx = 5)
- y (wall-normal): boundary layer, high gradients → keep full (ry = Ny = 100)
- z (spanwise):    homogeneous, periodic → reduce heavily (rz = 5)

Decomposition:
U ≈ Φₓ(x) ⊗ u_full(y) ⊗ Φᵧ(z) ⊗ Φₜ(t)

Result: y kept at full resolution, others reduced
```

**But even here**, better to use **anisotropic Tucker**:
$$
\mathcal{U} \approx \mathcal{G} \times_1 \Phi_x \times_2 I_y \times_3 \Phi_z \times_4 \Phi_t
$$
where $I_y$ is identity (no reduction in y), but other dimensions are reduced.

This is **not** single-dimension reduction—it's selective multi-dimensional reduction.

#### Scenario 2: Diagnostic/Exploratory Analysis

**Use Case**: Understanding which dimensions contain most structure

```python
# Explore structure in each dimension
for dim in [x, y, z, t, p1, p2]:
    singular_values = mode_n_SVD(U, mode=dim)
    plot_energy_decay(singular_values)

# Identify which dimensions can be reduced most
# → Then use full tensor decomposition with appropriate ranks
```

This is a **diagnostic step**, not the final compression strategy.

### Correct Approach: Multi-Dimensional Reduction

Instead of reducing only x, reduce **all dimensions with appropriate ranks**:

$$
\mathcal{U} \approx \mathcal{G} \times_1 \Phi_x \times_2 \Phi_y \times_3 \Phi_z \times_4 \Phi_t \times_5 \Phi_{p_1} \times_6 \Phi_{p_2}
$$

**Ranks can be different** (anisotropic):
```
Example with boundary layer:
rx = 10   (streamwise: moderate variation)
ry = 50   (wall-normal: high gradients, less reduction)
rz = 10   (spanwise: similar to streamwise)
rt = 15   (temporal: moderate dynamics)
rp₁ = 4   (parameter 1: smooth variation)
rp₂ = 4   (parameter 2: smooth variation)
```

This gives:
- Appropriate resolution in each dimension
- Massive overall compression
- Respects physical anisotropy

---

## Practical Recommendation for 6D Problems

### Recommended Workflow for $\mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}$

#### **Step 1: Assess Requirements**

**Questions to answer:**
1. Do you need intrusive ROM with Galerkin projection?
2. What is the spatial resolution? ($N_x \cdot N_y \cdot N_z$)
3. How many time steps? ($N_t$)
4. How many parameters? ($N_{p_1} \cdot N_{p_2}$)
5. Is the solution separable in different directions?
6. What computational resources are available?

#### **Step 2: Choose Strategy Based on Answers**

**Decision Matrix:**

| Intrusive? | Spatial Size | Dimensions | **Recommended Strategy** |
|------------|--------------|------------|--------------------------|
| Yes | < 10⁶ | Any | Strategy 1: Full Flatten |
| Yes | > 10⁶ | 4-6D | Strategy 3A: Hybrid |
| No | Any | 4D | Strategy 1 or 3 |
| No | Any | 5-6D | Strategy 2: Full Tensor |
| No | > 10⁶ | 5-6D | Strategy 2: Full Tensor |

#### **Step 3: Implement Chosen Strategy**

### **Recommended: Hybrid 4D Decomposition (Strategy 3A)**

This works for most practical 6D problems and balances all considerations.

**Algorithm:**

**Input:** $\mathcal{U} \in \mathbb{R}^{N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2}}$

**Step 1: Flatten Spatial Dimensions**
$$
X_{\text{space}} = \text{reshape}(\mathcal{U}) \in \mathbb{R}^{(N_x N_y N_z) \times (N_t N_{p_1} N_{p_2})}
$$

**Step 2: Spatial POD**
$$
X_{\text{space}} = U \Sigma V^T
$$
$$
\Phi_{\text{space}} = U(:, 1:r_s) \in \mathbb{R}^{(N_x N_y N_z) \times r_s}
$$

Choose $r_s$ to capture 99.9% energy:
$$
\frac{\sum_{i=1}^{r_s} \sigma_i^2}{\sum_{i=1}^{N_x N_y N_z} \sigma_i^2} \geq 0.999
$$

**Step 3: Project to Reduced Spatial Basis**
$$
\mathcal{U}_r = \Phi_{\text{space}}^T \mathcal{U}_{\text{flat}} \in \mathbb{R}^{r_s \times N_t \times N_{p_1} \times N_{p_2}}
$$

**Step 4: Tucker Decomposition on (t, p₁, p₂)**

Perform HOSVD on the 4D tensor $\mathcal{U}_r$:

**Mode-2 (time):**
$$
U_t^{(2)} = \mathcal{U}_r \text{ unfolded along time}
$$
$$
\Phi_t \in \mathbb{R}^{N_t \times r_t}
$$

**Mode-3 (parameter 1):**
$$
\Phi_{p_1} \in \mathbb{R}^{N_{p_1} \times r_{p_1}}
$$

**Mode-4 (parameter 2):**
$$
\Phi_{p_2} \in \mathbb{R}^{N_{p_2} \times r_{p_2}}
$$

**Core tensor:**
$$
\mathcal{G} = \mathcal{U}_r \times_2 \Phi_t^T \times_3 \Phi_{p_1}^T \times_4 \Phi_{p_2}^T \in \mathbb{R}^{r_s \times r_t \times r_{p_1} \times r_{p_2}}
$$

**Step 5: Reconstruction**

For new parameter values $(\mu_1^*, \mu_2^*)$:

1. Interpolate parameter modes:
$$
\phi_{p_1}^* = \text{interp}(\Phi_{p_1}, \mu_1^*)
$$
$$
\phi_{p_2}^* = \text{interp}(\Phi_{p_2}, \mu_2^*)
$$

2. Contract core tensor:
$$
H(t) = \mathcal{G} \times_3 \phi_{p_1}^* \times_4 \phi_{p_2}^* \in \mathbb{R}^{r_s \times r_t}
$$

3. Reconstruct in reduced spatial basis:
$$
a(t) = H \times_2 \Phi_t \in \mathbb{R}^{r_s}
$$

4. Reconstruct full solution:
$$
u(x,y,z,t;\mu_1^*,\mu_2^*) = \Phi_{\text{space}} \, a(t)
$$

### Compression Calculation

**Original:**
$$
N_x \times N_y \times N_z \times N_t \times N_{p_1} \times N_{p_2} = 50 \times 50 \times 50 \times 200 \times 5 \times 5 = 62.5M
$$

**Hybrid (rs=20, rt=10, rp₁=4, rp₂=4):**
$$
\begin{align}
\Phi_{\text{space}}: &\quad 125,000 \times 20 &&= 2,500,000 \\
\Phi_t: &\quad 200 \times 10 &&= 2,000 \\
\Phi_{p_1}: &\quad 5 \times 4 &&= 20 \\
\Phi_{p_2}: &\quad 5 \times 4 &&= 20 \\
\mathcal{G}: &\quad 20 \times 10 \times 4 \times 4 &&= 3,200 \\
\hline
\text{Total}: &\quad &&= 2,505,240
\end{align}
$$

**Compression ratio:**
$$
\frac{62,500,000}{2,505,240} = 24.95 \approx 25\times
$$

**Storage:** 4% of original

### Advantages of This Approach

✅ **Galerkin Projection Possible**: Can project spatial operators onto $\Phi_{\text{space}}$

✅ **Good Compression**: 25× better than pure POD with same spatial rank

✅ **Parameter Efficiency**: Tucker on (t, p₁, p₂) exploits temporal/parametric structure

✅ **Manageable Complexity**: 4D core tensor much smaller than 6D

✅ **Flexible**: Can adjust which dimensions to group based on problem

✅ **Implementable**: Standard tools (SVD + tensor toolboxes) available

### Implementation Pseudocode

```python
import numpy as np
from scipy.linalg import svd
import tensorly as tl
from tensorly.decomposition import tucker

# Step 1: Load snapshot tensor
U = load_snapshots()  # Shape: (Nx, Ny, Nz, Nt, Np1, Np2)

# Step 2: Flatten spatial dimensions
Nx, Ny, Nz, Nt, Np1, Np2 = U.shape
U_flat = U.reshape(Nx*Ny*Nz, Nt*Np1*Np2)

# Step 3: Spatial POD
Phi_space, S, Vt = svd(U_flat, full_matrices=False)

# Choose rank to capture 99.9% energy
energy = np.cumsum(S**2) / np.sum(S**2)
rs = np.argmax(energy >= 0.999) + 1

Phi_space = Phi_space[:, :rs]

# Step 4: Project to reduced spatial basis
U_reduced = Phi_space.T @ U_flat
U_reduced = U_reduced.reshape(rs, Nt, Np1, Np2)

# Step 5: Tucker decomposition on (t, p1, p2)
# Modes: [1, 2, 3] correspond to [t, p1, p2]
ranks = [rs, 10, 4, 4]  # Don't reduce mode 0 (already reduced spatial)
core, factors = tucker(U_reduced, rank=ranks)

Phi_t = factors[1]      # Time modes
Phi_p1 = factors[2]     # Parameter 1 modes
Phi_p2 = factors[3]     # Parameter 2 modes

# Step 6: Reconstruction for new parameters (mu1_star, mu2_star)
# Interpolate parameter modes
phi_p1_star = interpolate(Phi_p1, mu1_star, mu1_train)
phi_p2_star = interpolate(Phi_p2, mu2_star, mu2_train)

# Contract core tensor
H = tl.tenalg.multi_mode_dot(core,
                               [phi_p1_star, phi_p2_star],
                               modes=[2, 3])  # Shape: (rs, rt)

# Contract with time modes
a_t = H @ Phi_t.T  # Shape: (rs, Nt)

# Reconstruct full solution
u_full = Phi_space @ a_t  # Shape: (Nx*Ny*Nz, Nt)
u_full = u_full.reshape(Nx, Ny, Nz, Nt)
```

---

## Mathematical Comparison

### Summary Table

| **Aspect** | **Strategy 1: Flatten** | **Strategy 2: Full Tensor** | **Strategy 3: Hybrid** |
|------------|------------------------|----------------------------|----------------------|
| **Dimensions Reduced** | Spatial only | All 6 independently | Spatial + (t,p₁,p₂) |
| **Compression (example)** | ~300× | ~374× | ~25× |
| **Storage** | 0.3% | 0.27% | 4% |
| **Galerkin Projection** | ✅ Yes | ❌ Difficult | ✅ Yes |
| **Implementation** | Standard SVD | Tensor toolbox | SVD + Tensor |
| **Complexity** | Low | High | Medium |
| **Parametric Structure** | Not exploited | Fully exploited | Well exploited |
| **Best For** | 2-4D intrusive | 5-6D non-intrusive | 4-6D hybrid |

### Compression Ratio Formulas

**Strategy 1 (Flatten All Spatial):**
$$
C_1 = \frac{N_x N_y N_z N_t N_{p_1} N_{p_2}}{(N_x N_y N_z) \cdot r + r \cdot N_t N_{p_1} N_{p_2}}
$$

For $N_x N_y N_z \gg r$ and $N_t N_{p_1} N_{p_2}$ moderate:
$$
C_1 \approx \frac{N_x N_y N_z}{r}
$$

**Strategy 2 (Full Tucker):**
$$
C_2 = \frac{N_x N_y N_z N_t N_{p_1} N_{p_2}}{N_x r_x + N_y r_y + N_z r_z + N_t r_t + N_{p_1} r_{p_1} + N_{p_2} r_{p_2} + r_x r_y r_z r_t r_{p_1} r_{p_2}}
$$

For large $N_i$ and small $r_i$:
$$
C_2 \approx \frac{\prod N_i}{\sum N_i r_i}
$$

**Strategy 3 (Hybrid):**
$$
C_3 = \frac{N_x N_y N_z N_t N_{p_1} N_{p_2}}{(N_x N_y N_z) r_s + N_t r_t + N_{p_1} r_{p_1} + N_{p_2} r_{p_2} + r_s r_t r_{p_1} r_{p_2}}
$$

---

## Compression Efficiency Analysis

### Theoretical Analysis: Why Full Tensor Wins for High Dimensions

**Curse of Dimensionality in POD:**

For $d$ dimensions with size $N$ each:
- Original: $N^d$ parameters
- POD (flatten all): $N^d / N^{d-1} \cdot r = N \cdot r$ spatial modes, but still $N^{d-1}$ snapshots
- Effective compression: $\frac{N^d}{N \cdot r} = \frac{N^{d-1}}{r}$

**Blessing of Separability in Tucker:**

- Tucker: $d \cdot N \cdot r + r^d$ parameters
- For small $r$ and large $N$: compression $\approx \frac{N^d}{d \cdot N \cdot r} = \frac{N^{d-1}}{d \cdot r}$

**Ratio:**
$$
\frac{\text{Tucker compression}}{\text{POD compression}} = \frac{N^{d-1}/(d \cdot r)}{N^{d-1}/r} = \frac{1}{d}
$$

But actually much better because Tucker core is $r^d$ vs POD needs $r \cdot N^{d-1}$ coefficients!

### Numerical Examples

#### 4D Problem: $\mathbb{R}^{100 \times 100 \times 100 \times 100}$

```
Original: 100⁴ = 100,000,000 parameters

POD (r=10):
  Modes: 1,000,000 × 10 = 10,000,000
  Coeffs: 10 × 1,000,000 = 10,000,000
  Total: 20,000,000 (5× compression)

Tucker (10,10,10,10):
  Modes: 4 × (100×10) = 4,000
  Core: 10⁴ = 10,000
  Total: 14,000 (7,143× compression!)
```

#### 6D Problem: $\mathbb{R}^{50 \times 50 \times 50 \times 200 \times 5 \times 5}$

```
Original: 62,500,000

POD (r=20):
  Modes: 125,000 × 20 = 2,500,000
  Coeffs: 20 × 5,000 = 100,000
  Total: 2,600,000 (24× compression)

Tucker (8,8,8,10,4,4):
  Modes: 50×8 + 50×8 + 50×8 + 200×10 + 5×4 + 5×4
       = 400 + 400 + 400 + 2,000 + 20 + 20 = 3,240
  Core: 8×8×8×10×4×4 = 163,840
  Total: 167,080 (374× compression!)

Hybrid (rs=20, rt=10, rp₁=4, rp₂=4):
  Spatial modes: 2,500,000
  Temporal modes: 2,000
  Param modes: 40
  Core: 3,200
  Total: 2,505,240 (25× compression)
```

### Key Insight

**Compression grows exponentially with dimension for Tucker, linearly for POD.**

For $d$ dimensions of size $N$ with rank $r$:

- **POD**: $\mathcal{O}(N^{d-1} \cdot r)$
- **Tucker**: $\mathcal{O}(d \cdot N \cdot r + r^d)$

As $d$ increases, Tucker becomes exponentially better!

---

## Conclusion

### Main Takeaways

1. **Don't reduce only one dimension** — you forfeit most compression potential

2. **Choose strategy based on**:
   - Dimension count (POD for ≤4D, Tucker for ≥5D)
   - Intrusive vs non-intrusive requirement
   - Computational resources
   - Problem structure (separability)

3. **For 6D problems**:
   - **Best practice**: Hybrid approach (flatten spatial, Tucker on rest)
   - **Balances**: Compression, complexity, and capabilities

4. **Tensor decomposition** exploits structure in each dimension independently
   - Exponentially better compression for high dimensions
   - Critical for problems with ≥5 dimensions

5. **Anisotropic ranks** allow dimension-specific resolution
   - High rank in complex directions (boundary layers, high gradients)
   - Low rank in homogeneous directions (periodic, symmetric)

### Recommended Reading

- **POD Theory**: Holmes et al., "Turbulence, Coherent Structures, Dynamical Systems and Symmetry"
- **Tensor Decomposition**: Kolda & Bader, "Tensor Decompositions and Applications," SIAM Review (2009)
- **Tucker Decomposition**: De Lathauwer et al., "A Multilinear Singular Value Decomposition," SIMAX (2000)
- **Reduced Basis Methods**: Quarteroni et al., "Reduced Basis Methods for Partial Differential Equations"
- **Tensor Methods for PDEs**: Grasedyck et al., "A Literature Survey of Low-Rank Tensor Approximation Techniques"

---

*This document provides comprehensive guidance on dimensional reduction strategies for high-dimensional reduced order modeling. Choose the appropriate strategy based on your specific problem characteristics and requirements.*
