"""
POD ROM comparison on 1D heat equation u_t = μ u_xx, x in (0,1), u(0,t)=u(1,t)=0
IC: u(x,0) = sin(pi x) + 0.5 sin(3 pi x)

Two ROMs on the same training snapshots:
  (A) Intrusive POD–Galerkin (space-only POD, Galerkin project Laplacian)
  (B) Non-intrusive HOSVD/Tucker (space–time–parameter tensor + μ-linear interpolation)

Outputs: relative L2 error vs analytic truth at μ*, and runtime for each method.
"""

import time
import os
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# -----------------------------
# GPU Configuration
# -----------------------------
gpu_idx = 0  # Set GPU device index here (0, 1, 2, etc.)

# --- JAX or NumPy backend selection ---
import numpy as np   # used only for I/O/printing safety
import jax
import jax.numpy as jnp
from jax import config
from jax import jit
import tensorly as tl
from tensorly.decomposition import tucker, parafac
BACKEND = "JAX"
config.update("jax_enable_x64", True)  # Enable 64-bit precision
tl.set_backend('jax')  # Set TensorLy backend to JAX for consistency

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


# -----------------------------
# Problem setup
# -----------------------------
Nx = 101       # spatial grid points (including boundaries)
Nt = 200       # time steps
T  = 0.5
pi = jnp.pi

x = jnp.linspace(0.0, 1.0, Nx)
t = jnp.linspace(0.0, T, Nt)

# Training μ's and test μ*
mu_train = jnp.linspace(0.05, 0.25, 6)  # 6 training values
mu_star  = 0.18                        # test parameter (held out)

# Analytic ground truth:
# u(x,t;μ) = sin(πx) e^{-μ π^2 t} + 0.5 sin(3πx) e^{-μ (3π)^2 t}
def u_analytic(x, t, mu):
    term1 = jnp.sin(pi * x)[..., None] * jnp.exp(-mu * (pi**2)      * t)[None, ...]
    term2 = 0.5 * jnp.sin(3.0*pi * x)[..., None] * jnp.exp(-mu * (3.0*pi)**2 * t)[None, ...]
    return term1 + term2  # (Nx, Nt)

def build_snapshots(mu_vals):
    cols = []
    for mu in list(np.array(mu_vals, dtype=float)):
        cols.append(u_analytic(x, t, mu))  # (Nx, Nt)
    U = jnp.stack(cols, axis=2)             # (Nx, Nt, Nμ)
    return U

U_train = build_snapshots(mu_train)        # (Nx, Nt, Nμ)
U_star_true = u_analytic(x, t, mu_star)    # (Nx, Nt)

# -----------------------------
# Helpers
# -----------------------------
def rel_l2(u_hat, u_true):
    num = jnp.linalg.norm((u_hat - u_true).reshape(-1))
    den = jnp.linalg.norm(u_true.reshape(-1)) + 1e-16
    return float(num / den)

def to_interior(u):   # remove boundary DOFs (Dirichlet 0)
    return u[1:-1]

def from_interior(v): # pad zeros back at boundaries
    return jnp.concatenate([jnp.array([0.0]), v, jnp.array([0.0])])

# -----------------------------
# Intrusive POD–Galerkin
# -----------------------------
def compute_pod_modes_space(U, r):
    # U: (Nx, Nt, Nμ)  -> snapshot matrix X = (Nx) x (Nt*Nμ)
    X = U.reshape(Nx, -1).astype(jnp.float64)
    # economy SVD
    Uu, Ss, Vt = jnp.linalg.svd(X, full_matrices=False)
    Phi = Uu[:, :r].astype(jnp.float64)     # spatial modes
    energy = (Ss**2) / (Ss**2).sum()
    cum_energy = jnp.cumsum(energy)
    return Phi, energy, cum_energy

def build_laplacian_dirichlet(Nx, dx):
    # second-order FD on interior points
    n = Nx - 2
    main = -2.0 * jnp.ones((n,))
    off  =  1.0 * jnp.ones((n-1,))
    L = jnp.diag(main) + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    L = L / (dx*dx)
    return L  # (n, n)

@jit
def step_reduced(a, mu, Lr, dt):
    # Implicit Euler: (I - dt*mu*Lr) a_{n+1} = a_n
    # Solve: A_sys a_{n+1} = a_n where A_sys = I - dt*mu*Lr
    I = jnp.eye(Lr.shape[0])
    A_sys = I - dt * mu * Lr
    a_new = jnp.linalg.solve(A_sys, a)
    return a_new

def intrusive_rom(U, r, mu_eval, t_grid):
    # U: snapshot matrix X = (Nx) x (Nt*Nμ)
    # r: rank of POD modes
    # mu_eval: evaluation parameter
    # t_grid: time grid

    # Time step
    dt = float(t_grid[1] - t_grid[0])

    # POD modes (space only) on all training snapshots
    Phi_full, energy, cum_energy = compute_pod_modes_space(U, r)
    Phi_int = Phi_full[1:-1, :]       # restrict to interior (Nx-2, r)

    # Reduced operator Lr = Phi^T L Phi
    dx = float(x[1] - x[0])
    L = build_laplacian_dirichlet(Nx, dx)  # (Nx-2, Nx-2)
    Lr = Phi_int.T @ (L @ Phi_int)         # (r, r)

    # Initial condition (independent of μ): project to reduced coords
    u0 = jnp.sin(pi * x) + 0.5 * jnp.sin(3.0*pi * x) # (Nx,)
    a = Phi_int.T @ to_interior(u0)        # (r,)

    # JIT warmup if using JAX
    _ = step_reduced(a, mu_eval, Lr, dt)

    # Time-stepping for coefficients, then reconstruct
    Nt_local = t_grid.shape[0]
    A_hist = []
    for k in range(Nt_local):
        A_hist.append(a)
        if k < Nt_local - 1:
            a = step_reduced(a, mu_eval, Lr, dt)
    A_hist = jnp.stack(A_hist, axis=1)      # (r, Nt)

    # Reconstruction on full grid
    U_rom = jnp.stack([from_interior(Phi_int @ A_hist[:, k]) for k in range(Nt_local)], axis=1)  # (Nx, Nt)
    return U_rom, Phi_full, energy, cum_energy

# -----------------------------
# Non-intrusive HOSVD/Tucker
# -----------------------------
def tucker_hosvd(U, ranks: Tuple[int,int,int]):
    rx, rt, rmu = ranks

    # Mode-1 unfolding: (Nx, Nt*Nμ)
    U1 = U.reshape(U.shape[0], -1).astype(jnp.float64)
    Uu1, _, _ = jnp.linalg.svd(U1, full_matrices=False)
    Phi_x = Uu1[:, :rx].astype(jnp.float64)

    # Mode-2 unfolding: (Nt, Nx*Nμ)
    U2 = jnp.transpose(U, (1,0,2)).reshape(U.shape[1], -1).astype(jnp.float64)
    Uu2, _, _ = jnp.linalg.svd(U2, full_matrices=False)
    Phi_t = Uu2[:, :rt].astype(jnp.float64)

    # Mode-3 unfolding: (Nμ, Nx*Nt)
    U3 = jnp.transpose(U, (2,0,1)).reshape(U.shape[2], -1).astype(jnp.float64)
    Uu3, _, _ = jnp.linalg.svd(U3, full_matrices=False)
    Phi_mu = Uu3[:, :rmu].astype(jnp.float64)

    # Core tensor: G = U ×1 Φ_x^T ×2 Φ_t^T ×3 Φ_μ^T
    G = U
    # mode-1 product
    G = jnp.tensordot(Phi_x.T, G, axes=(1, 0))     # (rx, Nt, Nμ)
    # mode-2 product
    G = jnp.tensordot(Phi_t.T, G, axes=(1, 1))     # (rt, rx, Nμ)
    G = jnp.transpose(G, (1,0,2))                  # (rx, rt, Nμ)
    # mode-3 product
    G = jnp.tensordot(Phi_mu.T, G, axes=(1, 2))    # (rμ, rx, rt)
    G = jnp.transpose(G, (1,2,0))                  # (rx, rt, rμ)
    return Phi_x, Phi_t, Phi_mu, G

def linear_interp_mu_row(Phi_mu, mu_grid, mu_eval):
    # Phi_mu: (Nμ, rμ) defined on sampled μ_grid; return interpolated row φ_μ(μ_eval) ∈ R^{rμ}
    mu_grid = np.array(mu_grid, dtype=float)
    mu_eval = float(np.clip(mu_eval, mu_grid.min(), mu_grid.max()))
    # find interval
    idx_right = int(np.argmax(mu_grid >= mu_eval))
    idx_right = max(1, idx_right)
    idx_left  = idx_right - 1
    muL, muR  = mu_grid[idx_left], mu_grid[idx_right]
    w = 0.0 if abs(muR - muL) < 1e-16 else (mu_eval - muL) / (muR - muL)
    return (1.0 - w) * Phi_mu[idx_left, :] + w * Phi_mu[idx_right, :]

def reconstruct_tucker_at_mu(Phi_x, Phi_t, Phi_mu, G, mu_grid, mu_eval):
    phi_mu_eval = linear_interp_mu_row(Phi_mu, mu_grid, mu_eval)   # (rμ,)
    H = jnp.tensordot(G, phi_mu_eval, axes=(2, 0))                  # (rx, rt)
    U_hat = Phi_x @ H @ Phi_t.T                                    # (Nx, Nt)
    return U_hat

# -----------------------------
# TensorLy-based Tucker Decomposition
# -----------------------------
def tucker_hosvd_tensorly(U, ranks: Tuple[int,int,int]):
    """
    Tucker decomposition using TensorLy library.

    Parameters:
    -----------
    U : array (Nx, Nt, Nμ)
        Snapshot tensor
    ranks : tuple (rx, rt, rμ)
        Target ranks for each mode

    Returns:
    --------
    Phi_x : array (Nx, rx)
        Spatial mode basis
    Phi_t : array (Nt, rt)
        Temporal mode basis
    Phi_mu : array (Nμ, rμ)
        Parametric mode basis
    G : array (rx, rt, rμ)
        Core tensor
    """
    rx, rt, rmu = ranks

    # Ensure tensor is in float64 for numerical stability
    U_tensor = jnp.array(U, dtype=jnp.float64)

    # Perform Tucker decomposition using TensorLy
    # tucker() returns (core_tensor, [factor_matrices])
    core, factors = tucker(U_tensor, rank=ranks)

    # Extract factor matrices (mode bases)
    Phi_x = jnp.array(factors[0], dtype=jnp.float64)   # (Nx, rx)
    Phi_t = jnp.array(factors[1], dtype=jnp.float64)   # (Nt, rt)
    Phi_mu = jnp.array(factors[2], dtype=jnp.float64)  # (Nμ, rμ)
    G = jnp.array(core, dtype=jnp.float64)             # (rx, rt, rμ)

    return Phi_x, Phi_t, Phi_mu, G

# -----------------------------
# CP (CANDECOMP/PARAFAC) Decomposition
# -----------------------------
def cp_decomposition(U, rank: int):
    """
    CP (CANDECOMP/PARAFAC) decomposition using TensorLy library.

    CP decomposes tensor as sum of rank-1 tensors:
    U ≈ Σᵣ λᵣ (aᵣ ⊗ bᵣ ⊗ cᵣ)

    Parameters:
    -----------
    U : array (Nx, Nt, Nμ)
        Snapshot tensor
    rank : int
        Number of components (CP rank)

    Returns:
    --------
    weights : array (rank,)
        Component weights λ
    factors : list of 3 arrays
        [A (Nx, rank), B (Nt, rank), C (Nμ, rank)]
        Factor matrices for each mode
    """
    # Ensure tensor is in float64 for numerical stability
    U_tensor = jnp.array(U, dtype=jnp.float64)

    # Perform CP decomposition using TensorLy
    # parafac() returns (weights, [factor_matrices])
    cp_tensor = parafac(U_tensor, rank=rank, init='svd', n_iter_max=100)

    # Extract weights and factors
    weights = jnp.array(cp_tensor.weights, dtype=jnp.float64)  # (rank,)
    factors = [jnp.array(factor, dtype=jnp.float64) for factor in cp_tensor.factors]

    return weights, factors

def reconstruct_cp_at_mu(weights, factors, mu_grid, mu_eval):
    """
    Reconstruct solution at new parameter μ* using CP decomposition.

    Strategy: Interpolate in parametric factor space, then reconstruct.

    Parameters:
    -----------
    weights : array (rank,)
        Component weights
    factors : list [A (Nx, rank), B (Nt, rank), C (Nμ, rank)]
        CP factor matrices
    mu_grid : array (Nμ,)
        Training parameter grid
    mu_eval : float
        Target parameter value

    Returns:
    --------
    U_hat : array (Nx, Nt)
        Reconstructed solution at μ*
    """
    A, B, C = factors  # (Nx, rank), (Nt, rank), (Nμ, rank)
    rank = weights.shape[0]

    # Interpolate each component's parametric factor
    mu_grid_np = np.array(mu_grid, dtype=float)
    mu_eval_np = float(np.clip(mu_eval, mu_grid_np.min(), mu_grid_np.max()))

    # Find interpolation interval
    idx_right = int(np.argmax(mu_grid_np >= mu_eval_np))
    idx_right = max(1, idx_right)
    idx_left = idx_right - 1
    muL, muR = mu_grid_np[idx_left], mu_grid_np[idx_right]
    w = 0.0 if abs(muR - muL) < 1e-16 else (mu_eval_np - muL) / (muR - muL)

    # Interpolate parametric factors
    C_interp = (1.0 - w) * C[idx_left, :] + w * C[idx_right, :]  # (rank,)

    # Reconstruct: U ≈ Σᵣ λᵣ (aᵣ ⊗ bᵣ) cᵣ(μ*)
    # For each component r: λᵣ * (A[:, r] ⊗ B[:, r]) * C_interp[r]
    U_hat = jnp.zeros((A.shape[0], B.shape[0]))
    for r in range(rank):
        # Outer product: A[:, r][:, None] @ B[:, r][None, :] → (Nx, Nt)
        rank1_matrix = jnp.outer(A[:, r], B[:, r])
        U_hat += weights[r] * C_interp[r] * rank1_matrix

    return U_hat

# -----------------------------
# Animation Function
# -----------------------------
def create_animation(x_grid, t_grid, U_true, U_intrusive, U_hosvd, U_cp, mu_star, save_path="pod_comparison.gif"):
    """
    Create animated comparison of four ROM methods showing u(x) evolution over time.

    Parameters:
    -----------
    x_grid : array (Nx,)
        Spatial grid points
    t_grid : array (Nt,)
        Time grid points
    U_true : array (Nx, Nt)
        Exact solution
    U_intrusive : array (Nx, Nt)
        Intrusive POD-Galerkin solution
    U_hosvd : array (Nx, Nt)
        Non-intrusive HOSVD solution
    U_cp : array (Nx, Nt)
        Non-intrusive CP decomposition solution
    mu_star : float
        Parameter value
    save_path : str
        Output file path for animation (default: GIF format)
    """
    # Convert JAX arrays to NumPy for matplotlib compatibility
    x_np = np.array(x_grid)
    t_np = np.array(t_grid)
    U_true_np = np.array(U_true)
    U_intrusive_np = np.array(U_intrusive)
    U_hosvd_np = np.array(U_hosvd)
    U_cp_np = np.array(U_cp)

    # Determine global y-axis limits for consistent scaling
    u_min = min(U_true_np.min(), U_intrusive_np.min(), U_hosvd_np.min(), U_cp_np.min())
    u_max = max(U_true_np.max(), U_intrusive_np.max(), U_hosvd_np.max(), U_cp_np.max())
    margin = 0.1 * (u_max - u_min)
    ylim = [u_min - margin, u_max + margin]

    # Create figure with four subplots (2x2 grid)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.30, top=0.94, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Initialize line objects
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='Exact')
    line2, = ax2.plot([], [], 'r-', linewidth=2, label='POD-Galerkin')
    line3, = ax3.plot([], [], 'g-', linewidth=2, label='HOSVD/Tucker')
    line4, = ax4.plot([], [], 'm-', linewidth=2, label='CP/PARAFAC')

    # Set up subplot properties
    for ax, title in zip([ax1, ax2, ax3, ax4],
                         ['Exact Solution', 'Intrusive POD-Galerkin',
                          'Non-intrusive HOSVD/Tucker', 'Non-intrusive CP/PARAFAC']):
        ax.set_xlim([x_np[0], x_np[-1]])
        ax.set_ylim(ylim)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

    # Add text annotations with proper spacing to avoid overlap
    time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=14, fontweight='bold')
    param_text = fig.text(0.5, 0.04, f'μ = {mu_star:.3f}', ha='center', fontsize=12)

    def init():
        """Initialize animation."""
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, line4, time_text

    def update(frame):
        """Update function for each animation frame."""
        # Update line data for current time step
        line1.set_data(x_np, U_true_np[:, frame])
        line2.set_data(x_np, U_intrusive_np[:, frame])
        line3.set_data(x_np, U_hosvd_np[:, frame])
        line4.set_data(x_np, U_cp_np[:, frame])

        # Update time display
        time_text.set_text(f't = {t_np[frame]:.4f} s')

        return line1, line2, line3, line4, time_text

    # Create animation
    Nt = len(t_np)
    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=Nt, interval=50, blit=True, repeat=True
    )

    # Optimized save workflow: try ffmpeg first, fallback to GIF
    print(f"\nGenerating animation with {Nt} frames...")

    # Determine file extension and appropriate writer
    if save_path.endswith('.mp4'):
        try:
            import matplotlib
            matplotlib.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
            anim.save(save_path, writer='ffmpeg', fps=20, dpi=100, bitrate=1800)
            print(f"✓ Animation saved as MP4: {save_path}")
            plt.close(fig)
            return anim
        except Exception:
            # Fall back to GIF
            save_path = save_path.replace('.mp4', '.gif')

    # Save as GIF (default or fallback)
    try:
        anim.save(save_path, writer='pillow', fps=10, dpi=80)
        print(f"✓ Animation saved as GIF: {save_path}")
    except Exception as e:
        print(f"✗ Failed to save animation: {e}")

    plt.close(fig)
    return anim

# -----------------------------
# Run both methods and compare
# -----------------------------
if __name__ == "__main__":
    print(f"Backend: {BACKEND}")
    if BACKEND == "JAX":
        print(f"GPU Index: {gpu_idx} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        try:
            devices = jax.devices()
            print(f"JAX devices: {devices}")
        except:
            pass

    # Parameters
    r_pod     = 6            # spatial rank for intrusive POD
    r_tucker  = (8, 8, 4)    # (rx, rt, rμ) ranks for HOSVD/Tucker
    r_cp      = 12           # CP rank (number of components)

    # Intrusive ROM
    t0 = time.perf_counter()
    U_intrusive, Phi_full, energy, cum_energy = intrusive_rom(U_train, r_pod, mu_star, t)
    t1 = time.perf_counter()
    err_intrusive = rel_l2(U_intrusive, U_star_true)
    time_intrusive = t1 - t0

    # Non-intrusive HOSVD (Manual Implementation)
    t2 = time.perf_counter()
    Phi_x, Phi_t, Phi_mu, G = tucker_hosvd(U_train, r_tucker)
    U_hosvd = reconstruct_tucker_at_mu(Phi_x, Phi_t, Phi_mu, G, mu_train, mu_star)
    t3 = time.perf_counter()
    err_hosvd = rel_l2(U_hosvd, U_star_true)
    time_hosvd = t3 - t2

    # Non-intrusive HOSVD (TensorLy Implementation)
    t4 = time.perf_counter()
    Phi_x_tl, Phi_t_tl, Phi_mu_tl, G_tl = tucker_hosvd_tensorly(U_train, r_tucker)
    U_hosvd_tl = reconstruct_tucker_at_mu(Phi_x_tl, Phi_t_tl, Phi_mu_tl, G_tl, mu_train, mu_star)
    t5 = time.perf_counter()
    err_hosvd_tl = rel_l2(U_hosvd_tl, U_star_true)
    time_hosvd_tl = t5 - t4

    # Comparison between manual and TensorLy implementations
    diff_reconstruction = rel_l2(U_hosvd, U_hosvd_tl)
    diff_core = jnp.linalg.norm((G - G_tl).reshape(-1)) / (jnp.linalg.norm(G.reshape(-1)) + 1e-16)

    # CP Decomposition
    t6 = time.perf_counter()
    cp_weights, cp_factors = cp_decomposition(U_train, r_cp)
    U_cp = reconstruct_cp_at_mu(cp_weights, cp_factors, mu_train, mu_star)
    t7 = time.perf_counter()
    err_cp = rel_l2(U_cp, U_star_true)
    time_cp = t7 - t6

    # Report
    explained = float(100.0 * np.array(cum_energy)[min(r_pod, len(cum_energy))-1])
    print("\n=== POD ROM Comparison on 1D Heat Equation ===")
    print(f"Spatial POD cumulative energy at r={r_pod}: {explained:.2f}%")
    print(f"\n[Intrusive POD–Galerkin]")
    print(f"  Rel L2 Error: {err_intrusive:.3e} | Runtime: {time_intrusive:.4f} s")
    print(f"\n[Non-intrusive HOSVD - Manual Implementation]")
    print(f"  Rel L2 Error: {err_hosvd:.3e} | Runtime: {time_hosvd:.4f} s")
    print(f"\n[Non-intrusive HOSVD - TensorLy Implementation]")
    print(f"  Rel L2 Error: {err_hosvd_tl:.3e} | Runtime: {time_hosvd_tl:.4f} s")
    print(f"\n[Manual vs TensorLy Comparison]")
    print(f"  Reconstruction difference: {diff_reconstruction:.3e}")
    print(f"  Core tensor difference: {diff_core:.3e}")
    print(f"\n[CP Decomposition (TensorLy)]")
    print(f"  CP Rank: {r_cp}")
    print(f"  Rel L2 Error: {err_cp:.3e} | Runtime: {time_cp:.4f} s")

    # Get script directory for saving outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Optional: save arrays for inspection in pod/ directory
    try:
        np.save(os.path.join(script_dir, "U_star_true.npy"),  np.asarray(U_star_true))
        np.save(os.path.join(script_dir, "U_intrusive.npy"),  np.asarray(U_intrusive))
        np.save(os.path.join(script_dir, "U_hosvd.npy"),      np.asarray(U_hosvd))
        print(f"\nSaved arrays to: {script_dir}/")
    except Exception as e:
        print(f"\nSkipped saving arrays: {e}")

    # Generate animation in pod/ directory
    print("\n" + "="*60)
    output_path = os.path.join(script_dir, "pod_comparison.gif")
    create_animation(x, t, U_star_true, U_intrusive, U_hosvd, U_cp, mu_star, save_path=output_path)
    print("="*60)
