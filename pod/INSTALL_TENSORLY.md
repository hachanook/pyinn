# Installing TensorLy in Your Virtual Environment

This guide explains how to update your existing Python virtual environment to include the TensorLy package for Tucker decomposition.

---

## Option 1: Using pip with requirements.txt (Recommended)

If you're using a standard Python virtual environment (`.venv`):

### Step 1: Activate Your Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### Step 2: Install TensorLy

```bash
pip install tensorly>=0.8.0
```

**Or install all updated requirements:**
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import tensorly as tl; print(f'TensorLy version: {tl.__version__}')"
```

Expected output:
```
TensorLy version: 0.8.x
```

---

## Option 2: Using Conda with environment.yaml

If you're using a Conda environment (recommended for this project):

### Step 1: Activate Your Conda Environment

```bash
conda activate pyinn-env
```

### Step 2: Install TensorLy via pip

Since TensorLy is not available through conda-forge with JAX backend support, install via pip:

```bash
pip install tensorly>=0.8.0
```

### Step 3: Verify Installation

```bash
python -c "import tensorly as tl; print(f'TensorLy version: {tl.__version__}')"
```

---

## Option 3: Recreate Environment from Scratch (If Issues Occur)

If you encounter dependency conflicts, recreate the environment:

### Using pip (venv):

```bash
# Deactivate current environment
deactivate

# Remove old environment
rm -rf .venv  # Linux/macOS
# rmdir /s .venv  # Windows

# Create fresh environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Using Conda:

```bash
# Deactivate current environment
conda deactivate

# Remove old environment
conda env remove -n pyinn-env

# Recreate from environment.yaml
conda env create -f environment.yaml

# Activate
conda activate pyinn-env

# Install TensorLy
pip install tensorly>=0.8.0
```

---

## Verifying TensorLy with JAX Backend

After installation, verify TensorLy works with JAX:

```bash
python -c "
import tensorly as tl
import jax.numpy as jnp

tl.set_backend('jax')
print(f'TensorLy backend: {tl.get_backend()}')

# Test basic operation
tensor = jnp.ones((3, 4, 5))
print(f'Test tensor shape: {tensor.shape}')
print('✓ TensorLy with JAX backend working correctly')
"
```

Expected output:
```
TensorLy backend: jax
Test tensor shape: (3, 4, 5)
✓ TensorLy with JAX backend working correctly
```

---

## Running the Updated Script

Once TensorLy is installed, run the comparison:

```bash
cd pod/
python main_1D_heat.py
```

You should see output comparing three methods:
1. Intrusive POD-Galerkin
2. Non-intrusive HOSVD (Manual Implementation)
3. Non-intrusive HOSVD (TensorLy Implementation)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorly'"

**Solution:** Ensure you've activated the correct virtual environment before installation.

```bash
# Check which Python you're using
which python  # Linux/macOS
# where python  # Windows

# Should point to your virtual environment
```

### Issue: TensorLy JAX backend not working

**Solution:** Ensure JAX is installed correctly:

```bash
pip install --upgrade jax jaxlib
pip install --upgrade tensorly
```

### Issue: Dependency conflicts

**Solution:** Use the "Recreate Environment from Scratch" option above.

---

## Additional Notes

- **TensorLy Version:** We require `>=0.8.0` for the `tucker()` decomposition API
- **JAX Backend:** The script automatically sets TensorLy to use JAX backend for GPU acceleration
- **Compatibility:** TensorLy works seamlessly with JAX arrays and operations

---

## Quick Command Reference

```bash
# Activate environment
conda activate pyinn-env  # OR: source .venv/bin/activate

# Install TensorLy
pip install tensorly>=0.8.0

# Verify installation
python -c "import tensorly; print(tensorly.__version__)"

# Run the comparison script
cd pod/
python main_1D_heat.py
```
