# Gpu_3d_Block_Recursive.py - Documentation

## Overview
This code file implements a recursive algorithm to solve for the diagonal elements of the Non-Equilibrium Green's Function (NEGF). It is designed to handle matrices by 3 diagonals. The primary outputs are various Green's functions: Retarded (Grl, Grd, Gru), Electron (Gnl, Gnd, Gnu), and Hole (Gpl, Gpd, Gpu - though these are not present in the final kernel's signature). The calculations are performed on a GPU using Numba and CUDA.

## Key Components

### CUDA Kernel(s)
- **`Gpu_3d_Block_Recursive`**:
  - *Description*: This is the main CUDA kernel that implements the recursive algorithm to compute Green's functions. It takes coefficient matrices (AD, ALD, AUD) and self-energies (Sigin, Sigout) as inputs and calculates the retarded (Grl, Grd, Gru) and electron (Gnl, Gnd, Gnu) Green's functions. It also uses several helper device functions for matrix operations.
  - *Parameters*:
    - `AD (complex64[:,:,:,:])`: Matrix of coefficients (diagonal elements).
    - `ALD (complex64[:,:,:,:])`: Matrix of coefficients (lower diagonal elements).
    - `AUD (complex64[:,:,:,:])`: Matrix of coefficients (upper diagonal elements).
    - `Sigin (complex64[:,:,:,:])`: Matrix of in-scattering self-energies (diagonal).
    - `Sigout (complex64[:,:,:,:])`: Matrix of out-scattering self-energies (diagonal).
    - `gaL (complex64[:,:,:,:])`: Output array for advanced Green's function (left-connected).
    - `grL (complex64[:,:,:,:])`: Output array for retarded Green's function (left-connected).
    - `ginL (complex64[:,:,:,:])`: Output array for electron in-scattering Green's function (left-connected).
    - `gipL (complex64[:,:,:,:])`: Output array for hole in-scattering Green's function (left-connected). (Note: mentioned in comments, but not directly used in the same way as others in the kernel body for final output G's).
    - `Grl (complex64[:,:,:,:])`: Output array for Retarded Green's Function (lower diagonal).
    - `Grd (complex64[:,:,:,:])`: Output array for Retarded Green's Function (diagonal).
    - `Gru (complex64[:,:,:,:])`: Output array for Retarded Green's Function (upper diagonal).
    - `Gnd (complex64[:,:,:,:])`: Output array for Electron Green's Function (diagonal).
    - `Gnu (complex64[:,:,:,:])`: Output array for Electron Green's Function (upper diagonal).
    - `Gnl (complex64[:,:,:,:])`: Output array for Electron Green's Function (lower diagonal).
    - `Al_cr (complex64[:,:,:,:])`: Helper array for Hermitian conjugate of AUD.
    - `Ad_cr (complex64[:,:,:,:])`: Helper array for Hermitian conjugate of AD.
    - `Au_cr (complex64[:,:,:,:])`: Helper array for Hermitian conjugate of ALD.
    - `Gal (complex64[:,:,:,:])`: Output array for advanced Green's function (lower diagonal).
    - `Gad (complex64[:,:,:,:])`: Output array for advanced Green's function (diagonal).
    - `Gau (complex64[:,:,:,:])`: Output array for advanced Green's function (upper diagonal).
  - *Block/Grid Dimensions*: The kernel uses `i = cuda.grid(1)`, suggesting it is launched with a 1D grid of thread blocks. The size of the grid typically corresponds to `Ne`, the number of energy points or independent calculations.

### CUDA Device Functions
- **`inverse_matrix(mat, B, A)`**:
  - *Description*: Computes the inverse of a matrix `mat` using Gaussian elimination with partial pivoting and stores it in `B`. `A` is used as an augmented matrix internally.
  - *Parameters*:
    - `mat (complex64[:,:])`: Input matrix to be inverted.
    - `B (complex64[:,:])`: Output matrix to store the inverse.
    - `A (complex64[:,:])`: Internal temporary augmented matrix.
  - *Returns*: `complex64[:,:]` (reference to `B`).
- **`mul_mat(A, B, C)`**:
  - *Description*: Performs matrix multiplication of `A` and `B` (`A @ B`) and stores the result in `C`.
  - *Parameters*:
    - `A (complex64[:,:])`: First input matrix.
    - `B (complex64[:,:])`: Second input matrix.
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: `complex64[:,:]` (reference to `C`).
- **`add_num(a, b, c)`**:
  - *Description*: Adds two complex scalars `a` and `b` and stores the result in `c`. (Docstring incorrectly states matrix multiplication).
  - *Parameters*:
    - `a (complex64)`: First scalar.
    - `b (complex64)`: Second scalar.
    - `c (complex64)`: Output scalar for the sum.
  - *Returns*: `complex64` (the sum).
- **`sub_mat(A, B, C)`**:
  - *Description*: Subtracts matrix `B` from matrix `A` (`A - B`) and stores the result in `C`.
  - *Parameters*:
    - `A (complex64[:,:])`: First input matrix.
    - `B (complex64[:,:])`: Second input matrix (to be subtracted).
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: `complex64[:,:]` (reference to `C`).
- **`add_mat(A, B, C)`**:
  - *Description*: Adds matrix `A` and matrix `B` (`A + B`) and stores the result in `C`.
  - *Parameters*:
    - `A (complex64[:,:])`: First input matrix.
    - `B (complex64[:,:])`: Second input matrix.
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: (void, modifies `C` in-place).
- **`set_mat(B, A)`**:
  - *Description*: Copies the elements from matrix `A` into matrix `B`.
  - *Parameters*:
    - `B (complex64[:,:])`: Destination matrix.
    - `A (complex64[:,:])`: Source matrix.
  - *Returns*: (void, modifies `B` in-place).
- **`neg_mat(A)`**:
  - *Description*: Negates all elements in matrix `A`.
  - *Parameters*:
    - `A (complex64[:,:])`: Input matrix (modified in-place).
  - *Returns*: (void, modifies `A` in-place).
- **`pw_mul(A, B, C)`**:
  - *Description*: Performs element-wise multiplication of matrices `A` and `B` and stores the result in `C`.
  - *Parameters*:
    - `A (complex64[:,:])`: First input matrix.
    - `B (complex64[:,:])`: Second input matrix.
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: `complex64[:,:]` (reference to `C`).
- **`add_scalar_mat(a, B, C)`**:
  - *Description*: Adds a scalar `a` to all elements of matrix `B` and stores the result in `C`.
  - *Parameters*:
    - `a (complex64)`: Scalar value.
    - `B (complex64[:,:])`: Input matrix.
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: `complex64[:,:]` (reference to `C`).
- **`mul_scalar_mat(a, B, C)`**:
  - *Description*: Multiplies all elements of matrix `B` by a scalar `a` and stores the result in `C`.
  - *Parameters*:
    - `a (complex64)`: Scalar value.
    - `B (complex64[:,:])`: Input matrix.
    - `C (complex64[:,:])`: Output matrix for the result.
  - *Returns*: `complex64[:,:]` (reference to `C`).
- **`abs_err(A, error)`**:
  - *Description*: Calculates the sum of the absolute values (magnitudes) of all complex elements in matrix `A`. The result is stored in the first element of the `error` array.
  - *Parameters*:
    - `A (complex64[:,:])`: Input matrix.
    - `error (float32[:])`: Output array (1-element) to store the sum.
  - *Returns*: `float32[:]` (reference to `error`).
- **`conjugate(num)`**:
  - *Description*: Computes the complex conjugate of a scalar `num`.
  - *Parameters*:
    - `num (complex64)`: Input scalar.
  - *Returns*: `complex64` (the complex conjugate).
- **`hconj_mat(A, B)`**:
  - *Description*: Computes the Hermitian conjugate (conjugate transpose) of matrix `A` and stores it in `B`.
  - *Parameters*:
    - `A (complex64[:,:])`: Input matrix.
    - `B (complex64[:,:])`: Output matrix for the Hermitian conjugate.
  - *Returns*: `complex64[:,:]` (reference to `B`).

### Other Key Functions/Classes (if any)
- N/A: The file primarily consists of the main CUDA kernel and its associated device functions.

## Important Variables/Constants
- **`EPS (float)`**: A small constant `1.0e-19` used in `inverse_matrix` to handle potential division by zero or very small pivot elements, ensuring numerical stability.
- **`Np (int)`**: Represents the size of the matrices (e.g., number of points in a spatial dimension, or layers). Inferred from `1+Grl.shape[0]` in the kernel. The header comments also mention "Np = Size Of The Matrices".
- **`Ne (int)`**: Represents the number of independent calculations, often corresponding to energy points. Inferred from `Grl.shape[1]` in the kernel.
- **Green's Function Matrices**:
    - `Grl, Grd, Gru`: Components of the Retarded Green's Function (lower, diagonal, upper).
    - `Gnl, Gnd, Gnu`: Components of the Electron Green's Function (lower, diagonal, upper).
    - `Gal, Gad, Gau`: Components of the Advanced Green's Function (lower, diagonal, upper).
    - `grL, ginL, gaL`: Left-connected Green's functions used in intermediate calculations.
- **Coefficient Matrices**:
    - `AD, ALD, AUD`: Coefficient matrices representing the system's Hamiltonian or coupling terms (diagonal, lower, upper).
- **Self-Energy Matrices**:
    - `Sigin, Sigout`: In-scattering and out-scattering self-energies, crucial for NEGF calculations.

## Usage Examples
The `Gpu_3d_Block_Recursive` kernel is designed to be launched on a CUDA-capable GPU. Below is a conceptual example of how it might be set up and called.

```python
import numpy as np
from numba import cuda
# Assuming Gpu_3d_Block_Recursive is in the current scope or imported
# from current_module import Gpu_3d_Block_Recursive 

# Define problem parameters (example values)
Np_val = 10 # Number of points/layers
Ne_val = 100 # Number of energy points
mat_size = 4 # Size of the sub-matrices (e.g., 4x4)

# Initialize input matrices on the CPU (typically with numpy)
# These would be populated with actual physical parameters
AD_cpu = np.random.rand(Np_val, Ne_val, mat_size, mat_size).astype(np.complex64)
ALD_cpu = np.random.rand(Np_val, Ne_val, mat_size, mat_size).astype(np.complex64)
AUD_cpu = np.random.rand(Np_val, Ne_val, mat_size, mat_size).astype(np.complex64)
Sigin_cpu = np.random.rand(Np_val, Ne_val, mat_size, mat_size).astype(np.complex64)
Sigout_cpu = np.random.rand(Np_val, Ne_val, mat_size, mat_size).astype(np.complex64) # Though Sigout is not used in the current kernel code

# Allocate memory on the GPU and copy data
AD_gpu = cuda.to_device(AD_cpu)
ALD_gpu = cuda.to_device(ALD_cpu)
AUD_gpu = cuda.to_device(AUD_cpu)
Sigin_gpu = cuda.to_device(Sigin_cpu)
Sigout_gpu = cuda.to_device(Sigout_cpu) # Matching kernel signature

# Allocate output arrays on the GPU
# Note: Np in kernel (Np-1 for array dim) vs Np_val for allocation size
Grl_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)
Grd_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gru_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gnl_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gnd_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gnu_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)

# Allocate intermediate/helper arrays on the GPU
gaL_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
grL_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
ginL_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
gipL_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64) # As per signature
Al_cr_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Ad_cr_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Au_cr_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gal_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gad_gpu = cuda.device_array((Np_val, Ne_val, mat_size, mat_size), dtype=np.complex64)
Gau_gpu = cuda.device_array((Np_val - 1, Ne_val, mat_size, mat_size), dtype=np.complex64)


# Configure kernel launch parameters
threadsperblock = 32 # Example value, needs tuning
blockspergrid = (Ne_val + (threadsperblock - 1)) // threadsperblock

# Launch the kernel
Gpu_3d_Block_Recursive[blockspergrid, threadsperblock](
    AD_gpu, ALD_gpu, AUD_gpu, Sigin_gpu, Sigout_gpu,
    gaL_gpu, grL_gpu, ginL_gpu, gipL_gpu,
    Grl_gpu, Grd_gpu, Gru_gpu,
    Gnd_gpu, Gnu_gpu, Gnl_gpu,
    Al_cr_gpu, Ad_cr_gpu, Au_cr_gpu,
    Gal_gpu, Gad_gpu, Gau_gpu
)
cuda.synchronize()

# Copy results back to CPU if needed
Grd_results_cpu = Grd_gpu.copy_to_host()
# print(Grd_results_cpu)
```

## Dependencies and Interactions
- **Internal Dependencies**:
  - The main kernel `Gpu_3d_Block_Recursive` heavily relies on the various CUDA device functions defined within the same file (e.g., `inverse_matrix`, `mul_mat`, `hconj_mat`, etc.) for its computations.
- **External Libraries**:
  - `numpy`: Used for creating, initializing, and manipulating numerical arrays on the CPU before they are transferred to the GPU. Also used for handling results copied back from the GPU.
  - `numba.cuda`: Essential for the entire GPU computation. It provides:
    - `@cuda.jit` decorator for JIT compiling Python functions into CUDA kernels and device functions.
    - `cuda.to_device()` for transferring data from CPU to GPU.
    - `cuda.device_array()` for allocating memory directly on the GPU.
    - `cuda.local.array()` for creating statically sized arrays local to a thread within a kernel.
    - `cuda.grid(1)` for determining thread index within the kernel.
    - `cuda.synchronize()` for ensuring kernel completion.
- **Interactions**:
  - This file provides a core computational kernel for solving a part of a Non-Equilibrium Green's Function problem.
  - It is expected to be called by a higher-level Python script that sets up the overall physics simulation, prepares the input data (Hamiltonians, self-energies), manages GPU resources, and orchestrates the kernel launch.
  - The output Green's functions (Grd, Grl, Gru, Gnd, Gnl, Gnu) are fundamental quantities in NEGF theory and would be used by subsequent parts of the simulation for calculating physical observables like current, density of states, etc.
```
