using CUDA
using ..utils: vec_to_ltri

# ==============================================================================

@doc raw"""
    rrule(::typeof(vec_to_ltri), diag::CUDA.CuVector, lower::CUDA.CuVector)

This function defines the reverse mode rule (rrule) for the `vec_to_ltri`
function. The `vec_to_ltri` function converts a diagonal vector and a lower
triangular vector into a lower triangular matrix. The `rrule` function computes
the gradients of the inputs `diag` and `lower` with respect to the output lower
triangular matrix.

# Arguments
- `diag::CUDA.CuVector`: The diagonal vector.
- `lower::CUDA.CuVector`: The lower triangular vector.

# Returns
- `ltri`: The lower triangular matrix computed by `vec_to_ltri`.
- `vec_to_ltri_pullback`: The pullback function that computes the gradients of
  `diag` and `lower`.
"""
function ChainRulesCore.rrule(
    ::typeof(vec_to_ltri), diag::CUDA.CuVector, lower::CUDA.CuVector
)
    # Compute the lower triangular matrix
    ltri = vec_to_ltri(diag, lower)

    # Define the pullback function
    function vec_to_ltri_pullback(ΔLtri)
        # Define the dimensionality of the matrix based on the length of the
        # diagonal vector
        n = length(diag)

        # Initialize the gradients for 'diag' and 'lower' on the GPU
        Δdiag = CUDA.zeros(eltype(ΔLtri), n)
        Δlower = CUDA.zeros(eltype(ΔLtri), length(lower))

        # Define the CUDA kernel function for computing the gradients
        function kernel!(Δdiag, Δlower, ΔLtri, n)
            # Calculate the index for each thread
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x

            # Check if the thread index is within the matrix dimensions
            if i <= n
                # Compute the gradient for the diagonal elements
                Δdiag[i] = ΔLtri[i, i]
                # Compute the gradient for the lower triangular elements
                for j in 1:(i-1)
                    lower_index = (i - 1) * (i - 2) ÷ 2 + j
                    Δlower[lower_index] = ΔLtri[i, j]
                end
            end

            return nothing
        end

        # Define the size of the blocks and the grid for the CUDA kernel launch
        blocksize = 256
        gridsize = cld(n, blocksize)

        # Launch the CUDA kernel to compute the gradients
        CUDA.@cuda threads = blocksize blocks = gridsize kernel!(Δdiag, Δlower, ΔLtri, n)

        # Return the gradients for 'diag' and 'lower'
        return (ΔLtri, Δdiag, Δlower)
    end

    # Return the lower triangular matrix and the pullback function
    return ltri, vec_to_ltri_pullback
end

# ------------------------------------------------------------------------------

@doc raw"""
    rrule(::typeof(vec_to_ltri), diag::CUDA.CuMatrix, lower::CUDA.CuMatrix)

This function defines the reverse mode rule (rrule) for the `vec_to_ltri`
function. The `vec_to_ltri` function converts a diagonal matrix and a lower
triangular matrix into a lower triangular matrix. The `rrule` function computes
the gradients of the inputs `diag` and `lower` with respect to the output lower
triangular matrix.

# Arguments
- `diag::CUDA.CuMatrix`: The diagonal matrix.
- `lower::CUDA.CuMatrix`: The lower triangular matrix.

# Returns
- `ltri`: The lower triangular matrix computed by `vec_to_ltri`.
- `vec_to_ltri_pullback`: The pullback function that computes the gradients of
  `diag` and `lower`.
"""
function ChainRulesCore.rrule(
    ::typeof(vec_to_ltri), diag::CUDA.CuMatrix, lower::CUDA.CuMatrix
)
    # Compute the lower triangular matrix
    ltri = vec_to_ltri(diag, lower)

    # Define the pullback function
    function vec_to_ltri_pullback(ΔLtri)
        # Extract matrix dimensions and number of samples
        n, _, cols = size(ΔLtri)

        # Initialize the gradients for 'diag' and 'lower' on the GPU
        Δdiag = CUDA.zeros(eltype(ΔLtri), size(diag))
        Δlower = CUDA.zeros(eltype(ΔLtri), size(lower))

        # Define the CUDA kernel function for computing the gradients
        function kernel!(Δdiag, Δlower, ΔLtri, n, cols)
            # Calculate the index for each thread
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            k = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

            # Check if the thread index is within the matrix dimensions
            if i <= n && k <= cols
                # Compute the gradient for the diagonal elements
                Δdiag[i, k] = ΔLtri[i, i, k]
                # Compute the gradient for the lower triangular elements
                for j in 1:(i-1)
                    lower_index = (i - 1) * (i - 2) ÷ 2 + j + (k - 1) * (n * (n - 1) ÷ 2)
                    Δlower[lower_index] = ΔLtri[i, j, k]
                end
            end

            return nothing
        end

        # Define the size of the blocks and the grid for the CUDA kernel launch
        blocksize = (16, 16)
        gridsize = (cld(n, blocksize[1]), cld(cols, blocksize[2]))

        # Launch the CUDA kernel to compute the gradients
        CUDA.@cuda threads = blocksize blocks = gridsize kernel!(Δdiag, Δlower, ΔLtri, n, cols)

        # Return the gradients for 'diag' and 'lower'
        return (ΔLtri, Δdiag, Δlower)
    end

    # Return the lower triangular matrix and the pullback function
    return ltri, vec_to_ltri_pullback
end