# Import ML library
import Flux

# Import library for random sampling
import Distributions
import Random

# Import library for basic math
import LinearAlgebra

using AutoEncoderToolkit.utils

# ==============================================================================
# Functions that extend the methods in the utils.jl file for GPU arrays
# ==============================================================================

@doc raw"""
    vec_to_ltri(diag::AbstractVecOrMat, lower::AbstractVecOrMat)

GPU implementation of `vec_to_ltri`.
"""
function utils._vec_to_ltri(
    ::Type{T}, diag::CUDA.CuVector, lower::CUDA.CuVector
) where {T<:CUDA.CuArray}
    # Define the dimensionality of the matrix based on the length of the
    # diagonal vector
    n = length(diag)

    # Create a zero matrix of the same type as the diagonal vector on the GPU
    matrix = CUDA.zeros(eltype(diag), n, n)

    # Define the CUDA kernel function that will be executed on the GPU
    function kernel!(matrix, diag, lower, n)
        # Calculate the row and column indices based on the thread and block
        # indices
        i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y

        # Check if the indices are within the bounds of the matrix
        if i <= n && j <= n
            # If the row and column indices are equal, set the matrix element to
            # the corresponding element of the diagonal vector
            if i == j
                matrix[i, j] = diag[i]
                # If the row index is greater than the column index, set the
                # matrix element to the corresponding element of the lower
                # vector
            elseif i > j
                lower_index = (i - 1) * (i - 2) ÷ 2 + j
                matrix[i, j] = lower[lower_index]
            end
        end

        # Return nothing as the matrix is modified in-place
        return nothing
    end

    # Define the size of the blocks and the grid for the CUDA kernel launch
    blocksize = (16, 16)
    gridsize = (cld(n, blocksize[1]), cld(n, blocksize[2]))

    # Launch the CUDA kernel with the specified block and grid sizes
    CUDA.@cuda threads = blocksize blocks = gridsize kernel!(
        matrix, diag, lower, n
    )

    # Return the resulting matrix
    return matrix
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(diag::AbstractVecOrMat, lower::AbstractVecOrMat)

GPU implementation of `vec_to_ltri`.
"""
# Define a function to convert a diagonal and lower triangular matrix into a 3D
# tensor on the GPU
function utils._vec_to_ltri(
    ::Type{T}, diag::CUDA.CuMatrix, lower::CUDA.CuMatrix
) where {T<:CUDA.CuArray}
    # Extract the dimensions of the diagonal matrix and the number of samples
    # (columns)
    n, cols = size(diag)

    # Check if the dimensions of the lower triangular matrix match the expected
    # dimensions
    if size(lower) != (n * (n - 1) ÷ 2, cols)
        # If the dimensions do not match, throw an error
        error("Dimension mismatch between 'diag' and 'lower' matrices")
    end

    # Create a 3D tensor of zeros on the GPU with the same type as the diagonal
    # matrix
    tensor = CUDA.zeros(eltype(diag), n, n, cols)

    # Define the CUDA kernel function that will be executed on the GPU
    function kernel!(tensor, diag, lower, n, cols)
        # Calculate the row, column, and depth indices based on the thread and
        # block indices
        i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
        k = (CUDA.blockIdx().z - 1) * CUDA.blockDim().z + CUDA.threadIdx().z

        # Check if the indices are within the bounds of the tensor
        if i <= n && j <= n && k <= cols
            # If the row and column indices are equal, set the tensor element to
            # the corresponding element of the diagonal matrix
            if i == j
                tensor[i, j, k] = diag[i, k]
                # If the row index is greater than the column index, set the
                # tensor element to the corresponding element of the lower
                # triangular matrix
            elseif i > j
                lower_index = (i - 1) * (i - 2) ÷ 2 + j + (k - 1) * (n * (n - 1) ÷ 2)
                tensor[i, j, k] = lower[lower_index]
            end
        end
        # Return nothing as the tensor is modified in-place
        return nothing
    end

    # Define the size of the blocks and the grid for the CUDA kernel launch
    blocksize = (16, 16, 4)
    gridsize = (cld(n, blocksize[1]), cld(n, blocksize[2]), cld(cols, blocksize[3]))

    # Launch the CUDA kernel with the specified block and grid sizes
    CUDA.@cuda threads = blocksize blocks = gridsize kernel!(tensor, diag, lower, n, cols)

    # Return the resulting 3D tensor
    return tensor
end # function

# ==============================================================================

@doc raw"""
    slogdet(A::CUDA.CuArray; check::Bool=false)

GPU AbstractArray implementation of `slogdet`.
"""
function utils._slogdet(
    ::Type{T}, A::AbstractArray{<:Any,3}; check::Bool=false
) where {T<:CUDA.CuArray}
    # Compute the Cholesky decomposition of each slice of A. 
    chol = [
        x.L for x in LinearAlgebra.cholesky.(eachslice(A, dims=3), check=check)
    ]

    # compute the log determinant of each slice of A as the sum of the log of
    # the diagonal elements of the Cholesky decomposition
    logdetA = @. 2 * sum(log, LinearAlgebra.diag(chol))

    # Reupload to GPU since the output of each operation is a single scalar
    # returned to the CPU.
    return logdetA |> Flux.gpu
end # function

# ==============================================================================

"""
    _randn_samples(::Type{T}, z::AbstractArray) where {T<:CUDA.CuArray}

Generates a random sample with the same type and size as `z` using a standard
normal distribution. This function is used for GPU arrays.
"""
function utils._randn_samples(::Type{T}, z::CUDA.CuArray) where {T<:CUDA.CuArray}
    return CUDA.randn(eltype(z), size(z))
end # function

# ==============================================================================

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T}) where {T<:Number}

GPU AbstractMatrix implementation of `sample_MvNormalCanon`.
"""
function utils._sample_MvNormalCanon(
    ::Type{T}, Σ⁻¹::AbstractMatrix
) where {T<:CUDA.CuArray}
    # Invert the precision matrix
    Σ = LinearAlgebra.inv(Σ⁻¹)

    # Cholesky decomposition of the covariance matrix
    chol = LinearAlgebra.cholesky(Σ, check=false)

    # Define sample type
    if !(eltype(Σ⁻¹) <: AbstractFloat)
        N = Float32
    else
        N = eltype(Σ⁻¹)
    end # if

    # Sample from standard normal distribution
    r = CUDA.randn(N, size(Σ⁻¹, 1))

    # Return sample multiplied by the Cholesky decomposition
    return chol.L * r
end # function

# ==============================================================================

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T,3}) where {T<:Number}

GPU AbstractArray implementation of `sample_MvNormalCanon`.
"""
function utils._sample_MvNormalCanon(
    ::Type{T}, Σ⁻¹::CUDA.CuArray{<:Number,3}
) where {T<:CUDA.CuArray}
    # Extract dimensions
    dim = size(Σ⁻¹, 1)
    # Extract number of samples
    n_sample = size(Σ⁻¹, 3)

    # Invert the precision matrix
    Σ = last(CUDA.CUBLAS.matinv_batched(collect(eachslice(Σ⁻¹, dims=3))))

    # Cholesky decomposition of the covariance matrix
    chol = reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            x.L
            for x in LinearAlgebra.cholesky.(Σ, check=false)
        ]
    )

    # Define sample type
    if !(eltype(Σ⁻¹) <: AbstractFloat)
        N = Float32
    else
        N = eltype(Σ⁻¹)
    end # if

    # Sample from standard normal distribution
    r = CUDA.randn(N, dim, n_sample)

    # Return sample multiplied by the Cholesky decomposition
    return Flux.batched_vec(chol, r)
end # function

# ==============================================================================

@doc raw"""
    unit_vectors(x::CUDA.CuVector)

GPU AbstractVector implementation of `unit_vectors`.
"""
function utils._unit_vectors(
    ::Type{T}, x::AbstractVector
) where {T<:CUDA.CuArray}
    return [utils.unit_vector(x, i) for i in 1:length(x)] |> Flux.gpu
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    unit_vectors(x::CUDA.CuMatrix)

GPU AbstractMatrix implementation of `unit_vectors`.
"""
function utils._unit_vectors(
    ::Type{T}, x::CUDA.CuMatrix
) where {T<:CUDA.CuArray}
    vectors = [
        reduce(hcat, fill(utils.unit_vector(x, i), size(x, 2)))
        for i in 1:size(x, 1)
    ]
    return vectors |> Flux.gpu
end # function

# ==============================================================================

@doc raw"""
    finite_difference_gradient(
        f::Function,
        x::CUDA.CuArray;
        fdtype::Symbol=:central
    )

GPU AbstractVecOrMat implementation of `finite_difference_gradient`.
"""
function utils._finite_difference_gradient(
    ::Type{T},
    f::Function,
    x::AbstractVecOrMat;
    fdtype::Symbol=:central,
) where {T<:CUDA.CuArray}
    # Check that mode is either :forward or :central
    if !(fdtype in (:forward, :central))
        error("fdtype must be either :forward or :central")
    end # if

    # Check fdtype
    if fdtype == :forward
        # Define step size
        ε = √(eps(eltype(x)))
        # Generate unit vectors times step size for each element of x
        Δx = utils.unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.([x + δ for δ in Δx]) .- f(x)) ./ ε
    else
        # Define step size
        ε = ∛(eps(eltype(x)))
        # Generate unit vectors times step size for each element of x
        Δx = utils.unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.([x + δ for δ in Δx]) - f.([x - δ for δ in Δx])) ./ (2ε)
    end # if

    if typeof(x) <: AbstractVector
        return CUDA.cu(grad)
    elseif typeof(x) <: AbstractMatrix
        return CUDA.cu(permutedims(reduce(hcat, grad), [2, 1]))
    end # if
end # function