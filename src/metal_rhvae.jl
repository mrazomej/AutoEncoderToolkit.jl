@doc raw"""
    vec_to_ltri(diag::Metal.MtlVector{T}, lower::Metal.MtlVector{T}) where {T<:Number}

Construct a lower triangular matrix from a vector of diagonal elements and a
vector of lower triangular elements.

# Arguments
- `diag::Metal.MtlVector{T}`: A vector of `T` containing the diagonal elements
  of the matrix.
- `lower::Metal.MtlVector{T}`: A vector of `T` containing the elements of the
  lower triangle of the matrix.

# Returns
- A lower triangular matrix of type `T` with the diagonal and lower triangular
  elements populated from `diag` and `lower` respectively.

# Note
The function assumes that the `diag` and `lower` vectors have the correct
lengths for the matrix to be constructed. Specifically, `diag` should have `n`
elements and `lower` should have `n*(n-1)/2` elements, where `n` is the
dimension of the matrix.
"""
function vec_to_ltri(
    diag::Metal.MtlVector{T}, lower::Metal.MtlVector{T}
) where {T<:Number}
    # Calculate latent space dimensionality
    n = length(diag)
    # Initialize matrix of zeros
    L = Metal.zeros(T, n, n)
    # Obtain indices of lower-triangular matrix
    idx = tril_indices(n; offset=-1)
    # Loop through rows
    for (i, row) in enumerate(eachrow(idx))
        # Populate lower-triangular matrix lower elements
        L[row[1], row[2]] = vec(lower)[i]
    end # for
    # Populate lower-triangular matrix diagonal elements
    L[LinearAlgebra.diagind(L)] .= diag

    return L
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        vec_to_ltri(diag::Metal.MtlMatrix{T}, lower::Metal.MtlMatrix{T}) where {T<:Number}

Construct a set of lower triangular matrices from a matrix of diagonal elements
and a matrix of lower triangular elements, each column representing a sample.

# Arguments
- `diag::Metal.MtlMatrix{T}`: A matrix of `T` where each column contains the
  diagonal elements of the matrix for a specific sample.
- `lower::Metal.MtlMatrix{T}`: A matrix of `T` where each column contains the
  elements of the lower triangle of the matrix for a specific sample.

# Returns
- A 3D array of type `T` where each slice along the third dimension is a lower
  triangular matrix with the diagonal and lower triangular elements populated
  from `diag` and `lower` respectively.

# Note
The function assumes that the `diag` and `lower` matrices have the correct
dimensions for the matrices to be constructed. Specifically, `diag` and `lower`
should have `n` rows and `m` columns, where `n` is the dimension of the matrix
and `m` is the number of samples. The `lower` matrix should have `n*(n-1)/2`
non-zero elements per column, corresponding to the lower triangular part of the
matrix.
"""
function vec_to_ltri(
    diag::Metal.MtlMatrix{T}, lower::Metal.MtlMatrix{T}
) where {T<:Number}
    # Calculate latent space dimensionality
    n = size(diag, 1)
    # Calculate the number of samples
    n_samples = size(diag, 2)
    # Initialize matrix of zeros
    L = Metal.zeros(T, n, n, n_samples)
    # Obtain indices of lower-triangular matrix
    idx_low = tril_indices(n, n_samples; offset=-1)
    # Loop through rows
    for (i, row) in enumerate(eachrow(idx_low))
        # Populate lower-triangular matrix lower elements
        L[row[1], row[2], row[3]] = vec(lower)[i]
    end # for
    # Obtain indices of diagonal elements
    idx_diag = diag_indices(n, n_samples)
    # Loop through rows
    for (i, row) in enumerate(eachrow(idx_diag))
        # Populate lower-triangular matrix lower elements
        L[row[1], row[2], row[3]] = vec(diag)[i]
    end # for

    return L
end # function


@doc raw"""
        (m::MetricChain)(x::Metal.MtlArray{Float32}; matrix::Bool=false)

Perform a forward pass through the MetricChain.

# Arguments
- `x::Metal.MtlArray{Float32}`: The input data to be processed. This should be a
  Float32 array.
- `matrix::Bool=false`: A boolean flag indicating whether to return the result
  as a lower triangular matrix (if `true`) or as a tuple of diagonal and lower
  off-diagonal elements (if `false`). Defaults to `false`.

# Returns
- If `matrix` is `true`, returns a lower triangular matrix constructed from the
  outputs of the `diag` and `lower` components of the MetricChain.
- If `matrix` is `false`, returns a `NamedTuple` with two elements: `diag`, the
  output of the `diag` component of the MetricChain, and `lower`, the output of
  the `lower` component of the MetricChain.

# Example
```julia
m = MetricChain(...)
x = rand(Float32, 100, 10)
m(x, matrix=true)  # Returns a lower triangular matrix
```
"""
function (m::MetricChain)(x::Metal.MtlArray{Float32}; matrix::Bool=false)
    # Compute the output of the MLP
    mlp_out = m.mlp(x)

    # Compute the diagonal elements of the lower-triangular matrix
    diag_out = m.diag(mlp_out)

    # Compute the off-diagonal elements of the lower-triangular matrix
    lower_out = m.lower(mlp_out)

    # Check if matrix should be returned
    if matrix
        return vec_to_ltri(diag_out, lower_out)
    else
        return (diag=diag_out, lower=lower_out,)
    end # if
end # function

# ------------------------------------------------------------------------------

function G_inv(
    z::Metal.MtlVector{Float32},
    centroids_latent::Metal.MtlMatrix{Float32},
    M::Metal.MtlArray{Float32,3},
    T::Float32,
    λ::Float32,
)
    # Define dimensionality of latent space
    n = size(centroids_latent, 1)
    # Define number of centroids
    n_centroids = size(centroids_latent, 2)

    # Initialize array of zeros to save L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²)
    LLexp = Metal.zeros(Float32, n, n)

    # Loop through each centroid
    for i = 1:n_centroids
        # Compute L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²). Notes: 
        # - We do not use Distances.jl because that performs in-place operations
        #   on the input, and this is not compatible with Zygote.jl.
        # - We use Zygote.dropgrad to prevent backpropagation through the
        #   hyperparameters.
        LLexp .+= M[:, :, i] .*
                  exp(-sum(abs2, (z - centroids_latent[:, i]) ./
                                 Zygote.dropgrad(T)))
    end # for

    # Add regularization term
    LLexp[LinearAlgebra.diagind(LLexp)] .+= Zygote.dropgrad(λ)

    return LLexp



    # # Return the sum of the LLexp slices plus the regularization term
    # return LLexp +
    #        LinearAlgebra.diagm(Zygote.dropgrad(λ) * ones(Float32, size(z, 1)))
end # function