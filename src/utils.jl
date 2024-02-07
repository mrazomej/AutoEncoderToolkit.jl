# Import CUDA library
import CUDA

# Import ML library
import Flux

# Import AutoDiff library
import Zygote

# Import library to find nearest neighbors
import NearestNeighbors
import Clustering

# Import lobary to conditionally load functions when GPUs are available
import Requires

# Import library for random sampling
import Distributions
import StatsBase
import Random

# Import library for basic math
import LinearAlgebra

# Export functions
export shuffle_data, cycle_anneal, locality_sampler, vec_to_ltri,
    centroids_kmeans

## =============================================================================

@doc raw"""
    `step_scheduler(epoch, epoch_change, learning_rates)`

Simple function to define different learning rates at specified epochs.

# Arguments
- `epoch::Int`: Epoch at which to define learning rate.
- `epoch_change::Vector{<:Int}`: Number of epochs at which to change learning
  rate. It must include the initial learning rate!
- `learning_rates::Vector{<:AbstractFloat}`: Learning rate value for the epoch
  range. Must be the same length as `epoch_change`

# Returns
- `η::Abstr`
"""
function step_scheduler(
    epoch::Int,
    epoch_change::AbstractVector{<:Int},
    learning_rates::AbstractVector{<:AbstractFloat}
)
    # Check that arrays are of the same length
    if length(epoch_change) != length(learning_rates)
        error("epoch_change and learning_rates must be of the same length")
    end # if

    # Sort arrays to make sure it goes in epoch order.
    idx_sort = sortperm(epoch_change)
    sort!(epoch_change)
    learning_rates = learning_rates[idx_sort]

    # Check if there is any rate that belongs to the epoch
    if any(epoch .≤ epoch_change)
        # Return corresponding learning rate
        return first(learning_rates[epoch.≤epoch_change])
    else
        return learning_rates[end]
    end
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    `cycle_anneal(epoch, n_epoch, n_cycles; frac=0.5, βmax=1, βmin=0)`

Function that computes the value of the annealing parameter β for a variational
autoencoder as a function of the epoch number according to the cyclical
annealing strategy.

# Arguments
- `epoch::Int`: Epoch on which to evaluate the value of the annealing parameter.
- `n_epoch::Int`: Number of epochs that will be run to train the VAE.
- `n_cycles::Int`: Number of annealing cycles to be fit within the number of
  epochs.

## Optional Arguments
- `frac::AbstractFloat= 0.5`: Fraction of the cycle in which the annealing
  parameter β will increase from the minimum to the maximum value.
- `βmax::AbstractFloat=1.0`: Maximum value that the annealing parameter can
  reach.
- `βmin::AbstractFloat=0.0`: Minimum value that the annealing parameter can
  reach.

# Returns
- `β::Float32`: Value of the annealing parameter.

# Citation
> Fu, H. et al. Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
> Vanishing. Preprint at http://arxiv.org/abs/1903.10145 (2019).
"""
function cycle_anneal(
    epoch::Int,
    n_epoch::Int,
    n_cycles::Int;
    frac::AbstractFloat=0.5f0,
    βmax::AbstractFloat=1.0f0,
    βmin::AbstractFloat=0.0f0
)
    # Validate frac
    if !(0 ≤ frac ≤ 1)
        throw(ArgumentError("Frac must be between 0 and 1"))
    end # if

    # Define variable τ that will serve to define the value of β
    τ = mod(epoch - 1, ceil(n_epoch / n_cycles)) / (n_epoch / n_cycles)

    # Compute and return the value of β
    if τ ≤ frac
        return convert(Float32, (βmax - βmin) * τ / frac + βmin)
    else
        return convert(Float32, βmax)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
`locality_sampler(data, dist_tree, n_primary, n_secondary, k_neighbors;
index=false)`

Algorithm to generate mini-batches based on spatial locality as determined by a
pre-constructed nearest neighbors tree.

# Arguments
- `data::Matrix{<:Float32}`: A matrix containing the data points in its columns.
- `dist_tree::NearestNeighbors.NNTree`: `NearestNeighbors.jl` tree used to
  determine the distance between data points.
- `n_primary::Int`: Number of primary points to sample.
- `n_secondary::Int`: Number of secondary points to sample from the neighbors of
  each primary point.
- `k_neighbors::Int`: Number of nearest neighbors from which to potentially
  sample the secondary points.

# Optional Keyword Arguments
- `index::Bool`: If `true`, returns the indices of the selected samples. If
  `false`, returns the `data` corresponding to the indexes. Defaults to `false`.

# Returns
- `sample_idx::Vector{Int64}`: Indices of data points to include in the
  mini-batch.

# Description
This sampling algorithm consists of three steps:
1. For each datapoint, determine the `k_neighbors` nearest neighbors using the
   `dist_tree`.
2. Uniformly sample `n_primary` points without replacement from all data points.
3. For each primary point, sample `n_secondary` points without replacement from
   its `k_neighbors` nearest neighbors.

# Examples
```julia
# Pre-constructed NearestNeighbors.jl tree
dist_tree = NearestNeighbors.KDTree(data, metric)
sample_indices = locality_sampler(data, dist_tree, 10, 5, 50)
```

# Citation
> Skafte, N., Jø rgensen, M. & Hauberg, S. ren. Reliable training and estimation
> of variance networks. in Advances in Neural Information Processing Systems
> vol. 32 (Curran Associates, Inc., 2019).
"""
function locality_sampler(
    data::Matrix{<:Float32},
    dist_tree::NearestNeighbors.NNTree,
    n_primary::Int,
    n_secondary::Int,
    k_neighbors::Int;
    index::Bool=false
)
    # Check that n_secondary ≤ k_neighbors
    if !(n_secondary ≤ k_neighbors)
        # Report error
        error("n_secondary must be ≤ k_neighbors")
    end # if

    # Sample n_primary primary sampling units with uniform probability without
    # replacement among all N units
    idx_primary = StatsBase.sample(1:size(data, 2), n_primary, replace=false)

    # Extract primary sample
    sample_primary = @view data[:, idx_primary]

    # Compute k_nearest neighbors for each of the points
    k_idxs, dists = NearestNeighbors.knn(
        dist_tree, sample_primary, k_neighbors, true
    )

    # For each of the primary sampling units sample n_secondary secondary
    # sampling units among the primary sampling units k_neighbors nearest
    # neighbors with uniform probability without replacement.
    idx_secondary = vcat([
        StatsBase.sample(p, n_secondary, replace=false) for p in k_idxs
    ]...)

    # Return minibatch data
    if index
        return [idx_primary; idx_secondary]
    else
        return @view data[:, [idx_primary; idx_secondary]]
    end # if
end # function

## =============================================================================
# Convert vector to lower triangular matrix
## =============================================================================

@doc raw"""
        vec_to_ltri{T}(diag::AbstractVector{T}, lower::AbstractVector{T})

Convert two one-dimensional vectors into a lower triangular matrix.

# Arguments
- `diag::AbstractVector{T}`: The input vector to be converted into the diagonal
    of the matrix.
- `lower::AbstractVector{T}`: The input vector to be converted into the lower
    triangular part of the matrix. The length of this vector should be a
    triangular number (i.e., the sum of the first `n` natural numbers for some
    `n`).

# Returns
- A lower triangular matrix constructed from `diag` and `lower`.

# Description
This function constructs a lower triangular matrix from two input vectors,
`diag` and `lower`. The `diag` vector provides the diagonal elements of the
matrix, while the `lower` vector provides the elements below the diagonal. The
function uses a comprehension to construct the matrix, with the `lower_index`
function calculating the appropriate index in the `lower` vector for each
element below the diagonal.

# Example
```julia
diag = [1, 2, 3]
lower = [4, 5, 6]
vec_to_ltri(diag, lower)  # Returns a 3x3 lower triangular matrix
```
"""
function vec_to_ltri(
    diag::AbstractVector{T}, lower::AbstractVector{T},
) where {T<:Number}
    # Define dimensionality of the matrix
    n = length(diag)

    # Define a function to calculate the index in the 'lower' array
    lower_index = Zygote.ignore() do
        (i, j) -> (i - 1) * (i - 2) ÷ 2 + j
    end # function

    # Create the matrix using a comprehension
    return reshape(
        [
            i == j ? diag[i] :
            i > j ? lower[lower_index(i, j)] :
            zero(T) for i in 1:n, j in 1:n
        ],
        n, n
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        vec_to_ltri{T}(diag::AbstractVector{T}, lower::AbstractVector{T})

Convert two one-dimensional vectors into a lower triangular matrix.

# Arguments
- `diag::AbstractVector{T}`: The input vector to be converted into the diagonal
    of the matrix.
- `lower::AbstractVector{T}`: The input vector to be converted into the lower
    triangular part of the matrix. The length of this vector should be a
    triangular number (i.e., the sum of the first `n` natural numbers for some
    `n`).

# Returns
- A lower triangular matrix constructed from `diag` and `lower`.

# Description
This function constructs a lower triangular matrix from two input vectors,
`diag` and `lower`. The `diag` vector provides the diagonal elements of the
matrix, while the `lower` vector provides the elements below the diagonal. The
function uses a comprehension to construct the matrix, with the `lower_index`
function calculating the appropriate index in the `lower` vector for each
element below the diagonal.

# Example
```julia
using CUDA
diag = cu([1, 2, 3])
lower = cu([4, 5, 6])
vec_to_ltri(diag, lower)  # Returns a 3x3 lower triangular matrix
```
"""
function vec_to_ltri(
    diag::CUDA.CuVector{T}, lower::CUDA.CuVector{T},
) where {T<:Number}
    return vec_to_ltri(diag |> Flux.cpu, lower |> Flux.cpu) |> Flux.gpu
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(
        diag::AbstractMatrix{T}, lower::AbstractMatrix{T}
    ) where {T<:Number}

Construct a set of lower triangular matrices from a matrix of diagonal elements
and a matrix of lower triangular elements, each column representing a sample.

# Arguments
- `diag::AbstractMatrix{T}`: A matrix of `T` where each column contains the
    diagonal elements of the matrix for a specific sample.
- `lower::AbstractMatrix{T}`: A matrix of `T` where each column contains the
    elements of the lower triangle of the matrix for a specific sample.

# Returns
- A 3D array of type `T` where each slice along the third dimension is a lower
    triangular matrix with the diagonal and lower triangular elements populated
    from `diag` and `lower` respectively.

# Description
This function constructs a set of lower triangular matrices from two input
matrices, `diag` and `lower`. The `diag` matrix provides the diagonal elements
of the matrices, while the `lower` matrix provides the elements below the
diagonal. The function uses a comprehension to construct the matrices, with the
`lower_index` function calculating the appropriate index in the `lower` matrix
for each element below the diagonal.

# Note
The function assumes that the `diag` and `lower` matrices have the correct
dimensions for the matrices to be constructed. Specifically, `diag` and `lower`
should have `n` rows and `m` columns, where `n` is the dimension of the matrix
and `m` is the number of samples. The `lower` matrix should have `n*(n-1)/2`
non-zero elements per column, corresponding to the lower triangular part of the
matrix.
"""
function vec_to_ltri(
    diag::AbstractMatrix{T}, lower::AbstractMatrix{T}
) where {T<:Number}
    # Extract matrix dimensions and number of samples
    n, cols = size(diag)

    # Check that 'lower' has the correct dimensions
    if size(lower) != (n * (n - 1) ÷ 2, cols)
        error("Dimension mismatch between 'diag' and 'lower' matrices")
    end

    # Define a function to calculate the index in the 'lower' array for each
    # column
    lower_index = Zygote.ignore() do
        (col, i, j) -> (i - 1) * (i - 2) ÷ 2 + j + (col - 1) * (n * (n - 1) ÷ 2)
    end # function

    # Create the 3D tensor using a comprehension
    return reshape(
        [
            i == j ? diag[i, k] :
            i > j ? lower[lower_index(k, i, j)] :
            zero(T) for i in 1:n, j in 1:n, k in 1:cols
        ],
        n, n, cols
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(
        diag::AbstractMatrix{T}, lower::AbstractMatrix{T}
    ) where {T<:Number}

Construct a set of lower triangular matrices from a matrix of diagonal elements
and a matrix of lower triangular elements, each column representing a sample.

# Arguments
- `diag::AbstractMatrix{T}`: A matrix of `T` where each column contains the
    diagonal elements of the matrix for a specific sample.
- `lower::AbstractMatrix{T}`: A matrix of `T` where each column contains the
    elements of the lower triangle of the matrix for a specific sample.

# Returns
- A 3D array of type `T` where each slice along the third dimension is a lower
    triangular matrix with the diagonal and lower triangular elements populated
    from `diag` and `lower` respectively.

# Description
This function constructs a set of lower triangular matrices from two input
matrices, `diag` and `lower`. The `diag` matrix provides the diagonal elements
of the matrices, while the `lower` matrix provides the elements below the
diagonal. The function uses a comprehension to construct the matrices, with the
`lower_index` function calculating the appropriate index in the `lower` matrix
for each element below the diagonal.

# Note
The function assumes that the `diag` and `lower` matrices have the correct
dimensions for the matrices to be constructed. Specifically, `diag` and `lower`
should have `n` rows and `m` columns, where `n` is the dimension of the matrix
and `m` is the number of samples. The `lower` matrix should have `n*(n-1)/2`
non-zero elements per column, corresponding to the lower triangular part of the
matrix.
"""
function vec_to_ltri(
    diag::CUDA.CuMatrix{T}, lower::CUDA.CuMatrix{T}
) where {T<:Number}
    return CUDA.cu(vec_to_ltri(diag |> Flux.cpu, lower |> Flux.cpu))
end # function

## =============================================================================
# Define centroids via k-means
## =============================================================================

@doc raw"""
    centroids_kmeans(x::AbstractMatrix{<:AbstractFloat}, n_centroids::Int; assign::Bool=false)

Perform k-means clustering on the input and return the centers. This function
can be used to down-sample the number of points used when computing the metric
tensor in training a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractMatrix{<:AbstractFloat}`: The input data. Rows represent
  individual samples.
- `n_centroids::Int`: The number of centroids to compute.

# Optional Keyword Arguments
- `assign::Bool=false`: If true, also return the assignments of each point to a
  centroid.

# Returns
- If `assign` is false, returns a matrix where each column is a centroid.
- If `assign` is true, returns a tuple where the first element is the matrix of
  centroids and the second element is a vector of assignments.

# Examples
```julia
data = rand(100, 10)
centroids = centroids_kmeans(data, 5)
```
"""
function centroids_kmeans(
    x::AbstractMatrix{<:Number},
    n_centroids::Int;
    assign::Bool=false
)
    # Perform k-means clustering on the input and return the centers
    if assign
        # Compute clustering
        clustering = Clustering.kmeans(x, n_centroids)
        # Return centers and assignments
        return (clustering.centers, Clustering.assignments(clustering))
    else
        # Return centers
        return Clustering.kmeans(x, n_centroids).centers
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    centroids_kmeans(
        x::AbstractArray{<:AbstractFloat}, n_centroids::Int; reshape_centroids::Bool=true, assign::Bool=false
    )

Perform k-means clustering on the input and return the centers. This function
can be used to down-sample the number of points used when computing the metric
tensor in training a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

The input data is flattened into a matrix before performing k-means clustering.
This is done because k-means operates on a set of data points in a vector space
and cannot handle multi-dimensional arrays. Flattening the input ensures that
the k-means algorithm can process the data correctly.

By default, the output centroids are reshaped back to the original input shape.
This is controlled by the `reshape_centroids` argument.

# Arguments
- `x::AbstractArray{<:AbstractFloat}`: The input data. It can be a
  multi-dimensional array where the last dimension represents individual
  samples.
- `n_centroids::Int`: The number of centroids to compute.

# Optional Keyword Arguments
- `reshape_centroids::Bool=true`: If true, reshape the output centroids back to
  the original input shape.
- `assign::Bool=false`: If true, also return the assignments of each point to a
  centroid.

# Returns
- If `assign` is false, returns a matrix where each column is a centroid.
- If `assign` is true, returns a tuple where the first element is the matrix of
  centroids and the second element is a vector of assignments.

# Examples
```julia
data = rand(100, 10)
centroids = centroids_kmeans(data, 5)
```
"""
function centroids_kmeans(
    x::AbstractArray{<:Number},
    n_centroids::Int;
    reshape_centroids::Bool=true,
    assign::Bool=false
)
    # Flatten input into matrix
    x_flat = Flux.flatten(x)

    # Check if output should be reshaped
    if reshape_centroids
        # Perform k-means clustering on the input and return the centers
        if assign
            # Compute clustering
            clustering = Clustering.kmeans(x_flat, n_centroids)
            # Extract centeres
            centers = clustering.centers
            # Reshape centers
            centers = reshape(centers, size(x)[1:end-1]..., n_centroids)
            # Return centers and assignments
            return (centers, Clustering.assignments(clustering))
        else
            # Compute clustering
            clustering = Clustering.kmeans(x_flat, n_centroids)
            # Extract centeres
            centers = clustering.centers
            # Reshape centers
            centers = reshape(centers, size(x)[1:end-1]..., n_centroids)
            # Return centers
            return centers
        end # if
    else
        # Perform k-means clustering on the input and return the centers
        if assign
            # Compute clustering
            clustering = Clustering.kmeans(x_flat, n_centroids)
            # Return centers and assignments
            return (clustering.centers, Clustering.assignments(clustering))
        else
            # Return centers
            return Clustering.kmeans(x_flat, n_centroids).centers
        end # if
    end # if
end # function

# =============================================================================
# Computing the log determinant of a matrix via LU decomposition.
# This is inspired by TensorFlow's slogdet function.
# =============================================================================

@doc raw"""
    slogdet(A::AbstractMatrix{T}) where {T<:AbstractFloat}

Calculate the signed logarithm of the determinant of a matrix `A`.

# Arguments
- `A::AbstractMatrix{T}`: The input matrix, where `T` is a subtype of
  `AbstractFloat`.

# Returns
- The signed logarithm of the determinant of `A`.

# Details
This function computes the sign and the logarithm of the absolute value of the
determinant of a matrix using a partially pivoted LU decomposition. The sign and
the logarithm of the absolute value of the determinant are multiplied to return
the signed logarithm of the determinant.
"""
function slogdet(A::AbstractMatrix{T}) where {T<:AbstractFloat}
    # Compute the log determinant through a Partially Pivoted LU decomposition
    # Perform LU decomposition
    lu = LinearAlgebra.lu(A)
    # Get the LU factors
    LU = lu.factors
    # Compute the sign of the permutation matrix
    sign = LinearAlgebra.det(lu.P)
    # Get the diagonal elements of LU
    diag = LinearAlgebra.diag(LU)
    # Take the absolute value of the diagonal elements
    abs_diag = abs.(diag)
    # Compute the sum of the logarithm of absolute diagonal elements
    log_abs_det = sum(log.(abs_diag))
    # Compute the sign of the determinant
    sign = prod(diag ./ abs_diag)

    # Return the signed logarithm of the determinant
    return log_abs_det * sign
end # function

## =============================================================================
# Defining random number generators for different GPU backends
## =============================================================================

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::AbstractMatrix{T}) where {T<:AbstractFloat}

Draw a random sample from a multivariate normal distribution in canonical form.

# Arguments
- `Σ⁻¹::AbstractMatrix{T}`: The precision matrix (inverse of the covariance
  matrix) of the multivariate normal distribution. `T` is a subtype of
  `AbstractFloat`.

# Returns
- A random sample drawn from the multivariate normal distribution specified by
  the input precision matrix.
"""
function sample_MvNormalCanon(
    Σ⁻¹::AbstractMatrix{T}
) where {T<:AbstractFloat}
    # Ensure matrix is symmetric
    Σ⁻¹ = LinearAlgebra.symmetric(Σ⁻¹, :L)

    # Sample from the multivariate normal distribution
    return Random.rand(
        Distributions.MvNormalCanon(Σ⁻¹)
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::CUDA.CuMatrix{T}) where {T<:AbstractFloat}

Draw a random sample from a multivariate normal distribution in canonical form,
specifically for a precision matrix stored on the GPU.

# Arguments
- `Σ⁻¹::CUDA.CuMatrix{T}`: The precision matrix (inverse of the covariance
  matrix) of the multivariate normal distribution, stored on the GPU. `T` is a
  subtype of `AbstractFloat`.

# Returns
- A random sample drawn from the multivariate normal distribution specified by
  the input precision matrix, returned as a GPU array.

# Behavior
For `CuMatrix` inputs, this function first transfers the precision matrix to the
CPU using `Flux.cpu`. It then draws a sample from the multivariate normal
distribution using the `rand` function from the `Distributions.jl` package.
Finally, it transfers the sample back to the GPU using `Flux.gpu`.
"""
function sample_MvNormalCanon(
    Σ⁻¹::CUDA.CuMatrix{T}
) where {T<:AbstractFloat}
    # Ensure matrix is symmetric
    Σ⁻¹ = LinearAlgebra.symmetric(Σ⁻¹ |> Flux.cpu, :L)

    return Random.rand(
        Distributions.MvNormalCanon(Σ⁻¹)
    ) |> Flux.gpu
end # function

# Set Zygote to ignore the function when computing gradients
Zygote.@nograd sample_MvNormalCanon

## =============================================================================
# Define finite difference gradient function
## =============================================================================

"""
    unit_vector(x::AbstractVector, i::Int, T::Type=Float32)

Create a unit vector of the same length as `x` with the `i`-th element set to 1.

# Arguments
- `x::AbstractVector`: The vector whose length is used to determine the
  dimension of the unit vector.
- `i::Int`: The index of the element to be set to 1.
- `T::Type=Float32`: The type of the elements in the vector. Defaults to
  `Float32`.

# Returns
- A unit vector of type `T` and length equal to `x` with the `i`-th element set
  to 1.

# Description
This function creates a unit vector of the same length as `x` with the `i`-th
element set to 1. All other elements are set to 0. The type of the elements in
the vector is `T`, which defaults to `Float32`.

# Note
This function is marked with the `@nograd` macro from the Zygote package, which
means that Zygote will ignore any call to this function when computing
gradients.
"""
function unit_vector(x::AbstractVector, i::Int, T::Type=Float32)
    # Initialize a vector of zeros
    e = zeros(T, length(x))
    # Set the i-th element to 1
    e[i] = one(T)
    return e
end # function

# ------------------------------------------------------------------------------

"""
    unit_vector(x::CUDA.CuVector, i::Int, T::Type=Float32)

Create a unit vector of the same length as `x` with the `i`-th element set to 1,
specifically for a vector stored on the GPU.

# Arguments
- `x::CUDA.CuVector`: The GPU vector whose length is used to determine the
  dimension of the unit vector.
- `i::Int`: The index of the element to be set to 1.
- `T::Type=Float32`: The type of the elements in the vector. Defaults to
  `Float32`.

# Returns
- A unit vector of type `T` and length equal to `x` with the `i`-th element set
  to 1, returned as a GPU array.

# Description
This function creates a unit vector of the same length as `x` with the `i`-th
element set to 1. All other elements are set to 0. The type of the elements in
the vector is `T`, which defaults to `Float32`. The vector is created on the GPU
using the CUDA.jl package.

# Note
This function is marked with the `@nograd` macro from the Zygote package, which
means that Zygote will ignore any call to this function when computing
gradients.
"""
function unit_vector(x::CUDA.CuVector, i::Int, T::Type=Float32)
    # Create a unit vector with a list comprehension
    return CUDA.cu([j == i ? CUDA.one(T) : CUDA.zero(T) for j in 1:length(x)])
end # function


# Set Zygote to ignore the function when computing gradients
Zygote.@nograd unit_vector

# ------------------------------------------------------------------------------

"""
    finite_difference_gradient(
        f::Function, x::AbstractVector{T}; ε::T=sqrt(eps(T))
    ) where {T<:AbstractFloat}

Compute the finite difference gradient of a function `f` at a point `x`.

# Arguments
- `f::Function`: The function for which the gradient is to be computed.
- `x::AbstractVector{T}`: The point at which the gradient is to be computed.

# Optional Keyword Arguments
- `ε::T=sqrt(eps(Float32))`: The step size for the finite difference
  calculation. Defaults to the square root of the machine epsilon for type
  `Float32`.

# Returns
- A vector representing the gradient of `f` at `x`.

# Description
This function computes the finite difference gradient of a function `f` at a
point `x`. The gradient is a vector where the `i`-th element is the partial
derivative of `f` with respect to the `i`-th element of `x`.

The partial derivatives are computed using the central difference formula:

∂f/∂xᵢ ≈ [f(x + ε * eᵢ) - f(x - ε * eᵢ)] / 2ε

where eᵢ is the `i`-th unit vector.

# Example
```julia
f(x) = sum(x.^2)
x = [1.0, 2.0, 3.0]
finite_difference_gradient(f, x)  # Returns the vector [2.0, 4.0, 6.0]
```
"""
function finite_difference_gradient(
    f::Function,
    x::AbstractVector{T};
    ε::T=sqrt(eps(Float32))
) where {T<:AbstractFloat}
    # Compute the finite difference gradient for each element of x
    grad = [
        (
            f(x .+ ε * unit_vector(x, i, T)) -
            f(x .- ε * unit_vector(x, i, T))
        ) / 2ε for i in eachindex(x)
    ]
    return grad
end # function

# ------------------------------------------------------------------------------

"""
    active_selection(
        f::Function, x::CUDA.CuVector{T}; ε::T=sqrt(eps(Float32))
    ) where {T<:AbstractFloat}

Compute the finite difference gradient of a function `f` at a point `x` and
upload it to the GPU.

# Arguments
- `f::Function`: The function for which the gradient is to be computed.
- `x::CUDA.CuVector{T}`: The point at which the gradient is to be computed.

# Optional Keyword Arguments
- `ε::T=sqrt(eps(Float32))`: The step size for the finite difference
  calculation. Defaults to the square root of the machine epsilon for type
  `Float32`.

# Returns
- A CUDA array representing the gradient of `f` at `x`.

# Description
This function computes the finite difference gradient of a function `f` at a
point `x`. The gradient is a CUDA array where the `i`-th element is the partial
derivative of `f` with respect to the `i`-th element of `x`.

The partial derivatives are computed using the central difference formula:

∂f/∂xᵢ ≈ [f(x + ε * eᵢ) - f(x - ε * eᵢ)] / 2ε

where eᵢ is the `i`-th unit vector.

The output is uploaded to the GPU for efficient computation with CUDA arrays.

# Example
```julia
f(x) = sum(x.^2)
x = CUDA.cu([1.0, 2.0, 3.0])
active_selection(f, x)  # Returns the CUDA array [2.0, 4.0, 6.0]
```
"""
function finite_difference_gradient(
    f::Function,
    x::CUDA.CuVector{T};
    ε::T=sqrt(CUDA.eps(Float32))
) where {T<:AbstractFloat}
    # Compute the finite difference gradient for each element of x
    grad = CUDA.cu([
        (
            f(x .+ ε * unit_vector(x, i, T)) -
            f(x .- ε * unit_vector(x, i, T))
        ) / 2ε for i in eachindex(x)
    ])
    return grad
end # function
