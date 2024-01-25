# Import library to find nearest neighbors
import NearestNeighbors
import Clustering

# Import lobary to conditionally load functions when GPUs are available
import Requires

# Import library for random sampling
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

"""
        tril_indices(n::Int, offset::Int=0)

Return the row and column indices of the lower triangular part of an `n x n`
matrix, shifted by a given offset.

# Arguments
- `n::Int`: The size of the square matrix.

# Optional Keyword Arguments
- `offset::Int=0`: The offset by which to shift the main diagonal. A positive
  offset selects an upper diagonal, and a negative offset selects a lower
  diagonal.

# Returns
- A matrix of size `n*(n+1)/2 x 2` containing the row and column indices of the
  lower triangular part of the matrix, shifted by the offset. The indices are
  1-based, following Julia's convention.

# Note
This function does not actually create a matrix. It only calculates the indices
that you would use to access the lower triangular part of an `n x n` matrix,
shifted by the offset.
"""
function tril_indices(n::Int; offset::Int=0)
    # Initialize an empty array to store row indices
    rows = Int[]
    # Initialize an empty array to store column indices
    cols = Int[]
    # Iterate over the rows
    for i in 1:n
        # Iterate over the columns up to i+offset
        for j in 1:(i+offset)
            # Append the current row index to the rows array
            push!(rows, i)
            # Append the current column index to the cols array
            push!(cols, j)
        end # for j
    end # for i
    return hcat(rows, cols)
end

# ------------------------------------------------------------------------------

"""
    tril_indices(n::Int, m::Int; offset::Int=0)

Return the row, column, and sample indices of the lower triangular part of an `n x n` matrix for `m` samples, shifted by a given offset.

# Arguments
- `n::Int`: The size of the square matrix.
- `m::Int`: The number of samples.
- `offset::Int=0`: The offset by which to shift the main diagonal. A positive offset selects an upper diagonal, and a negative offset selects a lower diagonal.

# Returns
- A matrix of size `n*(n+1)/2 x 3` containing the row, column, and sample indices of the lower triangular part of the matrix, shifted by the offset. The indices are 1-based, following Julia's convention.

# Note
This function does not actually create a matrix. It only calculates the indices that you would use to access the lower triangular part of an `n x n` matrix, shifted by the offset, for `m` samples.
"""
function tril_indices(n::Int, m::Int; offset::Int=0)
    # Initialize an empty array to store row indices
    rows = Int[]
    # Initialize an empty array to store column indices
    cols = Int[]
    # Initialize an empty array to store sample indices
    samples = Int[]
    # Iterate over samples
    for k in 1:m
        # Iterate over the rows
        for i in 1:n
            # Iterate over the columns up to i+offset
            for j in 1:(i+offset)
                # Append the current row index to the rows array
                push!(rows, i)
                # Append the current column index to the cols array
                push!(cols, j)
                # Append the sample index
                push!(samples, k)
            end # for j
        end # for i
    end # for k
    # Return a tuple of the row and column indices
    return hcat(rows, cols, samples)
end # function

# ------------------------------------------------------------------------------

"""
    diag_indices(n::Int, m::Int)

Return the row, column, and sample indices of the diagonal elements of an `n x
n` matrix for `m` samples.

# Arguments
- `n::Int`: The size of the square matrix.
- `m::Int`: The number of samples.

# Returns
- A tuple of three arrays: the row indices, the column indices, and the sample
  indices of the diagonal elements of the matrix. The indices are 1-based,
  following Julia's convention.

# Note
This function does not actually create a matrix. It only calculates the indices
that you would use to access the diagonal elements of an `n x n` matrix for `m`
samples.
"""
function diag_indices(n::Int, m::Int)
    # Initialize an empty array to store row/column indices
    indices = Int[]
    # Initialize an empty array to store sample indices
    samples = Int[]
    # Iterate over samples
    for k in 1:m
        # Iterate over the rows/columns
        for i in 1:n
            # Append the current row/column index to the indices array
            push!(indices, i)
            # Append the sample index
            push!(samples, k)
        end # for i
    end # for k
    # Return a tuple of the row/column indices and sample indices
    return hcat(indices, indices, samples)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri{T}(diag::AbstractVector{T}, lower::AbstractVector{T}, z::T=zero(T))

Convert two one-dimensional vectors into a lower triangular matrix.

# Arguments
- `diag::AbstractVector{T}`: The input vector to be converted into the diagonal
  of the matrix.
- `lower::AbstractVector{T}`: The input vector to be converted into the lower
  triangular part of the matrix. The length of this vector should be a
  triangular number (i.e., the sum of the first `n` natural numbers for some
  `n`).
- `z::T=zero(T)`: The value to fill in the upper triangular part of the matrix.
  Defaults to zero.

# Returns
- A lower triangular matrix constructed from `diag` and `lower`, with the upper
  triangular part filled with `z`.

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
    # Calculate latent space dimensionality
    n = length(diag)
    # Initialize matrix of zeros
    L = zeros(T, n, n)
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
        vec_to_ltri(diag::AbstractMatrix{T}, lower::AbstractMatrix{T}) where {T<:Number}

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
    x::AbstractMatrix{<:AbstractFloat},
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

# =============================================================================



## =============================================================================
# Defining random number generators for different GPU backends
## =============================================================================

function randn_like(x::AbstractArray{T}) where {T<:AbstractFloat}
    return randn(T, size(x)...)
end

function __init__()

    # Define randn for CUDA
    Requires.@require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        function randn_like(x::CUDA.CuArray{T}) where {T<:AbstractFloat}
            return CUDA.randn(T, size(x)...)
        end
    end

    # Define randn for OpenCL
    Requires.@require OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2" begin
        function randn_like(x::OpenCL.CLArray{T}) where {T<:AbstractFloat}
            return OpenCL.randn(T, size(x)...)
        end
    end

    # Define randn for Metal
    Requires.@require Metal = "dde4c033-4e86-420c-a63e-0dd931031962" begin
        function randn_like(x::Metal.MtlArray{T}) where {T<:AbstractFloat}
            return Metal.randn(T, size(x)...)
        end
    end
end # function __init__