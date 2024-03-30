# Import CUDA library
import CUDA

# Import ML library
import Flux

# Import AutoDiff backends
import ChainRulesCore
import TaylorDiff

# Import library to find nearest neighbors
import NearestNeighbors
import Clustering
import Distances

# Import library to use Ellipsis Notation
using EllipsisNotation

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
    storage_type(A::AbstractArray)

Determine the storage type of an array.

This function recursively checks the parent of the array until it finds the base
storage type. This is useful for determining whether an array or its subarrays
are stored on the CPU or GPU.

# Arguments
- `A::AbstractArray`: The array whose storage type is to be determined.

# Returns
The type of the array that is the base storage of `A`.
"""
function storage_type(A::AbstractArray)
    # Get the parent of the array
    P = parent(A)
    # If the type of the array is the same as the type of its parent,
    # return the type of the array. Otherwise, recursively call storage_type
    # on the parent.
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end

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
    cycle_anneal(
        epoch::Int, 
        n_epoch::Int, 
        n_cycles::Int; 
        frac::AbstractFloat=0.5f0, 
        βmax::Number=1.0f0, 
        βmin::Number=0.0f0, 
        T::Type=Float32
    )

Function that computes the value of the annealing parameter β for a variational
autoencoder as a function of the epoch number according to the cyclical
annealing strategy.

# Arguments
- `epoch::Int`: Epoch on which to evaluate the value of the annealing parameter.
- `n_epoch::Int`: Number of epochs that will be run to train the VAE.
- `n_cycles::Int`: Number of annealing cycles to be fit within the number of
  epochs.

## Optional Arguments
- `frac::AbstractFloat= 0.5f0`: Fraction of the cycle in which the annealing
  parameter β will increase from the minimum to the maximum value.
- `βmax::Number=1.0f0`: Maximum value that the annealing parameter can reach.
- `βmin::Number=0.0f0`: Minimum value that the annealing parameter can reach.
- `T::Type=Float32`: The type of the output. The function will convert the
  output to this type.

# Returns
- `β::T`: Value of the annealing parameter.

# Citation
> Fu, H. et al. Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
> Vanishing. Preprint at http://arxiv.org/abs/1903.10145 (2019).
"""
function cycle_anneal(
    epoch::Int,
    n_epoch::Int,
    n_cycles::Int;
    frac::AbstractFloat=0.5f0,
    βmax::Number=1.0f0,
    βmin::Number=0.0f0,
    T::Type=Float32
)
    # Validate frac
    if !(0 ≤ frac ≤ 1)
        throw(ArgumentError("Frac must be between 0 and 1"))
    end # if

    # Define variable τ that will serve to define the value of β
    τ = mod(epoch - 1, ceil(n_epoch / n_cycles)) / (n_epoch / n_cycles)

    # Compute and return the value of β
    if τ ≤ frac
        return convert(T, (βmax - βmin) * τ / frac + βmin)
    else
        return convert(T, βmax)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
`locality_sampler(data, dist_tree, n_primary, n_secondary, k_neighbors;
index=false)`

Algorithm to generate mini-batches based on spatial locality as determined by a
pre-constructed nearest neighbors tree.

# Arguments
- `data::AbstractArray`: An array containing the data points. The data points
  can be of any dimension.
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
- If `index` is `true`, returns `sample_idx::Vector{Int64}`: Indices of data
  points to include in the mini-batch.
- If `index` is `false`, returns `sample_data::AbstractArray`: The data points
  to include in the mini-batch.

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
    data::AbstractArray,
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
    idx_primary = StatsBase.sample(
        1:size(data, ndims(data)), n_primary, replace=false
    )

    # Extract primary sample
    sample_primary = @view data[.., idx_primary]

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
        return @view data[.., [idx_primary; idx_secondary]]
    end # if
end # function

## =============================================================================
# Convert vector to lower triangular matrix
## =============================================================================

@doc raw"""
        vec_to_ltri(diag::AbstractVecOrMat, lower::AbstractVecOrMat)

Convert two one-dimensional vectors or matrices into a lower triangular matrix
or a 3D tensor.

# Arguments
- `diag::AbstractVecOrMat`: The input vector or matrix to be converted into the
  diagonal of the matrix. If it's a matrix, each column is considered as a
  separate vector.
- `lower::AbstractVecOrMat`: The input vector or matrix to be converted into the
  lower triangular part of the matrix. The length of this vector or the number
  of rows in this matrix should be a triangular number (i.e., the sum of the
  first `n` natural numbers for some `n`). If it's a matrix, each column is
  considered the lower part of a separate lower triangular matrix.

# Returns
- A lower triangular matrix or a 3D tensor where each slice is a lower
  triangular matrix constructed from `diag` and `lower`.

# Description
This function constructs a lower triangular matrix or a 3D tensor from two input
vectors or matrices, `diag` and `lower`. The `diag` vector or matrix provides
the diagonal elements of the matrix, while the `lower` vector or matrix provides
the elements below the diagonal. The function uses a comprehension to construct
the matrix or tensor, with the `lower_index` function calculating the
appropriate index in the `lower` vector or matrix for each element below the
diagonal.

# GPU Support
The function supports both CPU and GPU arrays. For GPU arrays, the data is first
transferred to the CPU, the lower triangular matrix or tensor is constructed,
and then it is transferred back to the GPU.
"""
function vec_to_ltri(
    diag::AbstractVecOrMat, lower::AbstractVecOrMat,
)
    _vec_to_ltri(storage_type(diag), diag, lower)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(diag::AbstractMatrix{T}, lower::AbstractMatrix{T})

AbstractVector implementation of `vec_to_ltri`.
"""
function _vec_to_ltri(
    ::Type, diag::AbstractVector, lower::AbstractVector,
)

    # Define dimensionality of the matrix
    n = length(diag)

    # Define a function to calculate the index in the 'lower' array
    lower_index = ChainRulesCore.ignore_derivatives() do
        (i, j) -> (i - 1) * (i - 2) ÷ 2 + j
    end # function

    # Create the matrix using a comprehension
    return reshape(
        [
            i == j ? diag[i] :
            i > j ? lower[lower_index(i, j)] :
            zero(eltype(diag)) for i in 1:n, j in 1:n
        ],
        n, n
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(diag::AbstractMatrix{T}, lower::AbstractMatrix{T})

AbstractMatrix implementation of `vec_to_ltri`.
"""
function _vec_to_ltri(
    ::Type, diag::AbstractMatrix, lower::AbstractMatrix
)
    # Extract matrix dimensions and number of samples
    n, cols = size(diag)

    # Check that 'lower' has the correct dimensions
    if size(lower) != (n * (n - 1) ÷ 2, cols)
        error("Dimension mismatch between 'diag' and 'lower' matrices")
    end

    # Define a function to calculate the index in the 'lower' array for each
    # column
    lower_index = ChainRulesCore.ignore_derivatives() do
        (col, i, j) -> (i - 1) * (i - 2) ÷ 2 + j + (col - 1) * (n * (n - 1) ÷ 2)
    end # function

    # Create the 3D tensor using a comprehension
    return reshape(
        [
            i == j ? diag[i, k] :
            i > j ? lower[lower_index(k, i, j)] :
            zero(eltype(diag)) for i in 1:n, j in 1:n, k in 1:cols
        ],
        n, n, cols
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_to_ltri(diag::AbstractVecOrMat, lower::AbstractVecOrMat)

GPU implementation of `vec_to_ltri`.
"""
function _vec_to_ltri(
    ::Type{T}, diag::AbstractVecOrMat, lower::AbstractVecOrMat
) where {T<:CUDA.CuArray}
    # transfer data to CPU
    diag, lower = diag |> Flux.cpu, lower |> Flux.cpu
    # convert to lower triangular matrix
    return _vec_to_ltri(typeof(diag), diag, lower) |> Flux.gpu
end # function

## =============================================================================
# Vector Matrix Vector multiplication
## =============================================================================

@doc raw"""
    vec_mat_vec_batched(
        v::AbstractVector, 
        M::AbstractMatrix, 
        w::AbstractVector
    )

Compute the product of a vector, a matrix, and another vector in the form v̲ᵀ
M̲̲ w̲.

This function takes two vectors `v` and `w`, and a matrix `M`, and computes the
product v̲ M̲̲ w̲. This function is added for consistency when calling multiple
dispatch.

# Arguments
- `v::AbstractVector`: A `d` dimensional vector.
- `M::AbstractMatrix`: A `d×d` matrix.
- `w::AbstractVector`: A `d` dimensional vector.

# Returns
A scalar which is the result of the product v̲ M̲̲ w̲ for the corresponding
vectors and matrix.

# Notes
This function uses the `LinearAlgebra.dot` function to perform the
multiplication of the matrix `M` with the vector `w`. The resulting vector is
then element-wise multiplied with the vector `v` and summed over the dimensions
to obtain the final result. This function is added for consistency when calling
multiple dispatch.
"""
function vec_mat_vec_batched(
    v::AbstractVector,
    M::AbstractMatrix,
    w::AbstractVector
)
    return LinearAlgebra.dot(v, M, w)
end # for

# ------------------------------------------------------------------------------

@doc raw"""
    vec_mat_vec_batched(
        v::AbstractMatrix, 
        M::AbstractArray, 
        w::AbstractMatrix
    )

Compute the batched product of vectors and matrices in the form v̲ᵀ M̲̲ w̲.

This function takes two matrices `v` and `w`, and a 3D array `M`, and computes
the batched product v̲ M̲̲ w̲. The computation is performed in a broadcasted
manner using the `Flux.batched_vec` function.

# Arguments
- `v::AbstractMatrix`: A `d×n` matrix, where `d` is the dimension of the vectors
  and `n` is the number of vectors.
- `M::AbstractArray`: A `d×d×n` array, where `d` is the dimension of the
  matrices and `n` is the number of matrices.
- `w::AbstractMatrix`: A `d×n` matrix, where `d` is the dimension of the vectors
  and `n` is the number of vectors.

# Returns
An `n` dimensional array where each element is the result of the product v̲ M̲̲
w̲ for the corresponding vectors and matrix.

# Notes
This function uses the `Flux.batched_vec` function to perform the batched
multiplication of the matrices in `M` with the vectors in `w`. The resulting
vectors are then element-wise multiplied with the vectors in `v` and summed over
the dimensions to obtain the final result.
"""
function vec_mat_vec_batched(
    v::AbstractMatrix,
    M::AbstractArray,
    w::AbstractMatrix
)
    # Compute v̲ M̲̲ w̲ in a broadcasted manner
    return vec(sum(v .* Flux.batched_vec(M, w), dims=1))
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_mat_vec_loop(
        v::AbstractVector, 
        M::AbstractMatrix, 
        w::AbstractVector
    )

Compute the product of a vector, a matrix, and another vector in the form v̲ᵀ
M̲̲ w̲ using loops.

This function takes two vectors `v` and `w`, and a matrix `M`, and computes the
product v̲ M̲̲ w̲ using nested loops. This method might be slower than using
batched operations, but it is needed when performing differentiation with
`Zygote.jl` over `TaylorDiff.jl`.

# Arguments
- `v::AbstractVector`: A `d` dimensional vector.
- `M::AbstractMatrix`: A `d×d` matrix.
- `w::AbstractVector`: A `d` dimensional vector.

# Returns
A scalar which is the result of the product v̲ M̲̲ w̲ for the corresponding
vectors and matrix.

# Notes
This function uses nested loops to perform the multiplication of the matrix `M`
with the vector `w`. The resulting vector is then element-wise multiplied with
the vector `v` and summed over the dimensions to obtain the final result. This
method might be slower than using batched operations, but it is needed when
performing differentiation with `Zygote.jl` over `TaylorDiff.jl`.
"""
function vec_mat_vec_loop(
    v::AbstractVector,
    M::AbstractMatrix,
    w::AbstractVector
)
    # Check dimensions to see if the multiplication is possible
    if size(v, 1) ≠ size(M, 1) || size(M, 2) ≠ size(w, 1)
        throw(DimensionMismatch("Dimensions of vectors and matrices do not match"))
    end # if
    # Compute v̲ M̲̲ w̲ in a loop
    return sum(
        begin
            v[i] * M[i, j] * w[j]
        end
        for i in axes(v, 1)
        for j in axes(w, 1)
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    vec_mat_vec_loop(
        v::AbstractMatrix, 
        M::AbstractArray, 
        w::AbstractMatrix
    )

Compute the product of vectors and matrices in the form v̲ᵀ M̲̲ w̲ using loops.

This function takes two matrices `v` and `w`, and a 3D array `M`, and computes
the product v̲ M̲̲ w̲ using nested loops. This method might be slower than using
batched operations, but it is needed when performing differentiation with
`Zygote.jl` over `TaylorDiff.jl`.

# Arguments
- `v::AbstractMatrix`: A `d×n` matrix, where `d` is the dimension of the vectors
  and `n` is the number of vectors.
- `M::AbstractArray`: A `d×d×n` array, where `d` is the dimension of the
  matrices and `n` is the number of matrices.
- `w::AbstractMatrix`: A `d×n` matrix, where `d` is the dimension of the vectors
  and `n` is the number of vectors.

# Returns
A `1×n` matrix where each element is the result of the product v̲ M̲̲ w̲ for the
corresponding vectors and matrix.

# Notes
This function uses nested loops to perform the multiplication of the matrices in
`M` with the vectors in `w`. The resulting vectors are then element-wise
multiplied with the vectors in `v` and summed over the dimensions to obtain the
final result. This method might be slower than using batched operations, but it
is needed when performing differentiation with `Zygote.jl` over `TaylorDiff.jl`.
"""
function vec_mat_vec_loop(
    v::AbstractMatrix,
    M::AbstractArray{<:Any,3},
    w::AbstractMatrix
)
    # Check dimensions to see if the multiplication is possible
    if size(v, 1) ≠ size(M, 1) || size(M, 2) != size(w, 1)
        throw(DimensionMismatch("Dimensions of vectors and matrices do not match"))
    end # if

    # Compute v̲ M̲̲ w̲ in a loop
    return [
        begin
            sum(
                begin
                    v[i, k] *
                    M[i, j, k] *
                    w[j, k]
                end
                for i in axes(v, 1)
                for j in axes(w, 1)
            )
        end for k in axes(v, 2)
    ]
end # function

## =============================================================================
# Define centroids via k-means
## =============================================================================

@doc raw"""
    centroids_kmeans(
        x::AbstractMatrix, 
        n_centroids::Int; 
        assign::Bool=false
    )

Perform k-means clustering on the input and return the centers. This function
can be used to down-sample the number of points used when computing the metric
tensor in training a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractMatrix`: The input data. Rows represent individual
  samples.
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
    x::AbstractMatrix,
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
        x::AbstractArray, 
        n_centroids::Int; 
        reshape_centroids::Bool=true, 
        assign::Bool=false
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
- `x::AbstractArray`: The input data. It can be a multi-dimensional
  array where the last dimension represents individual samples.
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
    x::AbstractArray,
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

## =============================================================================
# Define centroids via k-medoids
## =============================================================================

@doc raw"""
        centroids_kmedoids(
            x::AbstractMatrix, n_centroids::Int; assign::Bool=false
        )

Perform k-medoids clustering on the input and return the centers. This function
can be used to down-sample the number of points used when computing the metric
tensor in training a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractMatrix`: The input data. Rows represent individual
  samples.
- `n_centroids::Int`: The number of centroids to compute.
- `dist::Distances.PreMetric=Distances.Euclidean()`: The distance metric to use
  when computing the pairwise distance matrix.

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
centroids = centroids_kmedoids(data, 5)
```
"""
function centroids_kmedoids(
    x::AbstractMatrix,
    n_centroids::Int,
    dist::Distances.PreMetric=Distances.Euclidean();
    assign::Bool=false
)
    # Compute pairwise distance matrix
    dist_matrix = Distances.pairwise(dist, x, dims=2)
    # Perform k-means clustering on the input and return the centers
    if assign
        # Compute clustering
        clustering = Clustering.kmedoids(dist_matrix, n_centroids)
        # Return centers and assignments
        return (x[:, clustering.medoids], clustering.assignments)
    else
        # Return centers
        return x[:, Clustering.kmedoids(dist_matrix, n_centroids).medoids]
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    centroids_kmedoids(
        x::AbstractArray,
        n_centroids::Int,
        dist::Distances.PreMetric=Distances.Euclidean();
        assign::Bool=false
    )

Perform k-medoids clustering on the input and return the centers. This function
can be used to down-sample the number of points used when computing the metric
tensor in training a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractArray`: The input data. The last dimension of `x` should
  contain each of the samples that should be clustered.
- `n_centroids::Int`: The number of centroids to compute.
- `dist::Distances.PreMetric=Distances.Euclidean()`: The distance metric to use
  for the clustering. Defaults to Euclidean distance.

# Optional Keyword Arguments
- `assign::Bool=false`: If true, also return the assignments of each point to a
  centroid.

# Returns
- If `assign` is false, returns an array where each column is a centroid.
- If `assign` is true, returns a tuple where the first element is the array of
  centroids and the second element is a vector of assignments.

# Examples
```julia
data = rand(10, 100)
centroids = centroids_kmedoids(data, 5)
```
"""
function centroids_kmedoids(
    x::AbstractArray,
    n_centroids::Int,
    dist::Distances.PreMetric=Distances.Euclidean();
    assign::Bool=false
)
    # Compute pairwise distance matrix by collecting slices with respect to the
    # last dimension
    dist_matrix = Distances.pairwise(dist, collect(eachslice(x, dims=ndims(x))))
    # Perform k-means clustering on the input and return the centers
    if assign
        # Compute clustering
        clustering = Clustering.kmedoids(dist_matrix, n_centroids)
        # Return centers and assignments
        return (x[.., clustering.medoids], clustering.assignments)
    else
        # Return centers
        return x[.., Clustering.kmedoids(dist_matrix, n_centroids).medoids]
    end # if
end # function

# =============================================================================
# Computing the log determinant of a matrix via Cholesky decomposition.
# =============================================================================

@doc raw"""
    slogdet(A::AbstractArray{T}; check::Bool=false) where {T<:Number}

Compute the log determinant of a positive-definite matrix `A` or a 3D array of
such matrices.

# Arguments
- `A::AbstractArray{T}`: A positive-definite matrix or a 3D array of
  positive-definite matrices whose log determinant is to be computed.  
- `check::Bool=false`: A flag that determines whether to check if the input
  matrix `A` is positive-definite. Defaults to `false` due to numerical instability.

# Returns
- The log determinant of `A`. If `A` is a 3D array, returns a 1D array of log
  determinants, one for each slice along the third dimension of `A`.

# Description
This function computes the log determinant of a positive-definite matrix `A` or
a 3D array of such matrices. It first computes the Cholesky decomposition of
`A`, and then calculates the log determinant as twice the sum of the log of the
diagonal elements of the lower triangular matrix from the Cholesky
decomposition.

# Conditions
The input matrix `A` must be a positive-definite matrix, i.e., it must be
symmetric and all its eigenvalues must be positive. If `check` is set to `true`,
the function will throw an error if `A` is not positive-definite.

# GPU Support
The function supports both CPU and GPU arrays. 
"""
function slogdet(A::AbstractArray; check::Bool=false)
    _slogdet(storage_type(A), A; check=check)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    slogdet(A::AbstractArray; check::Bool=false)

AbstractMatrix implementation of `slogdet`.
"""
function _slogdet(
    ::Type, A::AbstractMatrix; check::Bool=false
)
    # Compute the Cholesky decomposition of A. 
    chol = LinearAlgebra.cholesky(A; check=check)
    # compute the log determinant of A as the sum of the log of the diagonal
    # elements of the Cholesky decomposition
    return 2 * sum(log.(LinearAlgebra.diag(chol.L)))
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    slogdet(A::AbstractArray{<:Number,3}; check::Bool=false)

AbstractArray implementation of `slogdet`.
"""
function _slogdet(
    ::Type, A::AbstractArray{<:Any,3}; check::Bool=false
)
    # Compute the Cholesky decomposition of each slice of A. 
    chol = [
        x.L for x in LinearAlgebra.cholesky.(eachslice(A, dims=3), check=check)
    ]

    # compute the log determinant of each slice of A as the sum of the log of
    # the diagonal elements of the Cholesky decomposition
    logdetA = @. 2 * sum(log, LinearAlgebra.diag(chol))

    return logdetA
end # function

@doc raw"""
    slogdet(A::CUDA.CuArray; check::Bool=false)

GPU AbstractArray implementation of `slogdet`.
"""
function _slogdet(
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

## =============================================================================
# Defining random number generators 
## =============================================================================

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::AbstractArray{T}) where {T<:Number}

Draw a random sample from a multivariate normal distribution in canonical form.

# Arguments
- `Σ⁻¹::AbstractArray{T}`: The precision matrix (inverse of the covariance
  matrix) of the multivariate normal distribution. This can be a 2D array
  (matrix) or a 3D array.

# Returns
- A random sample drawn from the multivariate normal distribution specified by
  the input precision matrix. If `Σ⁻¹` is a 3D array, returns a 2D array of
  samples, one for each slice along the third dimension of `Σ⁻¹`.

# Description
This function draws a random sample from a multivariate normal distribution
specified by a precision matrix `Σ⁻¹`. The precision matrix can be a 2D array
(matrix) or a 3D array. If `Σ⁻¹` is a 3D array, the function draws a sample for
each slice along the third dimension of `Σ⁻¹`.

The function first inverts the precision matrix to obtain the covariance matrix,
then performs a Cholesky decomposition of the covariance matrix. It then draws a
sample from a standard normal distribution and multiplies it by the lower
triangular matrix from the Cholesky decomposition to obtain the final sample.

# GPU Support
The function supports both CPU and GPU arrays.
"""
function sample_MvNormalCanon(Σ⁻¹::AbstractArray)
    return _sample_MvNormalCanon(storage_type(Σ⁻¹), Σ⁻¹)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::AbstractMatrix)

AbstractMatrix implementation of `sample_MvNormalCanon`.
"""
function _sample_MvNormalCanon(
    ::Type, Σ⁻¹::AbstractMatrix
)
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
    r = randn(N, size(Σ⁻¹, 1))

    # Return sample multiplied by the Cholesky decomposition
    return chol.L * r
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T}) where {T<:Number}

GPU AbstractMatrix implementation of `sample_MvNormalCanon`.
"""
function _sample_MvNormalCanon(
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

# ------------------------------------------------------------------------------

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::AbstractArray{T,3}) where {T<:Number}

AbstractArray implementation of `sample_MvNormalCanon`.
"""
function _sample_MvNormalCanon(
    ::Type, Σ⁻¹::AbstractArray{<:Number,3}
)
    # Extract dimensions
    dim = size(Σ⁻¹, 1)
    # Extract number of samples
    n_sample = size(Σ⁻¹, 3)

    # Invert the precision matrix
    Σ = LinearAlgebra.inv.(eachslice(Σ⁻¹, dims=3))

    # Cholesky decomposition of the covariance matrix
    chol = reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            begin
                LinearAlgebra.cholesky(slice, check=false).L
            end for slice in Σ
        ]
    )

    # Define sample type
    if !(eltype(Σ⁻¹) <: AbstractFloat)
        N = Float32
    else
        N = eltype(Σ⁻¹)
    end # if

    # Sample from standard normal distribution
    r = randn(N, dim, n_sample)

    # Return sample multiplied by the Cholesky decomposition
    return Flux.batched_vec(chol, r)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    sample_MvNormalCanon(Σ⁻¹::CUDA.CuArray{T,3}) where {T<:Number}

GPU AbstractArray implementation of `sample_MvNormalCanon`.
"""
function _sample_MvNormalCanon(
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
end

## =============================================================================
# Define finite difference gradient function
## =============================================================================

@doc raw"""
    unit_vector(x::AbstractVector, i::Int)

Create a unit vector of the same length as `x` with the `i`-th element set to 1.

# Arguments
- `x::AbstractVector`: The vector whose length is used to determine the
  dimension of the unit vector.
- `i::Int`: The index of the element to be set to 1.

# Returns
- A unit vector of type `eltype(x)` and length equal to `x` with the `i`-th
  element set to 1.

# Description
This function creates a unit vector of the same length as `x` with the `i`-th
element set to 1. All other elements are set to 0.

# Note
This function is marked with the `@ignore_derivatives` macro from the
`ChainRulesCore` package, which means that all AutoDiff backends will ignore any
call to this function when computing gradients.
"""
function unit_vector(x::AbstractVector, i::Int)
    # Extract type of elements in the vector
    T = eltype(x)
    # Build unit vector
    return [j == i ? one(T) : zero(T) for j in 1:length(x)]
end # function

@doc raw"""
    unit_vector(x::AbstractMatrix, i::Int)

Create a unit vector of the same length as the number of rows in `x` with the
`i`-th element set to 1.

# Arguments
- `x::AbstractMatrix`: The matrix whose number of rows is used to determine the
  dimension of the unit vector.
- `i::Int`: The index of the element to be set to 1.

# Returns
- A unit vector of type `eltype(x)` and length equal to the number of rows in
  `x` with the `i`-th element set to 1.

# Description
This function creates a unit vector of the same length as the number of rows in
`x` with the `i`-th element set to 1. All other elements are set to 0. 
"""
function unit_vector(x::AbstractMatrix, i::Int)
    # Extract type of elements in the vector
    T = eltype(x)
    # Build unit vector
    return [j == i ? one(T) : zero(T) for j in axes(x, 1)]
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        unit_vectors(x::AbstractVecOrMat)

Create a vector or matrix of unit vectors based on the length or size of `x`.

# Arguments
- `x::AbstractVecOrMat`: The vector or matrix whose length or size is used to
  determine the dimension of the unit vectors.

# Returns
- A vector or matrix of unit vectors. Each unit vector has the same length as
  `x` and has a single `1` at the position corresponding to its index in the
  returned vector or matrix, with all other elements set to `0`.

# Description
This function creates a vector or matrix of unit vectors based on the length or
size of `x`. Each unit vector has the same length as `x` and has a single `1` at
the position corresponding to its index in the returned vector or matrix, with
all other elements set to `0`.

If `x` is a matrix, the function returns a matrix of unit vectors, where each
column is a unit vector.

# GPU Support
This function supports both CPU and GPU arrays.
"""
function unit_vectors(x::AbstractVecOrMat)
    return _unit_vectors(storage_type(x), x)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    unit_vectors(x::AbstractVector)

AbstractVector implementation of `unit_vectors`.
"""
function _unit_vectors(::Type, x::AbstractVector)
    return [unit_vector(x, i) for i in 1:length(x)]
end # function

@doc raw"""
    unit_vectors(x::CUDA.CuVector)

GPU AbstractVector implementation of `unit_vectors`.
"""
function _unit_vectors(::Type{T}, x::AbstractVector) where {T<:CUDA.CuArray}
    return [unit_vector(x, i) for i in 1:length(x)] |> Flux.gpu
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    unit_vectors(x::AbstractMatrix)

AbstractMatrix implementation of `unit_vectors`.
"""
function _unit_vectors(::Type, x::AbstractMatrix)
    vectors = [
        reduce(hcat, fill(unit_vector(x, i), size(x, 2)))
        for i in 1:size(x, 1)
    ]
    return vectors
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    unit_vectors(x::CUDA.CuMatrix)

GPU AbstractMatrix implementation of `unit_vectors`.
"""
function _unit_vectors(::Type{T}, x::CUDA.CuMatrix) where {T<:CUDA.CuArray}
    vectors = [
        reduce(hcat, fill(unit_vector(x, i), size(x, 2)))
        for i in 1:size(x, 1)
    ]
    return vectors |> Flux.gpu
end # function

# ==============================================================================
# Finite difference gradient computation
# ==============================================================================

@doc raw"""
    finite_difference_gradient(
        f::Function,
        x::AbstractVecOrMat;
        fdtype::Symbol=:central
    )

Compute the finite difference gradient of a function `f` at a point `x`.

# Arguments
- `f::Function`: The function for which the gradient is to be computed. This
  function must return a scalar value.
- `x::AbstractVecOrMat`: The point at which the gradient is to be computed. Can
  be a vector or a matrix. If a matrix, each column represents a point where the
  function f is to be evaluated and the derivative computed.

# Optional Keyword Arguments
- `fdtype::Symbol=:central`: The finite difference type. It can be either
  `:forward` or `:central`. Defaults to `:central`.

# Returns
- A vector or a matrix representing the gradient of `f` at `x`, depending on the
  input type of `x`.

# Description
This function computes the finite difference gradient of a function `f` at a
point `x`. The gradient is a vector or a matrix where the `i`-th element is the
partial derivative of `f` with respect to the `i`-th element of `x`.

The partial derivatives are computed using the forward or central difference
formula, depending on the `fdtype` argument:

- Forward difference formula: ∂f/∂xᵢ ≈ [f(x + ε * eᵢ) - f(x)] / ε
- Central difference formula: ∂f/∂xᵢ ≈ [f(x + ε * eᵢ) - f(x - ε * eᵢ)] / 2ε

where ε is the step size and eᵢ is the `i`-th unit vector.

# GPU Support
This function supports both CPU and GPU arrays.
"""
function finite_difference_gradient(
    f::Function,
    x::AbstractVecOrMat;
    fdtype::Symbol=:central,
)
    return _finite_difference_gradient(storage_type(x), f, x; fdtype=fdtype)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    finite_difference_gradient(
        f::Function,
        x::AbstractVecOrMat;
        fdtype::Symbol=:central
    )

CPU AbstractVecOrMat implementation of `finite_difference_gradient`.
"""
function _finite_difference_gradient(
    ::Type,
    f::Function,
    x::AbstractVecOrMat;
    fdtype::Symbol=:central,
)
    # Check that mode is either :forward or :central
    if !(fdtype in (:forward, :central))
        error("fdtype must be either :forward or :central")
    end # if

    # Check fdtype
    if fdtype == :forward
        # Define step size
        ε = √(eps(eltype(x)))
        # Generate unit vectors times step size for each element of x
        Δx = unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.(Ref(x) .+ Δx) .- f(x)) ./ ε
    else
        # Define step size
        ε = ∛(eps(eltype(x)))
        # Generate unit vectors times step size for each element of x
        Δx = unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.(Ref(x) .+ Δx) - f.(Ref(x) .- Δx)) ./ (2ε)
    end # if

    if typeof(x) <: AbstractVector
        return grad
    elseif typeof(x) <: AbstractMatrix
        return permutedims(reduce(hcat, grad), [2, 1])
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    finite_difference_gradient(
        f::Function,
        x::CUDA.CuArray;
        fdtype::Symbol=:central
    )

GPU AbstractVecOrMat implementation of `finite_difference_gradient`.
"""
function _finite_difference_gradient(
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
        Δx = unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.([x + δ for δ in Δx]) .- f(x)) ./ ε
    else
        # Define step size
        ε = ∛(eps(eltype(x)))
        # Generate unit vectors times step size for each element of x
        Δx = unit_vectors(x) .* ε
        # Compute the finite difference gradient for each element of x
        grad = (f.([x + δ for δ in Δx]) - f.([x - δ for δ in Δx])) ./ (2ε)
    end # if

    if typeof(x) <: AbstractVector
        return grad
    elseif typeof(x) <: AbstractMatrix
        return permutedims(reduce(hcat, grad), [2, 1])
    end # if
end # function

# ==============================================================================
# Define TaylorDiff gradient function
# ==============================================================================

@doc raw"""
        taylordiff_gradient(
                f::Function,
                x::AbstractVecOrMat
        )

Compute the gradient of a function `f` at a point `x` using Taylor series
differentiation.

# Arguments
- `f::Function`: The function for which the gradient is to be computed. This
  must be a scalar function.
- `x::AbstractVecOrMat`: The point at which the gradient is to be computed. Can
  be a vector or a matrix. If a matrix, each column represents a point where the
  function f is to be evaluated and the derivative computed.

# Returns
- A vector or a matrix representing the gradient of `f` at `x`, depending on the
  input type of `x`.

# Description
This function computes the gradient of a function `f` at a point `x` using
Taylor series differentiation. The gradient is a vector or a matrix where the
`i`-th element or column is the partial derivative of `f` with respect to the
`i`-th element of `x`.

The partial derivatives are computed using the TaylorDiff.derivative function.

# GPU Support
This function currently only supports CPU arrays.
"""
function taylordiff_gradient(
    f::Function,
    x::AbstractVecOrMat;
)
    return _taylordiff_gradient(storage_type(x), f, x)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    taylordiff_gradient(f::Function, x::AbstractVector)

CPU AbstractVector implementation of `taylordiff_gradient`.
"""
function _taylordiff_gradient(
    ::Type,
    f::Function,
    x::AbstractVector;
)
    # Compute the gradient for each element of x
    grad = TaylorDiff.derivative.(Ref(f), Ref(x), unit_vectors(x), Ref(1))

    return grad
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    taylordiff_gradient(f::Function, x::AbstractMatrix)

CPU AbstractMatrix implementation of `taylordiff_gradient`.
"""
function _taylordiff_gradient(
    ::Type,
    f::Function,
    x::AbstractMatrix;
)
    # Compute the gradient for each column of x
    grad = permutedims(
        reduce(
            hcat,
            TaylorDiff.derivative.(
                Ref(f), Ref(x), unit_vectors(x[:, 1]), Ref(1)
            )
        ),
        [2, 1]
    )

    return grad
end # function

# function taylordiff_gradient(
#     f::Function,
#     x::CUDA.CuMatrix;
# )
#     # Compute the gradient for each column of x
#     grad = permutedims(
#         reduce(
#             hcat,
#             TaylorDiff.derivative.(
#                 [(f, x, u, 1) for u in unit_vectors(x[:, 1])]...
#             )
#         ),
#         [2, 1]
#     )
#     return grad
# end # function