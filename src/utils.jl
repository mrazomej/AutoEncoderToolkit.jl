# Import library to find nearest neighbors
import NearestNeighbors

# Import library for random sampling
import StatsBase

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
"""
function cycle_anneal(
    epoch::Int,
    n_epoch::Int,
    n_cycles::Int;
    frac::T=0.5,
    βmax::T=1.0,
    βmin::T=0.0
) where {T<:AbstractFloat}
    # Define variable τ that will serve to define the value of β
    τ = mod(epoch - 1, ceil(n_epoch / n_cycles)) / (n_epoch / n_cycles)

    # Compute and return the value of β
    if τ ≤ frac
        return convert(T, (βmax - βmin) * τ / frac + βmin)
    else
        return convert(T, βmax)
    end # if
end # function

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
        return @view data[:, [idx_primary; idx_secondary]]
    else
        return [idx_primary; idx_secondary]
    end # if
end # function