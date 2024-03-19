# ==============================================================================
# Encoder Riemmanian metric
# ==============================================================================

# Reference
# > Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational 
# > Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).

@doc raw"""
    `exponential_mahalanobis_kernel(x, y, Σ; ρ=1.0f0)`

Compute the exponential Mahalanobis Kernel between two matrices `x` and `y`,
defined as 

ω(x, y) = exp(-(x - y)ᵀ Σ⁻¹ (x - y) / ρ²)

# Arguments
- `x::AbstractVector{Float32}`: First input vector for the kernel.
- `y::AbstractVector{Float32}`: Second input vector for the kernel.
- `Σ::AbstractMatrix{Float32}`: The covariance matrix used in the Mahalanobis
  distance.

# Keyword Arguments
- `ρ::Float32=1.0f0`: Kernel width parameter. Larger ρ values lead to a wider
  spread of the kernel.

# Returns
- `k::AbstractMatrix{Float32}`: Kernel matrix 

# Examples
```julia
x = rand(10, 2) 
y = rand(20, 2)
Σ = I(2)
K = exponential_mahalanobis_kernel(x, y, Σ) # 10x20 kernel matrix
```

# Reference
> Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational 
> Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
"""
function exponential_mahalanobis_kernel(
    x::AbstractVector,
    y::AbstractVector,
    Σ::AbstractMatrix;
    ρ::Number=1.0f0,
)
    # return Gaussian kernel
    return exp(-Distances.sqmahalanobis(x, y, Σ) / ρ^2)
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    latent_kmedoids(encoder::AbstractVariationalEncoder,
    data::AbstractMatrix{Float32}, k::Int; distance::Distances.Metric =
    Distances.Euclidean()) -> Vector{Int}

Perform k-medoids clustering on the latent representations of data points
encoded by a Variational Autoencoder (VAE).

This function maps the input data to the latent space using the provided
encoder, computes pairwise distances between the latent representations using
the specified metric, and applies the k-medoids algorithm to partition the data
into k clusters. The function returns a vector of indices corresponding to the
medoids of each cluster.

# Arguments
- `encoder::AbstractVariationalEncoder`: An encoder from a VAE that outputs the
  mean (µ) and log variance (logσ) of the latent space distribution for input
  data.
- `data::AbstractMatrix{Float32}`: The input data matrix, where each column
  represents a data point.
- `k::Int`: The number of clusters to form.

# Optional Keyword Arguments
- `distance::Distances.Metric`: The distance metric to use when computing
  pairwise distances between latent representations, defaulting to
  Distances.Euclidean().

# Returns
- `Vector{Int}`: A vector of indices corresponding to the medoids of the clusters
  in the latent space.

# Example
```julia
# Assuming `encoder` is an instance of `AbstractVariationalEncoder` and `data` 
# is a matrix of Float32s:
medoid_indices = latent_kmedoids(encoder, data, 5)
# `medoid_indices` will hold the indices of the medoids after clustering into 
# 5 clusters.
```
"""
function latent_kmedoids(
    encoder::AbstractVariationalEncoder,
    data::AbstractMatrix{Float32},
    k::Int;
    distance::Distances.Metric=Distances.Euclidean()
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, _ = encoder(data)

    # Compute pairwise distances between elements
    dist = Distances.pairwise(distance, latent_µ, dims=2)

    # Compute kmedoids clustering with Clustering.jl and return indexes.
    return kmedoids(dist, k).medoids
end # function

# ------------------------------------------------------------------------------ 

"""
    encoder_metric_builder(encoder::JointLogEncoder, 
    data::AbstractMatrix{Float32}; λ::Number=0.0001f0, 
    τ::Number=eps(Float32)) -> Function

Build a metric function G(z) using a trained variational autoencoder (VAE)
model's encoder.

The metric G(z) is defined as a weighted sum of precision matrices Σ⁻¹, each
associated with a data point in the latent space, and a regularization term. The
weights are determined by the exponential Mahalanobis kernel between z and each
data point's latent mean.

# Arguments
- `encoder::JointLogEncoder`: A VAE model encoder that maps data to the latent
  space.
- `data`: The dataset in the form of a matrix, with each column representing a
  data point.

# Optional Keyword Arguments
- `ρ::Union{Nothing, Float32}=nothing`: The maximum Euclidean distance between
  any two closest neighbors in the latent space. If nothing, it will be computed
  from the data. Defaults to nothing.
- `λ::Number=0.0001f0`: A regularization parameter that controls the
  strength of the regularization term. Defaults to 0.0001.
- `τ::Number=eps(Float32)`: A small positive value to ensure numerical
  stability. Defaults to machine epsilon for Float32.

# Returns
- A function G that takes a vector z and returns the metric matrix.

## Mathematical Definition

For each data point i, the encoder provides the mean μᵢ and log variance logσᵢ
in the latent space. The precision matrix for each data point is given by 

Σ⁻¹ᵢ = Diagonal(exp.(-logσᵢ)). 

The metric is then constructed as follows:

G(z) = ∑ᵢ (Σ⁻¹ᵢ .* exp.(-||z - μᵢ||²_Σ⁻¹ᵢ / ρ²)) + λ * exp(-τ * ||z||²) * Iₙ

where:
- ρ is the maximum Euclidean distance between any two closest neighbors in the
  latent space.
- Iₙ is the identity matrix of size n, where n is the number of dimensions in
  the latent space.
- ||⋅||²_Σ⁻¹ᵢ denotes the squared Mahalanobis distance with respect to Σ⁻¹ᵢ.

# Examples
```julia
# Assuming `encoder` is a pre-trained JointLogEncoder and `data` is your 
# dataset:
G = encoder_metric_builder(encoder, data)
# Compute the metric for a given latent vector
metric_matrix = G(some_latent_vector)  
```

# Reference
> Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational
> Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
"""
function encoder_metric_builder(
    encoder::JointLogEncoder,
    data::AbstractMatrix{Float32};
    ρ::Union{Nothing,Float32}=nothing,
    λ::Number=0.0001f0,
    τ::Number=eps(Float32),
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, latent_logσ = encoder(data)

    # Generate all inverse covariance matrices (Σ⁻¹) from logσ
    Σ_inv_array = [
        LinearAlgebra.Diagonal(exp.(-latent_logσ[:, i]))
        for i in 1:size(latent_logσ, 2)
    ]

    # Compute Euclidean pairwise distance between points in latent space
    latent_euclid_dist = Distances.pairwise(
        Distances.Euclidean(), latent_µ, dims=2
    )

    # Compute ρ if not provided
    if typeof(ρ) <: Nothing
        # Define ρ as the maximum distance between two closest neighbors
        ρ = maximum([
            minimum(distances[distances.>0])
            for distances in eachrow(latent_euclid_dist)
        ])
    end # if

    # Identity matrix for regularization term, should match the dimensionality
    # of z
    I_d = LinearAlgebra.I(size(latent_µ, 1))

    # Build metric G(z) as a function
    function G(z::AbstractVector{Float32})
        # Initialize metric to zero matrix of appropriate size
        metric = zeros(Float32, size(latent_µ, 1), size(latent_µ, 1))

        # Accumulate weighted Σ_inv matrices, weighted by the kernel
        for i in 1:size(latent_µ, 2)
            # Compute ω terms with exponential mahalanobis kernel
            ω = exponential_mahalanobis_kernel(
                z, latent_µ[:, i], Σ_inv_array[i]; ρ=ρ
            )
            # Update value of metric as the precision metric multiplied
            # (elementwise for some reason) by the resulting value of the kernel
            metric .+= Σ_inv_array[i] .* ω
        end # for

        # Regularization term: λ * exp(-τ||z||²) * I_d
        regularization = λ * exp(-τ * sum(z .^ 2)) .* I_d

        return metric + regularization
    end # function

    return G
end # function

"""
    encoder_metric_builder(
        encoder::JointLogEncoder,
        data::AbstractMatrix{Float32},
        kmedoids_idx::Vector{<:Int};
        ρ::Union{Nothing, Float32}=nothing,
        λ::Number=0.0001f0,
        τ::Number=eps(Float32)
    ) -> Function

Build a metric function G(z) using a trained variational autoencoder (VAE)
model's encoder, sub-sampling the data points used to build the metric based on
k-medoids indices.

The metric G(z) is defined as a weighted sum of precision matrices Σ⁻¹, each
associated with a subset of data points in the latent space determined by
k-medoids clustering, and a regularization term. The weights are determined by
the exponential Mahalanobis kernel between z and each data point's latent mean
from the sub-sampled set.

# Arguments
- `encoder::JointLogEncoder`: A VAE model encoder that maps data to the latent
  space.
- `data`: The dataset in the form of a matrix, with each column representing a
  data point.
- `kmedoids_idx::Vector{<:Int}`: Indices of data points chosen as medoids from
  k-medoids clustering.

# Optional Keyword Arguments
- `ρ::Union{Nothing, Float32}=nothing`: The maximum Euclidean distance between
  any two closest neighbors in the latent space. If nothing, it will be computed
  from the data. Defaults to nothing.
- `λ::Number=0.0001f0`: A regularization parameter that controls the
  strength of the regularization term. Defaults to 0.0001.
- `τ::Number=eps(Float32)`: A small positive value to ensure numerical
  stability. Defaults to machine epsilon for Float32.

# Returns
- A function G that takes a vector z and returns the metric matrix.

## Mathematical Definition

For each data point i, the encoder provides the mean μᵢ and log variance logσᵢ
in the latent space. The precision matrix for each data point is given by 

Σ⁻¹ᵢ = Diagonal(exp.(-logσᵢ)). 

The metric is then constructed as follows:

G(z) = ∑ᵢ (Σ⁻¹ᵢ .* exp.(-||z - μᵢ||²_Σ⁻¹ᵢ / ρ²)) + λ * exp(-τ * ||z||²) * Iₙ

where:
- ρ is the maximum Euclidean distance between any two closest neighbors in the
  latent space.
- Iₙ is the identity matrix of size n, where n is the number of dimensions in
  the latent space.
- ||⋅||²_Σ⁻¹ᵢ denotes the squared Mahalanobis distance with respect to Σ⁻¹ᵢ.

# Examples
```julia
# Assuming `encoder` is a pre-trained JointLogEncoder, `data` is your dataset,
# and `kmedoids_idx` contains indices from a k-medoids clustering:
G = encoder_metric_builder(encoder, data, kmedoids_idx)
# Compute the metric for a given latent vector
metric_matrix = G(some_latent_vector)
```

# Reference
> Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational
> Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
"""
function encoder_metric_builder(
    encoder::JointLogEncoder,
    data::AbstractMatrix{Float32},
    kmedoids_idx::Vector{<:Int};
    ρ::Union{Nothing,Float32}=nothing,
    λ::Number=0.0001f0,
    τ::Number=eps(Float32),
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, latent_logσ = encoder(data[:, kmedoids_idx])

    # Generate all inverse covariance matrices (Σ⁻¹) from logσ
    Σ_inv_array = [
        LinearAlgebra.Diagonal(exp.(-latent_logσ[:, i]))
        for i in 1:size(latent_logσ, 2)
    ]

    # Compute ρ if not provided
    if typeof(ρ) <: Nothing
        # Compute Euclidean pairwise distance between points in latent space
        latent_euclid_dist = Distances.pairwise(
            Distances.Euclidean(), latent_µ, dims=2
        )
        # Define ρ as the maximum distance between two closest neighbors
        ρ = maximum([
            minimum(distances[distances.>0])
            for distances in eachrow(latent_euclid_dist)
        ])
    end # if

    # Identity matrix for regularization term, should match the dimensionality
    # of z
    I_d = LinearAlgebra.I(size(latent_µ, 1))

    # Build metric G(z) as a function
    function G(z::AbstractVector{Float32})
        # Initialize metric to zero matrix of appropriate size
        metric = zeros(Float32, size(latent_µ, 1), size(latent_µ, 1))

        # Accumulate weighted Σ_inv matrices, weighted by the kernel
        for i in 1:size(latent_µ, 2)
            # Compute ω terms with exponential mahalanobis kernel
            ω = exponential_mahalanobis_kernel(
                z, latent_µ[:, i], Σ_inv_array[i]; ρ=ρ
            )
            # Update value of metric as the precision metric multiplied
            # (elementwise for some reason) by the resulting value of the kernel
            metric .+= Σ_inv_array[i] .* ω
        end # for

        # Regularization term: λ * exp(-τ||z||²) * I_d
        regularization = λ * exp(-τ * sum(z .^ 2)) .* I_d

        return metric + regularization
    end # function

    return G
end # function