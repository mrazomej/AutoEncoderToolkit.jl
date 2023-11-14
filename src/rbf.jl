# Import ML libraries
import Flux
import Zygote

# Import basic math
import Distances
import LinearAlgebra
import StatsBase

# Import Clustering algorithms
import Clustering

using ..AutoEncode: AbstractVariationalAutoEncoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder, JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder, VAE, rbfVAE

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# > Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
# > Curvature of Deep Generative Models. Preprint at
# > http://arxiv.org/abs/1710.11379 (2021).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ------------------------------------------------------------------------------ 

"""
    latent_rbf_centers(
        vae::VAE, data::AbstractMatrix{Float32}, num_centers::Int
    ) -> Matrix{Float32}

Perform k-means clustering in the latent space of a trained Variational
Autoencoder (VAE) to determine the centers for a Radial Basis Function (RBF)
network.

This function processes an input data matrix through the encoder of the VAE to
obtain latent space representations, and then applies k-means clustering to find
a specified number of centers within this latent space. These centers are
crucial for defining the RBFs used in modeling the data-dependent variance of
the VAE's decoder.

# Arguments
- `encoder::JointLogEncoder`: An instance of a trained encoder.
- `data::AbstractMatrix{Float32}`: The input data matrix, where each column
  represents a single observation in the original feature space.
- `num_centers::Int`: The desired number of centers to find in the latent space
  for RBF network.

# Optional Keyword Arguments
- `assignment::Bool=true`: Boolean indicating if the assignment of the
  corresponding cluster should be returned for each element in data.
  Default=true

# Returns
- `centers::Matrix{Float32}`: A matrix where each column is a center of an RBF
  in the latent space.
- `assign::Vector{Int}`: If `assignment=true`. Vector with cluster assignment
  for each element in data.

# Example
```julia
vae = VAE(...) # A trained VAE instance
data = rand(Float32, input_dim, num_samples) # Example data matrix
num_centers = 10 # Number of desired RBF centers
rbf_centers = latent_rbf_centers(vae.encoder, data, num_centers)
```
"""
function latent_rbf_centers(
    encoder::JointLogEncoder,
    data::AbstractMatrix{Float32},
    num_centers::Int;
    assigment::Bool=true
)
    # Map the data to the latent space
    latent_µ, _ = encoder(data)

    # Perform k-means clustering on the latent space means and return the
    # centers
    if assigment
        # Compute clustering
        clustering = Clustering.kmeans(latent_µ, num_centers)
        return (clustering.centers, Clustering.assignments(clustering))
    else
        return Clustering.kmeans(latent_µ, num_centers).centers
    end # if
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
calculate_bandwidths(latent_space::AbstractMatrix{Float32},
                     centers::AbstractMatrix{Float32},
                     a::Float32)::Vector{Float32}

Calculates the bandwidths λₖ for the Radial Basis Function (RBF) kernels. These
bandwidths determine the extent to which each kernel influences the latent space
and are crucial for modeling variance that respects the geometry of the latent
manifold. The bandwidths are computed using the equation:


λₖ = (1/2) (a 1 / |Cₖ| ∑_(zⱼ∈Cₖ) ||zⱼ -  cₖ||₂)⁻²

where:
- λₖ is the bandwidth for the k-th RBF kernel.
- a is a hyper-parameter that controls the curvature of the Riemannian metric.
- Cₖ represents the set of points that are closest to the k-th center, cₖ.
- zⱼ represents a data point in the latent space that belongs to the cluster Cₖ.
- ||zⱼ - cₖ||₂ is the Euclidean distance between zⱼ and cₖ.

# Arguments
- `latent_space::AbstractMatrix{Float32}`: The latent space representations of
  the data.
- `centers::AbstractMatrix{Float32}`: The centers of the RBFs obtained from
  k-means clustering.
- `assignments::AbstractVector{<:Int}`: Cluster assignment for each of the
  elements in `latent_space`.
- `a::Float32`: The hyper-parameter that controls the curvature of the
  Riemannian metric.

# Returns
- `λ::Vector{Float32}`: The calculated bandwidths for each RBF kernel.

# Reference
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function calculate_bandwidths(
    latent_space::AbstractMatrix{Float32},
    centers::AbstractMatrix{Float32},
    assignments::AbstractVector{<:Int},
    a::Float32
)::Vector{Float32}
    # Count number of centers
    num_centers = size(centers, 2)

    # Initialize array to save bandwidths
    bandwidths = Vector{Float32}(undef, num_centers)

    # Loop through centers
    for (k, cₖ) in enumerate(eachcol(centers))
        # Find all the points in latent_space that are closest to center cₖ
        cluster_points = @view latent_space[:, assignments.==k]

        # Calculate the average squared distance from points in Cₖ to center cₖ
        if isempty(cluster_points)
            # Avoid division by zero if a cluster has no points
            avg_sq_dist = 0.0
        else
            # Compute mean distance from center. This is done using the pairwise
            # distance function, which is more efficient. The cₖ array is
            # reshaped as a matrix to be able to use the pairwise function.
            avg_sq_dist = StatsBase.mean(
                Distances.pairwise(
                    Distances.SqEuclidean(),
                    cluster_points,
                    reshape(cₖ, :, 1),
                    dims=2
                )
            )
        end # if

        # Calculate the bandwidth using the formula
        bandwidths[k] = 0.5 * (a * avg_sq_dist)^-2
    end # for

    return bandwidths
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    RBFlayer(centers, bandwidths, weights, bias, activation)

A mutable struct representing a layer in a Radial Basis Function (RBF) network.
This layer can serve as a decoder in a variational autoencoder architecture,
where the RBFs capture the variance structure of the data in the latent space.

# Fields
- `centers::AbstractMatrix{Float32}`: A matrix where each column is the center
  of an RBF.
- `bandwidths::AbstractVector{Float32}`: A vector containing the bandwidths for
  each RBF kernel.
- `weights::AbstractMatrix{Float32}`: A matrix of weights for the linear
  combination of RBF activations.
- `bias::AbstractVector{Float32}`: A vector of bias terms added to the weighted
  sum of activations.

# Reference
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
mutable struct RBFlayer
    centers::AbstractMatrix{Float32}
    bandwidths::AbstractVector{Float32}
    weights::AbstractMatrix{Float32}
    bias::AbstractVector{Float32}
end # struct

# ------------------------------------------------------------------------------ 

@doc raw"""
    (rbf::RBFlayer)(z::AbstractVector{Float32})

Calculate the output of the RBF layer for a single input vector `z`. The output
is computed as the weighted sum of RBF activations plus a bias term.

# Arguments
- `z::AbstractVector{Float32}`: A single input vector from the latent space.

# Returns
- The output of the RBF layer as a scalar value.

# Reference
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function (rbf::RBFlayer)(z::AbstractVector{Float32})
    # Calculate the RBF activations for each center
    activations = exp.(
        -rbf.bandwidths .* vec(sum((rbf.centers .- z) .^ 2, dims=1))
    )

    return rbf.weights * activations + rbf.bias
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    (rbf::RBFlayer)(z::AbstractMatrix{Float32})

Calculate the outputs of the RBF layer for a batch of input vectors `z`. The
function processes each column of `z` as a separate input and returns the
corresponding outputs.

# Arguments
- `z::AbstractMatrix{Float32}`: A matrix of input vectors from the latent space,
  where each column is a separate input.

# Returns
- A matrix containing the outputs of the RBF layer for each input.

# Reference
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function (rbf::RBFlayer)(z::AbstractMatrix{Float32})
    return reduce(hcat, rbf.(eachcol(z)))
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    (rbf::RBFlayer)(z::AbstractArray{Float32,3})

Calculate the outputs of the RBF layer for a batch of input vectors `z`. The
function processes each column of `z` as a separate input and returns the
corresponding outputs.

# Arguments
- `z::AbstractArray{Float32,3}`: A 3D tensor of input vectors from the latent
  space, where each column in dimension 2 is a data point, and each slice in
  dimension 3 is a sample from the prior.

# Returns
- A 3D tensor containing the outputs of the RBF layer for each input.

# Reference
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function (rbf::RBFlayer)(z::AbstractVector{Float32,3})
    return cat(eachslice(z, dims=3); dims=3)
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    `RBFVAE <: AbstractVariationalAutoEncoder`

Structure encapsulating a Variational Autoencoder (VAE) with an extra Radial
Basis Function (RBF) network used to compute the variance of the decoder output.

The implementation is inspired by the publication

> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).

# Fields
- `vae::VAE`: The core variational autoencoder, consisting of an encoder that
  maps input data into a latent space representation, and a decoder that
  attempts to reconstruct the input from the latent representation.
- `rbf::RBFlayer`: An RBFlayer struct defining the RBF network.

# Usage
The `RBFVAE` struct is utilized in a similar manner to a standard VAE, with the
added capability of approximating the decoder variance via an RBF network. This
network is usually trained after the vanilla VAE has been already trained to
force the variance to bound the data manifold.
"""
mutable struct RBFVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder},
    R::RBFlayer
} <: AbstractVariationalAutoEncoder
    vae::V
    rbf::M
end # struct

# ------------------------------------------------------------------------------ 

function (rbfvae::RBFVAE)(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
)
    # Check if latent variables and mutual information should be returned
    if latent
        outputs = rbfvae.vae(x, prior; latent=latent, n_samples=n_samples)

        # Add RBF output to dictionary
        outputs[:rbf] = rbfvae.rbf(outputs[:z])

        return outputs
    else
        # or return reconstructed data from decoder
        return rbfvae.vae(x, prior; latent=latent, n_samples=n_samples)
    end # if
end # function