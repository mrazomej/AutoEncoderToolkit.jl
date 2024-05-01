# Import ML libraries
import Flux
import Zygote

# Import basic math
import Distances
import LinearAlgebra
import StatsBase
import Distributions

# Import Clustering algorithms
import Clustering

using ..AutoEncoderToolkit: AbstractVariationalAutoEncoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder, JointGaussianLogEncoder, SimpleGaussianDecoder, JointGaussianLogDecoder,
    SplitGaussianLogDecoder, JointGaussianDecoder, SplitGaussianDecoder, VAE

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# > Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
# > Curvature of Deep Generative Models. Preprint at
# > http://arxiv.org/abs/1710.11379 (2021).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ------------------------------------------------------------------------------ 

"""
    latent_rbf_centers(
        vae::VAE, data::AbstractMatrix{Float32}, n_centers::Int
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
- `encoder::JointGaussianLogEncoder`: An instance of a trained encoder.
- `data::AbstractMatrix{Float32}`: The input data matrix, where each column
  represents a single observation in the original feature space.
- `n_centers::Int`: The desired number of centers to find in the latent space
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
n_centers = 10 # Number of desired RBF centers
rbf_centers = latent_rbf_centers(vae.encoder, data, n_centers)
```
"""
function latent_rbf_centers(
    encoder::JointGaussianLogEncoder,
    data::AbstractMatrix{Float32},
    n_centers::Int;
    assigment::Bool=true
)
    # Map the data to the latent space
    latent_µ, _ = encoder(data)

    # Perform k-means clustering on the latent space means and return the
    # centers
    if assigment
        # Compute clustering
        clustering = Clustering.kmeans(latent_µ, n_centers)
        return (clustering.centers, Clustering.assignments(clustering))
    else
        return Clustering.kmeans(latent_µ, n_centers).centers
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
- `a::AbstractFloat`: The hyper-parameter that controls the curvature of the
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
    a::AbstractFloat
)::Vector{Float32}
    # Count number of centers
    n_centers = size(centers, 2)

    # Initialize array to save bandwidths
    bandwidths = Vector{Float32}(undef, n_centers)

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

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training.
# In this case, we will only train the weights. Therefore, we indicate to
# @functor the fields that can be trained.
Flux.@functor RBFlayer (weights,)

# ------------------------------------------------------------------------------ 

@doc raw"""
    RBFlayer(
        vae::VAE{<:JointGaussianLogEncoder,<:Union{JointGaussianLogDecoder,SplitGaussianLogDecoder}},
        x::AbstractMatrix{Float32},
        n_centers::Int,
        a::AbstractFloat;
        bias::Union{Nothing, <:AbstractVector{Float32}}=nothing,
        init::Function=Flux.glorot_uniform
    ) -> RBFlayer

Constructs and initializes an `RBFlayer` struct, which serves as the decoder
part of a Variational Autoencoder (VAE) utilizing Radial Basis Function (RBF)
networks to model the variance structure of the latent space.

This function maps the input data `x` through the VAE's encoder to obtain latent
space representations. It then performs k-means clustering to determine the
centers for the RBF network and calculates the corresponding bandwidths.

# Arguments
- `vae::VAE`: A VAE model with a `JointGaussianLogEncoder` encoder and either a
  `JointGaussianLogDecoder` or `SplitGaussianLogDecoder` decoder.
- `x::AbstractMatrix{Float32}`: The input data matrix, where each column
  represents a single observation in the original feature space.
- `n_centers::Int`: The number of RBF centers to find in the latent space.
- `a::AbstractFloat`: The hyper-parameter controlling the curvature of the
  Riemannian metric in the latent space.

## Optional Keyword Arguments
- `bias::Union{Nothing, <:AbstractVector{Float32}}`: Optionally provide a bias
  vector for the RBF layer. If `nothing`, the bias is initialized based on the
  encoder's log variance.
- `init::Function`: The initialization function used for the RBF weights,
  defaults to `Flux.glorot_uniform`.

# Returns
- An initialized `RBFlayer` struct ready to be used as part of a VAE.

# Examples
```julia
vae_model = VAE(...) # A pre-defined VAE model
input_data = rand(Float32, input_dim, num_samples) # Sample data matrix
number_of_centers = 10 # Desired number of RBF centers
rbf_layer = RBFlayer(vae_model, input_data, number_of_centers, 0.1)
```

# Notes

- The function uses the VAE's encoder to project data into the latent space and
  uses this projection to initialize the RBF layer's centers via k-means
  clustering.
- The `bandwidths` for the RBF kernels are calculated using the latent space
  representations and the determined centers.
- If `bias` is not provided, it is calculated based on the mean of the encoder's
  log variance, scaled by a factor of 1000.
- The `weights` of the RBF network are initialized using the init function, with
  the shape `(input_dim, n_centers)`.
"""
function RBFlayer(
    vae::VAE{JointGaussianLogEncoder,D},
    x::AbstractMatrix{Float32},
    n_centers::Int,
    a::AbstractFloat;
    bias::Union{Nothing,<:AbstractVector{Float32}}=nothing,
    init::Function=Flux.glorot_uniform
) where {D<:Union{JointGaussianLogDecoder,SplitGaussianLogDecoder}}
    # Map data to latent space
    encoder_µ, encoder_logσ = vae.encoder(x)

    # Calculate latent space centers
    centers, assignments = latent_rbf_centers(vae.encoder, x, n_centers)

    # Calculate bandwidths for RBF network
    λs = calculate_bandwidths(encoder_µ, centers, assignments, a)

    # Compute bias values if necessary. Note: This is the option presented by
    # the original authors in
    # https://github.com/georgiosarvanitidis/geometric_ml/blob/master/python/example2.py#L169
    if typeof(bias) <: Nothing
        σ_rbf = 1000 * StatsBase.mean(encoder_logσ)
        bias = repeat([1 / (σ_rbf^2)], size(x, 1))
    end # if

    # Define initial weights using the provided init. This matrix should be
    # D × n_centers
    weights = init(size(x, 1), n_centers)

    return RBFlayer(centers, λs, weights, bias)
end # function

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
function (rbf::RBFlayer)(z::AbstractArray{Float32,3})
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

# Note
An RBFVAE can be defined with any encoder and decoder. However, to properly
train the RBF network, the procedure assumes that the decoder includes learning
the reconstruction variance. In that sense, all the RBF network does is to build
a "cage" around the data manifold. Thus, we recommend avoiding `SimpleGaussianDecoder`
as the decoders.
"""
mutable struct RBFVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder},
    R<:RBFlayer
} <: AbstractVariationalAutoEncoder
    vae::V
    rbf::R
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training In
# this case, we will only train the RBF network as the VAE is assumed to be
# pre-trained. Therefore, we indicate to @functor the fields that can be
# trained.
Flux.@functor RBFVAE (rbf,)

# ------------------------------------------------------------------------------ 

@doc raw"""
    (rbfvae::RBFVAE)(
        x::AbstractVecOrMat{Float32}, 
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
        latent::Bool=false, 
        n_samples::Int=1
    )

Processes the input data `x` through an RBFVAE, which is a Variational
Autoencoder with a Radial Basis Function (RBF) network as part of its decoder.
This VAE variant aims to capture the complex variance structure in the latent
space through the RBF network, enhancing the model's capacity to represent data
manifolds.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be processed. Can be a vector or a
  matrix where each column represents a separate data sample.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution for the
  latent space during the reparametrization trick. Defaults to a standard normal
  distribution.
- `latent::Bool`: If `true`, returns a dictionary containing the latent
  variables, RBF network outputs, and the reconstructed data. Defaults to
  `false`.
- `n_samples::Int=1`: The number of samples to draw from the latent distribution
  using the reparametrization trick.

# Returns
- If `latent=false`: A tuple containing the reconstructed data from the standard
  decoder and the RBF network, providing both the mean reconstruction and the
  RBF-adjusted output.
- If `latent=true`: A dictionary with keys `:encoder_µ`, `:encoder_(log)σ`,
  `:z`, `:decoder_µ`, `:decoder_(log)σ`, and `:rbf`, containing the mean and log
  variance from the encoder, the sampled latent variables, the mean
  reconstruction from the decoder, and the output of the RBF network,
  respectively.

# Description
The function first uses the encoder to map the input `x` to a latent
distribution, characterized by its mean and log variance. It then samples from
this distribution using the reparametrization trick. The sampled latent vectors
are passed through the decoder to reconstruct the data and through the RBF
network to obtain the RBF-adjusted output. If the `latent` flag is set to
`true`, the function also returns the latent variables and the RBF network's
outputs.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the RBFVAE. The `rbf` output provides the data-dependent variance
estimation which is key to differential geometry operations in the latent space.
"""
function (rbfvae::RBFVAE)(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
)
    # Run inputs through VAE and save all outputs regardless of latent
    outputs = rbfvae.vae(x, prior; latent=true, n_samples=n_samples)
    # Check if latent variables and mutual information should be returned
    if latent
        # Add RBF output to dictionary
        outputs[:rbf] = rbfvae.rbf(outputs[:z])

        return outputs
    else
        # or return reconstructed data from decoder
        return outputs[:decoder_µ], rbfvae.rbf(outputs[:z])
    end # if
end # function

# ==============================================================================
# Loss RBFVAE.VAE{JointGaussianLogEncoder,Union{JointGaussianLogDecoder,SplitGaussianLogDecoder}}
# ==============================================================================

@doc raw"""
    `loss(vae, rbf, x; n_samples=1, regularization=nothing, reg_strength=1.0f0)`

Computes the loss for a Variational Autoencoder (VAE) with a Radial Basis
Function (RBF) network as the decoder, by averaging over `n_samples` latent
space samples.

The loss function combines the reconstruction loss based on the estimated log
variance from the VAE's decoder and the RBF network's output, and possibly
includes a regularization term. The loss is computed as:

loss = MSE(2 × decoder_logσ, -log(rbf_outputs_safe)) + reg_strength × reg_term

Where:
- `decoder_logσ` is the log standard deviation of the reconstructed data from
  the decoder.
- `rbf_outputs_safe` is the RBF network's output for the latent representations,
  clamped to avoid log of non-positive values.

# Arguments
- `vae::VAE`: A VAE model with an encoder and a decoder network.
- `rbf::RBFlayer`: An RBF layer representing the decoder of the VAE.
- `x::AbstractVecOrMat{Float32}`: Input data. Can be a vector or a matrix where
  each column represents an observation.

# Optional Keyword Arguments
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the RBF outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed average loss value for the input `x`, including the
  mean squared error between the estimated and the RBF-adjusted log variances
  and possible regularization terms.

# Note
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- The `rbf_outputs_safe` is the output of the RBF network adjusted to ensure
  numerical stability when taking the logarithm.
- The RBF network aims to model the variance structure in the latent space,
  enhancing the VAE's capacity to represent complex data manifolds.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,D},
    rbf::RBFlayer,
    x::AbstractVecOrMat{Float32};
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {D<:Union{JointGaussianLogDecoder,SplitGaussianLogDecoder}}
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract log variance from VAE outputs. Note: Factor of 2 is to transform
    # logσ to logvar
    logvar_x̂ = 2 * vae_outputs[:decoder_logσ]

    # Run latent space outputs through RBF network to estimate variances
    rbf_outputs = rbf(vae_outputs[:z])
    # Ensure no zero or negative values before taking log
    rbf_outputs_safe = clamp.(rbf_outputs, eps(Float32), Inf)
    # Compute the log variance from the RBF outputs. Note var(x) = 1 / RBF(x)
    logvar_rbf = -log.(rbf_outputs_safe)

    # Compute regularization term if a regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(rbf_outputs) : 0.0f0

    # Mean squared error loss for log variances 
    mse_loss = Flux.mse(logvar_x̂, logvar_rbf)
    # Include regularization
    total_loss = mse_loss + reg_strength * reg_term

    return total_loss
end # function

# ==============================================================================
# Loss RBFVAE.VAE{JointGaussianLogEncoder,Union{JointGaussianDecoder,SplitGaussianDecoder}}
# ==============================================================================

@doc raw"""
    `loss(vae, rbf, x; n_samples=1, regularization=nothing, reg_strength=1.0f0)`

Computes the loss for a Variational Autoencoder (VAE) with a Radial Basis
Function (RBF) network as the decoder, by averaging over `n_samples` latent
space samples.

The loss function combines the reconstruction loss based on the estimated log
variance from the VAE's decoder and the RBF network's output, and possibly
includes a regularization term. The loss is computed as:

loss = MSE(2 × decoder_logσ, -log(rbf_outputs_safe)) + reg_strength × reg_term

Where:
- `decoder_logσ` is the log standard deviation of the reconstructed data from
  the decoder.
- `rbf_outputs_safe` is the RBF network's output for the latent representations,
  clamped to avoid log of non-positive values.

# Arguments
- `vae::VAE`: A VAE model with an encoder and a decoder network.
- `rbf::RBFlayer`: An RBF layer representing the decoder of the VAE.
- `x::AbstractVecOrMat{Float32}`: Input data. Can be a vector or a matrix where
  each column represents an observation.

# Optional Keyword Arguments
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the RBF outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed average loss value for the input `x`, including the
  mean squared error between the estimated and the RBF-adjusted log variances
  and possible regularization terms.

# Note
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- The `rbf_outputs_safe` is the output of the RBF network adjusted to ensure
  numerical stability when taking the logarithm.
- The RBF network aims to model the variance structure in the latent space,
  enhancing the VAE's capacity to represent complex data manifolds.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,D},
    rbf::RBFlayer,
    x::AbstractVecOrMat{Float32};
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {D<:Union{JointGaussianDecoder,SplitGaussianDecoder}}
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract log variance from VAE outputs. Note: Factor of 2 is to transform
    # logσ to logvar
    logvar_x̂ = log.(vae_outputs[:decoder_σ] .^ 2)

    # Run latent space outputs through RBF network to estimate variances
    rbf_outputs = rbf(vae_outputs[:z])
    # Ensure no zero or negative values before taking log
    rbf_outputs_safe = clamp.(rbf_outputs, eps(Float32), Inf)
    # Compute the log variance from the RBF outputs. Note var(x) = 1 / RBF(x)
    logvar_rbf = -log.(rbf_outputs_safe)

    # Compute regularization term if a regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(rbf_outputs) : 0.0f0

    # Mean squared error loss for log variances 
    mse_loss = Flux.mse(logvar_x̂, logvar_rbf)
    # Include regularization
    total_loss = mse_loss + reg_strength * reg_term

    return total_loss
end # function

# ==============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# RBFVAE training functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ==============================================================================

"""
    `train!(rbfvae, x, optimizer; loss_function=loss, loss_kwargs=Dict())`

Customized training function to update parameters of a Variational Autoencoder
(VAE) with a Radial Basis Function (RBF) network as part of its decoder. This
function takes a pre-trained VAE model with an RBF layer and performs a single
update step using the specified loss function for the RBF layer only.

The RBFVAE loss function is aimed at modeling the variance structure in the
latent space, enhancing the VAE's capacity to represent complex data manifolds.

# Arguments
- `rbfvae::RBFVAE`: Struct containing the elements of an RBFVAE, including a VAE
  and an RBF network.
- `x::AbstractVecOrMat{Float32}`: Data on which to evaluate the loss function.
  Can be a vector or a matrix where each column represents a single data point.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword arguments
- `loss_function::Function`: The loss function to be used during training,
  defaulting to `loss`.
- `loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
  passed to the loss function.

# Description
Performs one step of gradient descent on the loss function to train the RBF
network within the VAE. The RBF network's parameters are updated to minimize the
discrepancy between the VAE decoder's estimated log variances and those adjusted
by the RBF network. The function allows for customization of loss
hyperparameters during training.

# Examples
```julia
optimizer = Flux.Optimise.ADAM(1e-3)

# Assuming 'x' is your input data and 'rbfvae' is an instance of RBFVAE
train!(rbfvae, x, optimizer)
````

# Notes

- Ensure that the dimensionality of the input data x aligns with the encoder's
  expected input in the RBFVAE.
- The training function assumes that the RBF network's centers and bandwidths
  are fixed and that only the weights and biases are updated.
- The provided loss function should compute the loss based on the VAE's output
  and the RBF network's output.
"""
function train!(
    rbfvae::RBFVAE,
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(rbfvae) do rbf_model
        loss_function(rbf_model.vae, rbf_model.rbf, x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, rbfvae, ∇loss_[1])
end # function