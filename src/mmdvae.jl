# Import ML libraries
import Flux

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

# Import ChainRulesCore to ignore functions when computing gradients
using ChainRulesCore: @ignore_derivatives

##

# Import Abstract Types

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: BernoulliDecoder, SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import Concrete VAE type
using ..AutoEncode: VAE

# Import log-probability functions
using ..AutoEncode: decoder_loglikelihood, spherical_logprior,
    encoder_logposterior, encoder_kl, Flatten

# Import functions from other modules
using ..VAEs: reparameterize

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoVAE
# Maximum-Mean Discrepancy Variational Autoencoders
# Zhao, S., Song, J. & Ermon, S. InfoVAE: Information Maximizing Variational
# Autoencoders. Preprint at http://arxiv.org/abs/1706.02262 (2018).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `MMDVAE{
        V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
        } <: AbstractVariationalAutoEncoder`

A struct representing a Maximum-Mean Discrepancy Variational Autoencoder
(MMD-VAE).

# Fields
- `vae::V`: A Variational Autoencoder (VAE) that forms the basis of the MMD-VAE.
  The VAE should be composed of an `AbstractVariationalEncoder` and an
  `AbstractVariationalDecoder`.

# Description
The `MMDVAE` struct is a subtype of `AbstractVariationalAutoEncoder` and
represents a specific type of VAE known as an MMD-VAE. The MMD-VAE modifies the
standard VAE by replacing the KL-divergence term in the loss function with a
Maximum-Mean Discrepancy (MMD) term, which measures the distance between the
aggregated posterior of the latent codes and the prior. This can help to
alleviate the issue of posterior collapse, where the aggregated posterior fails
to cover significant parts of the prior, commonly seen in VAEs.

# Citation
> Maximum-Mean Discrepancy Variational Autoencoders. Zhao, S., Song, J. & Ermon,
> S. InfoVAE: Information Maximizing Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1706.02262 (2018).
"""
struct MMDVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
end # struct

# Mark struct as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor MMDVAE

# ------------------------------------------------------------------------------

@doc raw"""
    (mmdvae::MMDVAE)(x::AbstractArray; latent::Bool=false)

Defines the forward pass for the Maximum-Mean Discrepancy Variational
Autoencoder (MMD-VAE).

# Arguments
- `x::AbstractArray`: Input data.

# Optional Keyword Arguments
- `latent::Bool`: Whether to return the latent variables along with the decoder
  output. If `true`, the function returns a tuple containing the encoder
  outputs, the latent sample, and the decoder outputs. If `false`, the function
  only returns the decoder outputs. Defaults to `false`.  

# Returns
- If `latent` is `true`, returns a `NamedTuple` containing:
    - `encoder`: The outputs of the encoder.
    - `z`: The latent sample.
    - `decoder`: The outputs of the decoder.
- If `latent` is `false`, returns the outputs of the decoder.
"""
function (mmdvae::MMDVAE)(
    x::AbstractArray;
    latent::Bool=false
)
    # Return reconstructed output
    return mmdvae.vae(x, latent=latent)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    gaussian_kernel(
        x::AbstractArray, y::AbstractArray; ρ::Float32=1.0f0, dims::Int=2
    )

Function to compute the Gaussian Kernel between two arrays `x` and `y`, defined
as 

        k(x, y) = exp(-||x - y ||² / ρ²)

# Arguments
- `x::AbstractArray`: First input array for the kernel.
- `y::AbstractArray`: Second input array for the kernel.  

# Optional Keyword Arguments
- `ρ=1.0f0`: Kernel amplitude hyperparameter. Larger ρ gives a smoother
  kernel.
- `dims::Int=2`: Number of dimensions to compute pairwise distances over.

# Returns
- `k::AbstractArray`: Kernel matrix where each element is computed as 

# Theory
The Gaussian kernel measures the similarity between two points `x` and `y`. It
is widely used in many machine learning algorithms. This implementation computes
the squared Euclidean distance between all pairs of rows in `x` and `y`, scales
the distance by ρ² and takes the exponential.
"""
function gaussian_kernel(
    x::AbstractArray,
    y::AbstractArray;
    ρ=1.0f0,
    dims::Int=2
)
    # return Gaussian kernel
    return exp.(
        -Distances.pairwise(
            Distances.SqEuclidean(), x, y; dims=dims
        ) ./ ρ^2 ./ size(x, 1)
    )
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    mmd_div(
        x::AbstractArray, y::AbstractArray; 
        kernel::Function=gaussian_kernel, 
        kernel_kwargs::Union{NamedTuple,Dict}=Dict()
    )

Compute the Maximum Mean Discrepancy (MMD) divergence between two arrays `x` and
`y`.

# Arguments  
- `x::AbstractArray`: First input array.
- `y::AbstractArray`: Second input array.

# Keyword Arguments
- `kernel::Function=gaussian_kernel`: Kernel function to use. Default is the
  Gaussian kernel.
- `kernel_kwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments
  to be passed to the kernel function.

# Returns
- `mmd::Number`: MMD divergence value. 

# Theory
MMD measures the difference between two distributions based on embeddings in a
Reproducing Kernel Hilbert Space (RKHS). It is widely used for two-sample tests.

This function implements MMD as:

MMD(x, y) = mean(k(x, x)) - 2 * mean(k(x, y)) + mean(k(y, y))

where k is a positive definite kernel (e.g., Gaussian).
"""
function mmd_div(
    x::AbstractArray,
    y::AbstractArray;
    kernel::Function=gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Compute and return MMD divergence
    return StatsBase.mean(kernel(x, x; kernel_kwargs...)) +
           StatsBase.mean(kernel(y, y; kernel_kwargs...)) -
           2 * StatsBase.mean(kernel(x, y; kernel_kwargs...))
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    logP_mmd_ratio(
        mmdvae::MMDVAE, x::AbstractArray; 
        n_latent_samples::Int=100, kernel=gaussian_kernel, 
        kernel_kwargs::Union{NamedTuple,Dict}=NamedTuple(), 
        reconstruction_loglikelihood::Function=decoder_loglikelihood
    )

Function to compute the absolute ratio between the log likelihood ⟨log p(x|z)⟩
and the MMD divergence MMD-D(qᵩ(z|x)||p(z)).

# Arguments
- `mmdvae::MMDVAE`: Struct containing the elements of the MMD-VAE.
- `x::AbstractArray`: Data to train the MMD-VAE.

# Optional Keyword Arguments
- `n_latent_samples::Int=100`: Number of samples to take from the latent space
  prior p(z) when computing the MMD divergence.
- `kernel=gaussian_kernel`: Kernel used to compute the divergence.  Default is
  the Gaussian Kernel.
- `kernel_kwargs::Union{NamedTuple,Dict}=NamedTuple()`: Tuple containing
  arguments for the Kernel function.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: Function that
  computes the log likelihood of the reconstructed input.

# Returns
abs(⟨log p(x|z)⟩ / MMD-D(qᵩ(z|x)||p(z)))

# Description
This function calculates:

1. The log likelihood ⟨log p(x|z)⟩ of x under the MMD-VAE decoder, averaged over
all samples. 2. The MMD divergence between the encoder distribution q(z|x) and
prior p(z). 

The absolute ratio of these two quantities is returned.

# Note
This ratio is useful for setting the Lagrangian multiplier λ in training
MMD-VAEs.
"""
function logP_mmd_ratio(
    mmdvae::MMDVAE,
    x::AbstractArray;
    n_latent_samples::Int=100,
    kernel=gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=NamedTuple(),
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
)
    # Run input through reconstruct function
    mmdvae_output = mmdvae(x, latent=true)

    # Compute ⟨log p(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = StatsBase.mean(
        reconstruction_loglikelihood(
            x, mmdvae_output.z, mmdvae.vae.decoder, mmdvae_output.decoder
        )
    )

    # Sample latent variables from decoder qᵩ(z|x) using the reparameterization
    # trick. (This was already done in the forward pass of the MMD-VAE)
    q_z_x = mmdvae_output.z

    # Sample latent variables from prior p(z) ~ Normal(0, 1)
    p_z = Random.randn(eltype(x), size(q_z_x, 1), n_latent_samples)

    # Compute MMD divergence between prior dist samples p(z) and sampled latent
    # variables qᵩ(z|x) 
    mmd_q_p = mmd_div(q_z_x, p_z; kernel=kernel, kernel_kwargs...)

    # Return ratio of quantities
    return abs(log_likelihood / mmd_q_p)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    loss(mmdvae::MMDVAE, x::AbstractArray; σ::Number=1.0f0, λ::Number=1.0f0, α::Number=0.0f0, n_latent_samples::Int=50, kernel::Function=gaussian_kernel, kernel_kwargs::Union{NamedTuple,Dict}=Dict(), reconstruction_loglikelihood::Function=decoder_loglikelihood, kl_divergence::Function=encoder_kl)

Loss function for the Maximum-Mean Discrepancy variational autoencoder
(MMD-VAE). The loss function is defined as:

loss = -⟨log p(x|z)⟩ + (1 - α) * Dₖₗ(qᵩ(z | x) || p(z)) + (λ + α - 1) *
MMD-D(qᵩ(z) || p(z)),

# Arguments
- `mmdvae::MMDVAE`: Struct containing the elements of the MMD-VAE.
- `x::AbstractArray`: Input data.

# Optional Arguments
- `λ::Number=1.0f0`: Hyperparameter that emphasizes the importance of the KL
  divergence between qᵩ(z) and π(z) during training.
- `α::Number=0.0f0`: Hyperparameter that emphasizes the importance of the Mutual
  Information term during optimization.
- `n_latent_samples::Int=50`: Number of samples to take from the latent space
  prior π(z) when computing the MMD divergence.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments
  to be passed to the kernel function.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: Function that
  computes the log likelihood of the reconstructed input.
- `kl_divergence::Function=encoder_kl`: Function that computes the
  Kullback-Leibler divergence between the encoder distribution and the prior.

# Returns
- Single value defining the loss function for entry `x` when compared with
  reconstructed output `x̂`.

# Description
This function calculates the loss for the MMD-VAE. It computes the log
likelihood of the reconstructed input, the MMD divergence between the encoder
distribution and the prior, and the Kullback-Leibler divergence between the
approximate decoder and the prior. These quantities are combined according to
the formula above to compute the loss.
"""
function loss(
    mmdvae::MMDVAE,
    x::AbstractArray;
    λ::Number=1.0f0,
    α::Number=0.0f0,
    n_latent_samples::Int=50,
    kernel::Function=gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict(),
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
)
    # Run input through reconstruct function
    mmdvae_output = mmdvae(x, latent=true)

    # Compute ⟨log p(x|z)⟩ for a Gaussian decoder averaged over all samples
    loglikelihood = StatsBase.mean(
        reconstruction_loglikelihood(
            x, mmdvae_output.z, mmdvae.vae.decoder, mmdvae_output.decoder
        )
    )

    # Sample latent variables from decoder qᵩ(z|x) using the reparameterization
    # trick. (This was already done in the forward pass of the MMD-VAE)
    q_z_x = mmdvae_output.z

    # Sample latent variables from prior p(z) ~ Normal(0, 1)
    p_z = Random.randn(eltype(x), size(q_z_x, 1), n_latent_samples)

    # Compute MMD divergence between prior dist samples p(z) and sampled latent
    # variables qᵩ(z|x) 
    mmd_q_p = mmd_div(q_z_x, p_z; kernel=kernel, kernel_kwargs...)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution p(z)
    kl_div = StatsBase.mean(
        kl_divergence(mmdvae.vae.encoder, mmdvae_output.encoder)
    )

    # Compute loss function
    return -loglikelihood + (1 - α) * kl_div + (λ + α - 1) * mmd_q_p
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    loss(
        mmdvae::MMDVAE, x_in::AbstractArray, x_out::AbstractArray; 
        λ::Number=1.0f0, α::Number=0.0f0, 
        n_latent_samples::Int=50, 
        kernel::Function=gaussian_kernel, 
        kernel_kwargs::Union{NamedTuple,Dict}=Dict(), 
        reconstruction_loglikelihood::Function=decoder_loglikelihood, 
        kl_divergence::Function=encoder_kl
    )

Loss function for the Maximum-Mean Discrepancy variational autoencoder
(MMD-VAE). The loss function is defined as:

loss = -⟨log p(x|z)⟩ + (1 - α) * Dₖₗ(qᵩ(z | x) || p(z)) + (λ + α - 1) *
MMD-D(qᵩ(z) || p(z)),

# Arguments
- `mmdvae::MMDVAE`: Struct containing the elements of the MMD-VAE.
- `x_in::AbstractArray`: Input data.
- `x_out::AbstractArray`: Data against which to compare the reconstructed
  output.

# Optional Arguments
- `λ::Number=1.0f0`: Hyperparameter that emphasizes the importance of the KL
  divergence between qᵩ(z) and π(z) during training.
- `α::Number=0.0f0`: Hyperparameter that emphasizes the importance of the Mutual
  Information term during optimization.
- `n_latent_samples::Int=50`: Number of samples to take from the latent space
  prior π(z) when computing the MMD divergence.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments
  to be passed to the kernel function.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: Function that
  computes the log likelihood of the reconstructed input.
- `kl_divergence::Function=encoder_kl`: Function that computes the
  Kullback-Leibler divergence between the encoder distribution and the prior.

# Returns
- Single value defining the loss function for entry `x` when compared with
  reconstructed output `x̂`.

# Description
This function calculates the loss for the MMD-VAE. It computes the log
likelihood of the reconstructed input, the MMD divergence between the encoder
distribution and the prior, and the Kullback-Leibler divergence between the
approximate decoder and the prior. These quantities are combined according to
the formula above to compute the loss.
"""
function loss(
    mmdvae::MMDVAE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    λ::Number=1.0f0,
    α::Number=0.0f0,
    n_latent_samples::Int=50,
    kernel::Function=gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict(),
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
)
    # Run input through reconstruct function
    mmdvae_output = mmdvae(x_in, latent=true)

    # Compute ⟨log p(x|z)⟩ for a Gaussian decoder averaged over all samples
    loglikelihood = StatsBase.mean(
        reconstruction_loglikelihood(
            x_out, mmdvae_output.z, mmdvae.vae.decoder, mmdvae_output.decoder
        )
    )

    # Sample latent variables from decoder qᵩ(z|x) using the reparameterization
    # trick. (This was already done in the forward pass of the MMD-VAE)
    q_z_x = mmdvae_output.z

    # Sample latent variables from prior p(z) ~ Normal(0, 1)
    p_z = Random.randn(eltype(x_in), size(q_z_x, 1), n_latent_samples)

    # Compute MMD divergence between prior dist samples p(z) and sampled latent
    # variables qᵩ(z|x) 
    mmd_q_p = mmd_div(q_z_x, p_z; kernel=kernel, kernel_kwargs...)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution p(z)
    kl_div = StatsBase.mean(
        kl_divergence(mmdvae.vae.encoder, mmdvae_output.encoder)
    )

    # Compute loss function
    return -loglikelihood + (1 - α) * kl_div + (λ + α - 1) * mmd_q_p
end # function

# ------------------------------------------------------------------------------
# Training Functions
# ------------------------------------------------------------------------------

@doc raw"""
    train!(mmdvae, x, opt; loss_function, loss_kwargs, verbose, loss_return)

Customized training function to update parameters of a variational autoencoder
given a specified loss function.

# Arguments
- `mmdvae::MMDVAE`: A struct containing the elements of a Maximum-Mean
  Discrepancy Variational Autoencoder (MMD-VAE).
- `x::AbstractArray`: Data on which to evaluate the loss function. The last
  dimension is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the MMDVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `α`, or `β`, depending on the
  specific loss function in use.
- `verbose::Bool=false`: If true, the loss value will be printed during
  training.
- `loss_return::Bool=false`: If true, the loss value will be returned after
  training.

# Description
Trains the MMDVAE by:
1. Computing the gradient of the loss w.r.t the MMDVAE parameters.
2. Updating the MMDVAE parameters using the optimizer.
"""
function train!(
    mmdvae::MMDVAE,
    x::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(mmdvae) do mmdvae_model
        loss_function(mmdvae_model, x; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, mmdvae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    train!(mmdvae, x_in, x_out, opt; loss_function, loss_kwargs, verbose, loss_return)

Customized training function to update parameters of a variational autoencoder
given a specified loss function.

# Arguments
- `mmdvae::MMDVAE`: A struct containing the elements of a Maximum-Mean
  Discrepancy Variational Autoencoder (MMD-VAE).
- `x_in::AbstractArray`: Data on which to evaluate the loss function. The last
  dimension is taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Data against which to compare the reconstructed
  output.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the MMDVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `α`, or `β`, depending on the
  specific loss function in use.
- `verbose::Bool=false`: If true, the loss value will be printed during
  training.
- `loss_return::Bool=false`: If true, the loss value will be returned after
  training.

# Description
Trains the MMDVAE by:
1. Computing the gradient of the loss w.r.t the MMDVAE parameters.
2. Updating the MMDVAE parameters using the optimizer.
"""
function train!(
    mmdvae::MMDVAE,
    x_in::AbstractArray,
    x_out::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(mmdvae) do mmdvae_model
        loss_function(mmdvae_model, x_in, x_out; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, mmdvae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function