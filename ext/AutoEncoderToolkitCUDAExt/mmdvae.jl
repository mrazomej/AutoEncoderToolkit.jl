# Import basic math
import StatsBase

import AutoEncoderToolkit as AET
using AutoEncoderToolkit.MMDVAEs

# Import AutoDiff backends
using ChainRulesCore: ignore_derivatives

@doc raw"""
    gaussian_kernel(
        x::CuArray, y::CuArray; ρ::Float32=1.0f0
    )
GPU-compatible function to compute the Gaussian Kernel between columns of arrays `x` and `y`, defined
as 
        k(x_i, y_j) = exp(-||x_i - y_j||² / ρ²)
# Arguments
- `x::CuArray`: First input array for the kernel.
- `y::CuArray`: Second input array for the kernel.  
# Optional Keyword Arguments
- `ρ=1.0f0`: Kernel amplitude hyperparameter. Larger ρ gives a smoother
  kernel.
# Returns
- `k::CuArray`: Kernel matrix where each element (i,j) is computed as k(x_i, y_j)
# Theory
The Gaussian kernel measures the similarity between two points `x_i` and `y_j`. It
is widely used in many machine learning algorithms. This implementation computes
the squared Euclidean distance between all pairs of columns in `x` and `y`, scales
the distance by ρ² and takes the exponential.
"""
function MMDVAEs.gaussian_kernel(
    x::CUDA.CuArray,
    y::CUDA.CuArray;
    ρ::Float32=1.0f0
)
    # Compute dot products
    xx = sum(abs2, x, dims=1)
    yy = sum(abs2, y, dims=1)
    xy = CUDA.transpose(x) * y

    # Compute pairwise squared Euclidean distances
    dist = CUDA.transpose(xx) .+ yy .- 2 .* xy

    # Scale distances and compute exponential
    return exp.(-dist ./ (ρ^2 * size(x, 1)))
end

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
function MMDVAEs.loss(
    mmdvae::MMDVAEs.MMDVAE,
    x::CUDA.CuArray;
    λ::Number=1.0f0,
    α::Number=0.0f0,
    n_latent_samples::Int=50,
    kernel::Function=MMDVAEs.gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict(),
    reconstruction_loglikelihood::Function=AET.decoder_loglikelihood,
    kl_divergence::Function=AET.encoder_kl,
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
    p_z = ignore_derivatives() do
        CUDA.randn(eltype(x), size(q_z_x, 1), n_latent_samples)
    end # ignore_derivatives

    # Compute MMD divergence between prior dist samples p(z) and sampled latent
    # variables qᵩ(z|x) 
    mmd_q_p = MMDVAEs.mmd_div(q_z_x, p_z; kernel=kernel, kernel_kwargs...)

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
function MMDVAEs.loss(
    mmdvae::MMDVAEs.MMDVAE,
    x_in::CUDA.CuArray,
    x_out::CUDA.CuArray;
    λ::Number=1.0f0,
    α::Number=0.0f0,
    n_latent_samples::Int=50,
    kernel::Function=MMDVAEs.gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict(),
    reconstruction_loglikelihood::Function=AET.decoder_loglikelihood,
    kl_divergence::Function=AET.encoder_kl,
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
    p_z = ignore_derivatives() do
        CUDA.randn(eltype(x_in), size(q_z_x, 1), n_latent_samples)
    end # ignore_derivatives

    # Compute MMD divergence between prior dist samples p(z) and sampled latent
    # variables qᵩ(z|x) 
    mmd_q_p = MMDVAEs.mmd_div(q_z_x, p_z; kernel=kernel, kernel_kwargs...)

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
function MMDVAEs.train!(
    mmdvae::MMDVAEs.MMDVAE,
    x::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=MMDVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(mmdvae) do mmdvae_model
            loss_function(mmdvae_model, x; loss_kwargs...)
        end # do block
    end

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
function MMDVAEs.train!(
    mmdvae::MMDVAEs.MMDVAE,
    x_in::CUDA.CuArray,
    x_out::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=MMDVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(mmdvae) do mmdvae_model
            loss_function(mmdvae_model, x_in, x_out; loss_kwargs...)
        end # do block
    end

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