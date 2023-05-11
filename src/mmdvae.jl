# Import ML libraries
import Flux

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

##

# Import Types

using ..VAEs: VAE, recon

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Maximum-Mean Discrepancy Variational Autoencoders
# Zhao, S., Song, J. & Ermon, S. InfoVAE: Information Maximizing Variational
# Autoencoders. Preprint at http://arxiv.org/abs/1706.02262 (2018).
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `gaussian_kernel(x, y)`
Function to compute the Gaussian Kernel between two vectors `x` and `y`, defined
as
    k(x, y) = exp(-||x - y ||² / ρ²)

# Arguments
- `x::AbstractMatrix{Float32}`: First array in kernel
- `y::AbstractMatrix{Float32}`: Second array in kernel

## Optional Arguments
- `ρ::Float32`: Kernel amplitude hyperparameter.

# Returns
k(x, y) = exp(-||x - y ||² / ρ²)
"""
function gaussian_kernel(
    x::AbstractMatrix{Float32}, y::AbstractMatrix{Float32}; ρ::Float32=1.0f0
)
    # return Gaussian kernel
    return exp.(
        -Distances.pairwise(
            Distances.SqEuclidean(), x, y
        ) ./ ρ^2 ./ size(x, 1)
    )
end # function

@doc raw"""
    `mmd_div(x, y)`
Function to compute the MMD divergence between two vectors `x` and `y`, defined
as
    D(x, y) = k(x, x) - 2 k(x, y) + k(y, y),
where k(⋅, ⋅) is any positive definite kernel.

# Arguments
- `x::AbstractMatrix{Float32}`: First array in kernel
- `y::AbstractMatrix{Float32}`: Second array in kernel

## Optional argument
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kwargs::NamedTuple`: Tuple containing arguments for the Kernel function.
"""
function mmd_div(
    x::AbstractMatrix{Float32},
    y::AbstractMatrix{Float32};
    kernel::Function=gaussian_kernel,
    kwargs...
)
    # Compute and return MMD divergence
    return StatsBase.mean(kernel(x, x; kwargs...)) +
           StatsBase.mean(kernel(y, y; kwargs...)) -
           2 * StatsBase.mean(kernel(x, y; kwargs...))
end # function

@doc raw"""
    `logP_mmd_ratio(x, vae; σ, n_latent_samples)`
Function to compute the ratio between the log probability ⟨log P(x|z)⟩ and the
MMD divergence MMD-D(qᵩ(z|x)||P(z)).

NOTE: This function is useful to define the value of the hyperparameter λ for
the MMD-VAE (InfoVAE) training.

# Arguments
- `x::AbstractMatrix{Float32}`: Data to train the infoVAE.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional Arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `n_latent_samples::Int`: Number of samples to take from the latent space prior
  P(z) when computing the MMD divergence.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
abs(⟨log P(x|z)⟩ / MMD-D(qᵩ(z|x)||P(z)))
"""
function logP_mmd_ratio(
    x::AbstractMatrix{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    n_latent_samples::Int=100,
    reconstruct=recon,
    kernel=gaussian_kernel,
    kernel_kwargs...
)
    # Initialize value to save log probability
    logP_x_z = 0.0f0
    # Initialize value to save MMD divergence
    mmd_q_p = 0.0f0

    # Loop through dataset
    for (i, x_datum) in enumerate(eachcol(x))
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x_datum, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x_datum) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x_datum .- x̂) .^ 2)

        # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
        # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
        mmd_q_p += mmd_div(
            # Sample latent variables from decoder qᵩ(z|x) ~ Normal(µ,
            # exp(2logσ)⋅I)
            µ .+ (Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
            ).*exp.(logσ))[:, :],
            # Sample latent variables from prior P(z) ~ Normal(0, 1)
            Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0),
                length(µ),
                n_latent_samples,
            );
            kernel=kernel,
            kernel_kwargs...
        )
    end # for

    # Return ratio of quantities
    return convert(Float32, abs(logP_x_z / mmd_q_p))
end # function

@doc raw"""
    `loss(x, vae; σ, λ, α, reconstruct, n_samples, kernel_kwargs...)`

Loss function for the Maximum-Mean Discrepancy variational autoencoder. The loss
function is defined as

loss = argmin -⟨⟨log P(x|z)⟩⟩ + (1 - α) ⟨Dₖₗ(qᵩ(z | x) || P(z))⟩ + 
              (λ + α - 1) Dₖₗ(qᵩ(z) || P(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z) encodes the
function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f̲(x), σI̲̲).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

# Arguments
- `x::AbstractVector{Float32}`: Input to the neural network.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `λ::Float32=1`: 
- `α::Float32=1`: Related to the annealing inverse temperature for the
  KL-divergence term.
- `reconstruct::Function`: Function that reconstructs the input x̂ by passing it
  through the autoencoder.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `n_samples::Int`: Number of samples to take from the latent space when
  computing ⟨logP(x|z)⟩.
- `n_latent_samples::Int`: Number of samples to take from the latent space prior
  P(z) when computing the MMD divergence.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    x::AbstractVector{Float32},
    vae::VAE;
    σ::Float32=1.0f0,
    λ::Float32=1.0f0,
    α::Float32=0.0f0,
    reconstruct::Function=recon,
    kernel::Function=gaussian_kernel,
    n_samples::Int=1,
    n_latent_samples::Int=50,
    kernel_kwargs...
)
    # Initialize arrays to save µ and logσ
    µ = similar(Flux.params(vae.µ)[2])
    logσ = similar(µ)

    # Initialize value to save log probability
    logP_x_z = 0.0f0
    # Initialize value to save MMD divergence
    mmd_q_p = 0.0f0

    # Loop through latent space samples
    for i = 1:n_samples
        # Run input through reconstruct function
        µ, logσ, x̂ = reconstruct(x, vae)

        # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
        logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
                    1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

        # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
        # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
        mmd_q_p += mmd_div(
            # Sample the decoder qᵩ(z | x) ~ Normal(µ, exp(2logσ)⋅I)
            µ .+ (Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
            ).*exp.(logσ))[:, :],
            # Sample the prior probability P(z) ~ Normal(0, 1)
            Random.rand(
                Distributions.Normal{Float32}(0.0f0, 1.0f0),
                length(µ),
                n_latent_samples,
            );
            kernel=kernel,
            kernel_kwargs...
        )
    end # for

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z / n_samples + (1 - α) * kl_qₓ_p +
           (λ + α - 1) * mmd_q_p / n_samples
end # function