# Import ML libraries
import Flux

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

##

# Import Types

using ..VAEs: VAE

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
- `x::AbstractVecOrMat{Float32}`: First array in kernel
- `y::AbstractVecOrMat{Float32}`: Second array in kernel

## Optional Arguments
- `ρ::Float32=1.0f0`: Kernel amplitude hyperparameter.
- `dims::Int=2`: 

# Returns
k(x, y) = exp(-||x - y ||² / ρ²)
"""
function gaussian_kernel(
    x::AbstractVecOrMat{Float32},
    y::AbstractVecOrMat{Float32};
    ρ::Float32=1.0f0,
    dims::Int=2
)
    # return Gaussian kernel
    return exp.(
        -Distances.pairwise(
            Distances.SqEuclidean(), x, y; dims=dims
        ) ./ ρ^2 ./ size(x, 1)
    )
end # function

@doc raw"""
    `mmd_div(x, y)`
Function to compute the MMD divergence between two vectors `x` and `y`, defined
as D(x, y) = k(x, x) - 2 k(x, y) + k(y, y), where k(⋅, ⋅) is any positive
    definite kernel.

# Arguments
- `x::AbstractVecOrMat{Float32}`: First array in kernel
- `y::AbstractVecOrMat{Float32}`: Second array in kernel

## Optional argument
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
- `MMD-Divergence::Float32`
"""
function mmd_div(
    x::AbstractVecOrMat{Float32},
    y::AbstractVecOrMat{Float32};
    kernel::Function=gaussian_kernel,
    kernel_kwargs...
)
    # Compute and return MMD divergence
    return StatsBase.mean(kernel(x, x; kernel_kwargs...)) +
           StatsBase.mean(kernel(y, y; kernel_kwargs...)) -
           2 * StatsBase.mean(kernel(x, y; kernel_kwargs...))
end # function

@doc raw"""
    `logP_mmd_ratio(vae, x; σ, n_latent_samples)`
Function to compute the ratio between the log probability ⟨log P(x|z)⟩ and the
MMD divergence MMD-D(qᵩ(z|x)||P(z)).

NOTE: This function is useful to define the value of the hyperparameter λ for
the MMD-VAE (InfoVAE) training.

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Data to train the infoVAE.

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
    vae::VAE,
    x::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    n_latent_samples::Int=100,
    kernel=gaussian_kernel,
    kernel_kwargs...
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x, latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
    # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
    mmd_q_p = mmd_div(
        # Sample latent variables from decoder qᵩ(z|x) ~ Normal(µ,
        # exp(2logσ)⋅I)
        µ .+ (Random.rand(
            Distributions.Normal{Float32}(0.0f0, 1.0f0), size(µ)...
        ).*exp.(logσ))[:, :],
        # Sample latent variables from prior P(z) ~ Normal(0, 1)
        Random.rand(
            Distributions.Normal{Float32}(0.0f0, 1.0f0),
            size(µ, 1)...,
            n_latent_samples,
        );
        kernel=kernel,
        kernel_kwargs...
    )

    # Return ratio of quantities
    return abs(logP_x_z / mmd_q_p)
end # function

@doc raw"""
    `loss(vae, x; σ, λ, α, kernel_kwargs...)`

Loss function for the Maximum-Mean Discrepancy variational autoencoder. The loss
function is defined as

loss = argmin -⟨⟨log P(x|z)⟩⟩ + (1 - α) ⟨Dₖₗ(qᵩ(z | x) || P(z))⟩ + 
              (λ + α - 1) Dₖₗ(qᵩ(z) || P(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z)
encodes the function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f̲(x), σI̲̲).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `λ::Float32=1`: Hyperparameter that emphasizes how relevant the KL dirvergence
  between qᵩ(z) and P(z) should be during training.
- `α::Float32=1`: Hyperparameter that emphasizes how relevant the Mutual
  Information term should be during optimization.
- `n_latent_samples::Int`: Number of samples to take from the latent space prior
  P(z) when computing the MMD divergence.
- `kernel::Function=gaussian_kernel`: Kernel used to compute the divergence.
  Default is the Gaussian Kernel.
- `kernel_kwargs::NamedTuple`: Tuple containing arguments for the Kernel
  function.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    vae::VAE,
    x::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    λ::Float32=1.0f0,
    α::Float32=0.0f0,
    n_latent_samples::Int=50,
    kernel::Function=gaussian_kernel,
    kernel_kwargs::Union{NamedTuple,Dict}=Dict(
        :ρ => 1.0f0,
        :dims => 2
    )
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x, latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute MMD divergence between prior dist samples P(z) ~ Normal(0, 1)
    # and sampled latent variables qᵩ(z|x) ~ Normal(µ, exp(2logσ)⋅I)
    mmd_q_p = mmd_div(
        # Sample the decoder qᵩ(z | x) ~ Normal(µ, exp(2logσ)⋅I)
        µ .+ (Random.rand(
            Distributions.Normal{Float32}(0.0f0, 1.0f0), size(µ)...
        ).*exp.(logσ))[:, :],
        # Sample the prior probability P(z) ~ Normal(0, 1)
        Random.rand(
            Distributions.Normal{Float32}(0.0f0, 1.0f0),
            size(µ, 1),
            n_latent_samples,
        );
        kernel=kernel,
        kernel_kwargs...
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z + (1 - α) * kl_qₓ_p + (λ + α - 1) * mmd_q_p
end # function

@doc raw"""
    `train!(vae, x, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to evaluate
  the loss function. NOTE: Every column should represent a single input.
- `opt::NamedTuple`: State of optimizer that will be used to update parameters.
  NOTE: This is in agreement with `Flux.jl ≥ 0.13` where implicit `Zygote`
  gradients are not allowed. This `opt` object can be initialized using 
  `Flux.Train.setup`. For example, one can run
  ```
  opt_state = Flux.Train.setup(Flux.Optimisers.Adam(1E-1), vae)
  ```

## Optional arguments
- `loss_kwargs::Union{NamedTuple,Dict}`: Tuple containing arguments for the loss
    function. For `loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(
        :σ => 1.0f0,
        :λ => 1.0f0,
        :α => 0.0f0,
        :kernel => gaussian_kernel,
        :n_latent_samples => 50,
        :kernel_kwargs => Dict(
            :ρ => 1.0f0,
            :dims => 2
        )
    )
)
    # Compute gradient
    ∇loss_ = Flux.gradient(vae) do vae
        loss(vae, x; loss_kwargs...)
    end # do

    # Update the network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        opt,
        vae,
        ∇loss_[1]
    )
end # function