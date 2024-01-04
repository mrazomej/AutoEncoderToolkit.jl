# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

# Import GPU library
import CUDA

##

# Import Abstract Types

using ..AutoEncode: AbstractVariationalAutoEncoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder, JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder, VAE

using ..VAEs: reparameterize

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caterini, A. L., Doucet, A. & Sejdinovic, D. Hamiltonian Variational
# Auto-Encoder. 11 (2018).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# HVAE struct and forward pass methods
# ==============================================================================

@doc raw"""
    struct HVAE{
        V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
    } <: AbstractVariationalAutoEncoder

Hamiltonian Variational Autoencoder (HVAE) model defined for `Flux.jl`.

# Fields
- `vae::V`: A Variational Autoencoder (VAE) model that forms the basis of the
    HVAE. `V` is a subtype of `VAE` with a specific `AbstractVariationalEncoder`
    and `AbstractVariationalDecoder`.

An HVAE is a type of Variational Autoencoder (VAE) that uses Hamiltonian Monte
Carlo (HMC) to sample from the posterior distribution in the latent space. The
VAE's encoder compresses the input into a low-dimensional probabilistic
representation q(z|x). The VAE's decoder tries to reconstruct the original input
from a sampled point in the latent space p(x|z). 

The HMC sampling in the latent space allows the HVAE to better capture complex
posterior distributions compared to a standard VAE, which assumes a simple
Gaussian posterior. This can lead to more accurate reconstructions and better
disentanglement of latent variables.
"""
mutable struct HVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor HVAE

# ------------------------------------------------------------------------------ 

"""
    (hvae::HVAE)(x::AbstractVecOrMat{Float32}, prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0); latent::Bool=false, n_samples::Int=1)

This function performs a forward pass of the Hamiltonian Variational Autoencoder
(HVAE) with no leapfrog steps. It is equivalent to a regular forward pass of the
VAE.

# Arguments
- `x`: The input data.
- `prior`: The prior distribution for the latent space. Defaults to a standard
  normal distribution.
- `latent`: If true, the function returns the latent variables and mutual
  information. If false, it returns the reconstructed data from the decoder.
- `n_samples`: The number of samples to draw from the posterior distribution in
  the latent space.

# Returns
- If `latent` is true, a dictionary containing the latent variables and mutual
  information.
- If `latent` is false, the reconstructed data from the decoder.
"""
function (hvae::HVAE)(
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
    n_samples::Int=1
)
    return hvae.vae(x; prior=prior, latent=latent, n_samples=n_samples)
end

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Functions to construct decoder distribution
# ==============================================================================

@doc raw"""
        MvDiagGaussianDecoder(decoder, z; σ=1.0f0)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::SimpleDecoder`: A `SimpleDecoder` struct representing the decoder
  model.
- `z::AbstractVector{T}`: An abstract vector of type `AbstractVector{T}`
  representing the latent variable, where `T` is a subtype of `AbstractFloat`.

# Optional Keyword Arguments
- `σ::T`: A float of type `T` representing the standard deviation of the
  Gaussian distribution. Default is 1.0f0.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = SimpleDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z, σ=1.0f0)
```
"""
function MvDiagGaussianDecoder(
    decoder::SimpleDecoder,
    z::AbstractVector{T};
    σ::T=1.0f0,
) where {T<:AbstractFloat}
    # Compute the mean of the decoder output given the latent variable z
    μ = decoder(z)

    # Create a multivariate diagonal Gaussian distribution with mean μ and standard deviation σ
    decoder_dist = Distributions.MvNormal(μ, σ)

    # Return the decoder distribution
    return decoder_dist
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        MvDiagGaussianDecoder(decoder, z)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::JointLogDecoder`: A `JointLogDecoder` struct representing the
    decoder model.
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z)
```
"""
function MvDiagGaussianDecoder(
    decoder::JointLogDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute intermediate representation
    intermediate = decoder.decoder(z)

    # Compute mean
    μ = decoder.µ(intermediate)

    # Compute log standard deviation
    logσ = decoder.logσ(intermediate)

    # Construct multivariate diagonal Gaussian distribution
    decoder_dist = Distributions.MvNormal(μ, exp.(logσ))

    # Return decoder distribution
    return decoder_dist
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        MvDiagGaussianDecoder(decoder, z)

Constructs a multivariate diagonal Gaussian distribution for decoding.

# Arguments
- `decoder::JointDecoder`: A `JointDecoder` struct representing the decoder
  model.
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.

# Returns
- `decoder_dist`: A `Distributions.MvNormal{T}` object representing the
  multivariate diagonal Gaussian distribution, where `T` is the type of elements
  in `z`.

# Example
```julia
# Define a decoder
decoder = JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define a latent variable
z = rand(Float32, 2)

# Construct the decoder distribution
decoder_dist = MvDiagGaussianDecoder(decoder, z)
```
"""
function MvDiagGaussianDecoder(
    decoder::JointDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute intermediate representation
    intermediate = decoder.decoder(z)

    # Compute mean
    μ = decoder.µ(intermediate)

    # Compute standard deviation
    σ = decoder.σ(intermediate)

    # Construct multivariate diagonal Gaussian distribution
    decoder_dist = Distributions.MvNormal(μ, σ)

    # Return decoder distribution
    return decoder_dist
end # function

# ==============================================================================
# Function to construct prior distribution
# ==============================================================================

@doc raw"""
        SphericalPrior(z::AbstractVector{T}, σ::T=1.0f0) where {T<:AbstractFloat}

Generates a prior distribution as a spherical Gaussian. A spherical Gaussian is
a multivariate Gaussian distribution with a diagonal covariance matrix where all
the diagonal elements (variances) are the same. This results in a distribution
that is symmetric (or "spherical") in all dimensions.

# Arguments
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.
- `σ::T=1.0f0`: The standard deviation of the spherical Gaussian distribution.
  Defaults to 1.0.

# Returns
- `prior`: A `Distributions.MvNormal{T}` object representing the multivariate
  spherical Gaussian distribution, where `T` is the type of elements in `z`.

# Example
```julia
# Define a latent variable
z = rand(Float32, 2)
# Define the standard deviation
σ = 0.5f0
# Generate the spherical Gaussian prior distribution
prior = SphericalPrior(z, σ)
```
"""
function SphericalPrior(
    z::AbstractVector{T}, σ::T=1.0f0
) where {T<:AbstractFloat}
    # Generate prior distribution as spherical Gaussian
    prior = Distributions.MvNormal(zeros(T, length(z)), σ .* ones(T, length(z)))
    # Return prior
    return prior
end # function

# ==============================================================================
# Function to compute potential energy
# ==============================================================================

@doc raw"""
        potential_energy(hvae::HVAE;
                         decoder_dist::Function = MvDiagGaussianDecoder,
                         prior::Function = SphericalPrior)

Compute the potential energy of a Hamiltonian Variational Autoencoder (HVAE). In
the context of Hamiltonian Monte Carlo (HMC), the potential energy is defined as
the negative log-posterior. This function returns a function `U` that computes
the potential energy for given data `x` and latent variable `z`. It does this by
computing the log-likelihood of `x` under the distribution defined by
`decoder_dist(decoder, z)`, and the log-prior of `z` under the `prior`
distribution. The potential energy is then computed as the negative sum of the
log-likelihood and the log-prior.

# Arguments
- `hvae::HVAE`: An `HVAE` struct representing the Hamiltonian Variational
  Autoencoder model.

# Optional Keyword Arguments
- `decoder_dist::Function=MvDiagGaussianDecoder`: A function representing the
  distribution function used by the decoder. Default is `MvDiagGaussianDecoder`.
- `prior::Function=SphericalPrior`: A function representing the prior
  distribution used in the autoencoder. Default is `SphericalPrior`.

# Returns
- `U::Function`: A function that computes the potential energy given an input
  `x` and latent variable `z`.

# Example
```julia
# Define an HVAE
hvae = HVAE(VAE(JointEncoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2))), JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))))

# Compute the potential energy
U = potential_energy(hvae, MvDiagGaussianDecoder, SphericalPrior)

# Use the function U to compute the potential energy for a given x and z
x = rand(2)
z = rand(2)
energy = U(x, z)
```
"""
function potential_energy(
    hvae::HVAE;
    decoder_dist::Function=MvDiagGaussianDecoder,
    prior::Function=SphericalPrior,
)
    # Extract decoder
    decoder = hvae.vae.decoder

    # Define function to compute potential energy
    function U(
        x::AbstractVector{T}, z::AbstractVector{T}
    ) where {T<:AbstractFloat}
        # Compute log-likelihood
        log_likelihood = Distributions.logpdf(decoder_dist(decoder, z), x)

        # Compute log-prior
        log_prior = Distributions.logpdf(prior(z), z)

        # Compute potential energy
        energy = -log_likelihood - log_prior

        return energy
    end # function

    return U
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        leapfrog_step(hvae, x, z, ρ, ϵ, potential_energy, potential_energy_kwargs)

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `hvae::HVAE`: An `HVAE` struct representing the Hamiltonian Variational
  Autoencoder model.
- `x::AbstractVector{T}`: The input data.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::AbstractVector{T}`: The step size for each dimension in the HMC algorithm.

## Optional Keyword Arguments
- `potential_energy::Function=potential_energy`: The potential energy function
  used in the HMC algorithm.
- `potential_energy_kwargs::Dict=Dict()`: The keyword arguments for the
  potential energy function. See `potential_energy` for details.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define an HVAE
hvae = HVAE(VAE(JointEncoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2))), JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))))

# Define input data, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = rand(2)

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(hvae, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    hvae::HVAE,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::AbstractVector{T};
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(),
) where {T<:AbstractFloat}
    # Build potential energy function
    U = potential_energy(hvae; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(Ƶ::AbstractVector{T}) = first(Zygote.gradient(Ƶ -> U(x, Ƶ), Ƶ))

    # Update momentum variable with half-step
    ρ̃ = ρ - ϵ .* ∇U(z) / 2

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - ϵ .* ∇U(z̄) / 2

    return z̄, ρ̄
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, potential_energy_kwargs)

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `hvae::HVAE`: An `HVAE` struct representing the Hamiltonian Variational
  Autoencoder model.
- `x::AbstractVector{T}`: The input data.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::AbstractFloat`: The step size for the HMC algorithm.

## Optional Keyword Arguments
- `potential_energy::Function=potential_energy`: The potential energy function
  used in the HMC algorithm.
- `potential_energy_kwargs::Dict=Dict()`: The keyword arguments for the
  potential energy function. See `potential_energy` for details.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define a decoder
decoder = JointDecoder(
    Flux.Chain(Dense(10, 5, relu), Dense(5, 2)),
    Flux.Dense(2, 2), Flux.Dense(2, 2)
)

# Define input data, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = 0.01

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    hvae::HVAE,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::T;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(),
) where {T<:AbstractFloat}
    # Build potential energy function
    U = potential_energy(hvae; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(Ƶ::AbstractVector{T}) = first(Zygote.gradient(Ƶ -> U(x, Ƶ), Ƶ))

    # Update momentum variable with half-step
    ρ̃ = ρ - ϵ * ∇U(z) / 2

    # Update position variable with full-step
    z̄ = z + ϵ * ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - ϵ * ∇U(z̄) / 2

    return z̄, ρ̄
end # function

# ==============================================================================
# Tempering Functions
# ==============================================================================

@doc raw"""
    quadratic_tempering(βₒ::AbstractFloat, k::Int, K::Int)

Compute the inverse temperature `βₖ` at a given stage `k` of a tempering
schedule with `K` total stages, using a quadratic tempering scheme. 

Tempering is a technique used in sampling algorithms to improve mixing and
convergence. It involves running parallel chains of the algorithm at different
"temperatures", and swapping states between the chains. The "temperature" of a
chain is controlled by an inverse temperature parameter `β`, which is varied
according to a tempering schedule. 

In a quadratic tempering schedule, the inverse temperature `βₖ` at stage `k` is
computed as the square of the quantity `((1 - 1 / √(βₒ)) * (k / K)^2 + 1 /
√(βₒ))`, where `βₒ` is the initial inverse temperature. This schedule starts at
`βₒ` when `k = 0`, and increases quadratically as `k` increases, reaching 1 when
`k = K`.

# Arguments
- `βₒ::AbstractFloat`: The initial inverse temperature.
- `k::Int`: The current stage of the tempering schedule.
- `K::Int`: The total number of stages in the tempering schedule.

# Returns
- `βₖ::AbstractFloat`: The inverse temperature at stage `k`.

# Example
```julia
# Define the initial inverse temperature, current stage, and total stages
βₒ = 0.5
k = 5
K = 10

# Compute the inverse temperature at stage k
βₖ = quadratic_tempering(βₒ, k, K)
```
"""
function quadratic_tempering(
    βₒ::T,
    k::Int,
    K::Int,
) where {T<:AbstractFloat}
    # Compute βₖ
    βₖ = ((1 - 1 / √(βₒ)) * (k / K)^2 + 1 / √(βₒ))^(-2)

    return T(βₖ)
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        null_tempering(βₒ::T, k::Int, K::Int) where {T<:AbstractFloat}

Return the initial inverse temperature `βₒ`. This function is used in the
context of tempered Hamiltonian Monte Carlo (HMC) methods, where tempering
involves running HMC at different "temperatures" to improve mixing and
convergence. 

In this case, `null_tempering` is a simple tempering schedule that does not
actually change the temperature—it always returns the initial inverse
temperature `βₒ`. This can be useful as a default or placeholder tempering
schedule.

# Arguments
- `βₒ::AbstractFloat`: The initial inverse temperature. 
- `k::Int`: The current step in the tempering schedule. Not used in this
    function, but included for compatibility with other tempering schedules.
- `K::Int`: The total number of steps in the tempering schedule. Not used in
    this function, but included for compatibility with other tempering
    schedules.

# Returns
- `β::T`: The inverse temperature for the current step, which is always `βₒ` in
  this case.

# Example
```julia
βₒ = 0.5
k = 1
K = 10
β = null_tempering(βₒ, k, K)  # β will be 0.5
```
"""
function null_tempering(
    βₒ::AbstractFloat,
    k::Int,
    K::Int,
)
    return βₒ
end # function

# ==============================================================================
# Forward pass methods for HVAE with Hamiltonian steps
# ==============================================================================

@doc raw"""
    leapfrog_tempering_step(
        hvae::HVAE,
        x::AbstractVector{T},
        zₒ::AbstractVector{T},
        K::Int,
        ϵ::Union{T,<:AbstractVector{T}},
        βₒ::T;
        potential_energy::Function=potential_energy,
        potential_energy_kwargs::Dict=Dict(
            :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
        ),
        tempering_schedule::Function=null_tempering,
    ) where {T<:AbstractFloat}

Combines the leapfrog and tempering steps into a single function for the
Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The HVAE model.
- `x::AbstractVector{T}`: The data to be processed.
- `zₒ::AbstractVector{T}`: The initial latent variable.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.

# Optional Keyword Arguments
- `potential_energy::Function`: The function to compute the potential energy in
  the HMC algorithm. Defaults to `potential_energy`.
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments to pass to
  the `potential_energy` function. Defaults to `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to `null_tempering`.

# Returns
- `zₖ::AbstractVector{T}`: The final latent variable after `K` leapfrog steps.
- `ρₖ::AbstractVector{T}`: The final momentum variable after `K` leapfrog steps.

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the HVAE model.
"""
function leapfrog_tempering_step(
    hvae::HVAE,
    x::AbstractVector{T},
    zₒ::AbstractVector{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=null_tempering,
) where {T<:AbstractFloat}
    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(SphericalPrior(zₒ))
    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Initialize variables to have them available outside the for loop
    zₖ = Vector{T}(undef, length(zₒ))
    ρₖ = Vector{T}(undef, length(ρₒ))

    Zygote.ignore() do
        # Loop over K steps
        for k = 1:K
            # 1) Leapfrog step
            zₖ, ρₖ = leapfrog_step(
                hvae, x, zₖ₋₁, ρₖ₋₁, ϵ;
                potential_energy=potential_energy,
                potential_energy_kwargs=potential_energy_kwargs,
            )

            # 2) Tempering step
            # Compute previous step's inverse temperature
            βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
            # Compute current step's inverse temperature
            βₖ = tempering_schedule(βₒ, k, K)

            # Update momentum variable with tempering
            ρₖ = ρₖ * √(βₖ₋₁ / βₖ)
            # Update zₖ₋₁, ρₖ₋₁ for next iteration
            zₖ₋₁, ρₖ₋₁ = zₖ, ρₖ
        end # for
    end # do block
    return zₖ, ρₖ
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        (hvae::HVAE{VAE{JointLogEncoder,SimpleDecoder}})(
                x::AbstractVector{T},
                K::Int,
                ϵ::Union{T,<:AbstractVector{T}},
                βₒ::T;
                potential_energy::Function=potential_energy,
                potential_energy_kwargs::Dict=Dict(
                    :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
                ),
                tempering_schedule::Function=null_tempering,
                prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
                latent::Bool=false,
        ) where {T<:Float32}

This function performs the forward pass of the Hamiltonian Variational
Autoencoder (HVAE) with a `JointLogEncoder` and a `SimpleDecoder`.

# Arguments
- `hvae::HVAE{VAE{JointLogEncoder,SimpleDecoder}}`: The HVAE model.
- `x::AbstractVector{T}`: The input data.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.

# Optional Keyword Arguments
- `potential_energy::Function`: The function to compute the potential energy in
  the HMC algorithm. Defaults to `potential_energy`.
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments to pass to
  the `potential_energy` function. Defaults to `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to `null_tempering`.
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  variables. Defaults to a standard normal distribution.
- `latent::Bool`: Whether to return the latent variables. If `true`, the
  function returns a dictionary containing the encoder mean and log standard
  deviation, the initial and final latent variables, and the decoder mean. If
  `false`, the function returns the decoder mean.

# Returns
- If `latent` is `true`, returns a `Dict` with the following keys:
    - `:encoder_µ`: The mean of the encoder's output distribution.
    - `:encoder_logσ`: The log standard deviation of the encoder's output
      distribution.
    - `:z_init`: The initial latent variable.
    - `:z_final`: The final latent variable after `K` leapfrog steps.
    - `:decoder_µ`: The mean of the decoder's output distribution.
- If `latent` is `false`, returns the mean of the decoder's output distribution.

# Description
The function first runs the input data through the encoder to obtain the mean
and log standard deviation. It then uses the reparameterization trick to
generate an initial latent variable. Next, it performs `K` leapfrog steps, each
followed by a tempering step, to generate a new sample from the latent space.
Finally, it runs the final latent variable through the decoder to obtain the
output data.

# Note
Ensure the input data `x` matches the expected input dimensionality for the HVAE
model.
"""
function (hvae::HVAE{VAE{JointLogEncoder,SimpleDecoder}})(
    x::AbstractVector{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=null_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
) where {T<:Float32}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = hvae.vae.encoder(x)

    # Run reparametrization trick to generate latent variable zₒ
    zₒ = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=1, log=true
    )

    # Run leapfrog and tempering steps
    zₖ, ρₖ = leapfrog_tempering_step(
        hvae, x, zₒ, K, ϵ, βₒ;
        potential_energy=potential_energy,
        potential_energy_kwargs=potential_energy_kwargs,
        tempering_schedule=tempering_schedule,
    )

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder and collect outputs in dictionary
        return Dict(
            :encoder_µ => encoder_µ,
            :encoder_logσ => encoder_logσ,
            :z_init => zₒ,
            :z_final => zₖ,
            :decoder_µ => hvae.vae.decoder(zₖ),
        )
    else
        # Run latent sample through decoder
        return hvae.vae.decoder(zₖ)
    end # if
end # function
# ------------------------------------------------------------------------------ 

@doc raw"""
        (hvae::HVAE{VAE{JointLogEncoder,D}})(
                x::AbstractVector{T},
                K::Int,
                ϵ::Union{T,<:AbstractVector{T}},
                βₒ::T;
                prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
                potential_energy::Function=potential_energy,
                potential_energy_kwargs::Dict=Dict(
                        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
                ),
                tempering_schedule::Function=null_tempering,
                latent::Bool=false,
        ) where {D<:Union{JointLogDecoder,SplitLogDecoder},T<:Float32}

Processes the input data `x` through a Hamiltonian Variational Autoencoder
(HVAE), consisting of an encoder and either a `JointLogDecoder` or a
`SplitLogDecoder`.

# Arguments
- `hvae::HVAE{VAE{JointLogEncoder,D}}`: The HVAE model.
- `x::AbstractVector{T}`: The data to be processed. This should be a vector of
  type `T`.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.

# Optional Keyword Arguments
- `potential_energy::Function`: The function to compute the potential energy in
  the HMC algorithm. Defaults to `potential_energy`.
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments to pass to
  the `potential_energy` function. Defaults to `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to `null_tempering`.  
- `prior::Distributions.Sampleable`: Specifies the prior distribution to be used
  during the reparametrization trick. Defaults to a standard normal
  distribution.
- `latent::Bool`: If set to `true`, returns a dictionary containing the latent
  variables as well as the mean and log standard deviation of the reconstructed
  data. Defaults to `false`.

# Returns
- If `latent` is `true`, returns a `Dict` with the following keys:
    - `:encoder_µ`: The mean of the encoder's output distribution.
    - `:encoder_logσ`: The log standard deviation of the encoder's output
      distribution.
    - `:z_init`: The initial latent variable.
    - `:z_final`: The final latent variable after `K` leapfrog steps.
    - `:decoder_µ`: The mean of the decoder's output distribution.
    - `:decoder_logσ`: The log standard deviation of the decoder's output
      distribution.
- If `latent` is `false`, returns the mean and log standard deviation of the
  decoder's output distribution.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution. Then, it performs `K` steps of HMC with tempering to
generate a new sample from the latent space, which is then decoded.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the HVAE.
"""
function (hvae::HVAE{VAE{JointLogEncoder,D}})(
    x::AbstractVector{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=null_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
) where {D<:Union{JointLogDecoder,SplitLogDecoder},T<:Float32}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = hvae.vae.encoder(x)

    # Run reparametrization trick to generate latent variable zₒ
    zₒ = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=1, log=true
    )

    # Run leapfrog and tempering steps
    zₖ, ρₖ = leapfrog_tempering_step(
        hvae, x, zₒ, K, ϵ, βₒ;
        potential_energy=potential_energy,
        potential_energy_kwargs=potential_energy_kwargs,
        tempering_schedule=tempering_schedule,
    )

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder to optain mean and log std
        decoder_µ, decoder_logσ = hvae.vae.decoder(zₖ)
        # Colect outputs in dictionary
        return Dict(
            :encoder_µ => encoder_µ,
            :encoder_logσ => encoder_logσ,
            :z_init => zₒ,
            :z_final => zₖ,
            :decoder_µ => decoder_µ,
            :decoder_logσ => decoder_logσ
        )
    else
        # Run latent sample through decoder
        return hvae.vae.decoder(zₖ)
    end # if
end # function

# ==============================================================================
# Hamiltonian ELBO
# ==============================================================================

@doc raw"""
    hamiltonian_elbo(
        vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
        x::AbstractVector{T},
        vae_outputs::Dict{Symbol,<:AbstractVector{T}};
        K::Int=3,
        βₒ::T=0.3f0,
        ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
        potential_energy::Function=potential_energy,
        potential_energy_kwargs::Dict=Dict(
            :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
        ),
        tempering_schedule::Function=quadratic_tempering,
    ) where {T<:AbstractFloat}

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a variational autoencoder (VAE) with a joint log encoder and an
abstract variational decoder.

This function takes as input a VAE, a vector of input data `x`, and a dictionary
of VAE outputs `vae_outputs`. It performs `K` HMC steps with a leapfrog
integrator and a tempering schedule to estimate the ELBO. The ELBO is computed
as the difference between the log evidence estimate `log p̄` and the log
variational estimate `log q̄`.

# Arguments
- `vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder}`: The VAE used to
  encode the input data and decode the latent space.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`.
- `vae_outputs::Dict{Symbol,<:AbstractVector{T}}`: A dictionary of VAE outputs,
  including the latent space mean `encoder_µ` and log standard deviation
  `encoder_logσ`.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `potential_energy::Function`: The potential energy function used in the HMC
  (default is `potential_energy`).
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments for the
  potential energy function (default is `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).

# Returns
- `elbo::T`: The HMC estimate of the ELBO.

# Note
- It is assumed that the mapping from latent space to decoder parameters
  (`decoder_µ` and `decoder_σ`) has been performed prior to calling this
  function. 

# Example
```julia
# Define a VAE
vae = VAE(
    JointLogEncoder(
        Flux.Chain(Flux.Dense(784, 400, relu)), 
        Flux.Dense(400, 20), 
        Flux.Dense(400, 20)
    ),
    AbstractVariationalDecoder()
)

# Define input data
x = rand(Float32, 784)

# Define VAE outputs
vae_outputs = Dict(
    :z => rand(Float32, 20),
    :encoder_µ => rand(Float32, 20),
    :encoder_logσ => rand(Float32, 20)
)

# Compute the HMC estimate of the ELBO
elbo = hamiltonian_elbo(vae, x, vae_outputs)
```
"""
function hamiltonian_elbo(
    vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
    x::AbstractVector{T},
    vae_outputs::Dict{Symbol,<:AbstractVector{T}};
    K::Int=3,
    βₒ::T=0.3f0,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Unpack zₒ from vae_outputs 
    # (equivalent to sampling zₒ ~ variational prior) 
    zₒ = vae_outputs[:z]

    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(SphericalPrior(zₒ))
    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Initialize variables to have them available outside the for loop
    zₖ = Vector{T}(undef, length(zₒ))
    ρₖ = Vector{T}(undef, length(ρₒ))

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(
            vae.decoder,
            x,
            zₖ₋₁,
            ρₖ₋₁,
            ϵ;
            potential_energy=potential_energy,
            potential_energy_kwargs=potential_energy_kwargs,
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering
        ρₖ = ρₖ * √(βₖ₋₁ / βₖ)
        # Update zₖ₋₁, ρₖ₋₁ for next iteration
        zₖ₋₁, ρₖ₋₁ = zₖ, ρₖ
    end # for

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    # log p̄ = - U(x, zₖ) + log p(ρₖ)
    log_p̄ = -potential_energy(vae.decoder; potential_energy_kwargs...)(x, zₖ) +
             Distributions.logpdf(SphericalPrior(ρₖ), ρₖ)

    # log q̄ = log q(zₒ) + log p(ρₒ)
    log_q̄ = Distributions.logpdf(
        Distributions.MvNormal(
            vae_outputs[:encoder_µ], vae_outputs[:encoder_logσ]
        ),
        zₒ
    ) + Distributions.logpdf(SphericalPrior(ρₒ, βₒ^-1), ρₒ)

    return log_p̄ - log_q̄
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        hamiltonian_elbo(
                vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
                x::AbstractMatrix{T},
                vae_outputs::Dict{Symbol,<:AbstractMatrix{T}},
                index::Int;
                K::Int=3,
                βₒ::T=0.3f0,
                ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
                potential_energy::Function=potential_energy,
                potential_energy_kwargs::Dict=Dict(
                        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
                ),
                tempering_schedule::Function=quadratic_tempering,
        ) where {T<:AbstractFloat}

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a variational autoencoder (VAE) with a joint log encoder and an
abstract variational decoder.

This function takes as input a VAE, a matrix of input data `x`, a dictionary of
VAE outputs `vae_outputs`, and an index `index`. It performs `K` HMC steps with
a leapfrog integrator and a tempering schedule to estimate the ELBO for the
`index`-th data point. The ELBO is computed as the difference between the log
evidence estimate `log p̄` and the log variational estimate `log q̄`.

# Arguments
- `vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder}`: The VAE used to
  encode the input data and decode the latent space.
- `x::AbstractMatrix{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`. Each column of `x` represents a separate data sample.
- `vae_outputs::Dict{Symbol,<:AbstractMatrix{T}}`: A dictionary of VAE outputs,
  including the latent space mean `encoder_µ` and log standard deviation
  `encoder_logσ`.
- `index::Int`: The index of the data point for which to compute the ELBO.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `potential_energy::Function`: The potential energy function used in the HMC
  (default is `potential_energy`).
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments for the
  potential energy function (default is `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).

# Returns
- `elbo::T`: The HMC estimate of the ELBO.

# Note
- It is assumed that the mapping from latent space to decoder parameters
    (`decoder_µ` and `decoder_σ`) has been performed prior to calling this
    function. 
- Although the function accepts multiple samples (one per column of `x`), it
  processes only one sample at a time, specified by the `index` argument.

# Example
```julia
# Define a VAE
vae = VAE(
        JointLogEncoder(
                Flux.Chain(Flux.Dense(784, 400, relu)), 
                Flux.Dense(400, 20), 
                Flux.Dense(400, 20)
        ),
        AbstractVariationalDecoder()
)

# Define input data
x = rand(Float32, 784, 100)

# Define VAE outputs
vae_outputs = Dict(
        :z => rand(Float32, 20, 100),
        :encoder_µ => rand(Float32, 20, 100),
        :encoder_logσ => rand(Float32, 20, 100)
)

# Compute the HMC estimate of the ELBO for the first data point
elbo = hamiltonian_elbo(vae, x, vae_outputs, 1)
```
"""
function hamiltonian_elbo(
    vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
    x::AbstractMatrix{T},
    vae_outputs::Dict{Symbol,<:AbstractMatrix{T}},
    index::Int;
    K::Int=3,
    βₒ::T=0.3f0,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Unpack zₒ from vae_outputs 
    # (equivalent to sampling zₒ ~ variational prior) 
    zₒ = vae_outputs[:z][:, index]

    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(SphericalPrior(zₒ))
    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Initialize variables to have them available outside the for loop
    zₖ = Vector{T}(undef, length(zₒ))
    ρₖ = Vector{T}(undef, length(ρₒ))

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(
            vae.decoder,
            x[:, index],
            zₖ₋₁,
            ρₖ₋₁,
            ϵ;
            potential_energy=potential_energy,
            potential_energy_kwargs=potential_energy_kwargs,
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering
        ρₖ = ρₖ * √(βₖ₋₁ / βₖ)
        # Update zₖ₋₁, ρₖ₋₁ for next iteration
        zₖ₋₁, ρₖ₋₁ = zₖ, ρₖ
    end # for

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    # log p̄ = - U(x, zₖ) + log p(ρₖ)
    log_p̄ = -potential_energy(
        vae.decoder; potential_energy_kwargs...
    )(x[:, index], zₖ) +
             Distributions.logpdf(SphericalPrior(ρₖ), ρₖ)

    # log q̄ = log q(zₒ) + log p(ρₒ)
    log_q̄ = Distributions.logpdf(
        Distributions.MvNormal(
            vae_outputs[:encoder_µ][:, index],
            vae_outputs[:encoder_logσ][:, index]
        ),
        zₒ
    ) + Distributions.logpdf(SphericalPrior(ρₒ, βₒ^-1), ρₒ)

    return log_p̄ - log_q̄
end # function

# ==============================================================================
# Loss Function for HVAE
# ==============================================================================

function loss(
    vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
    x::AbstractVector{T};
    K::Int=3,
    βₒ::T=0.3f0,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true)

    # Compute the HMC estimate of the ELBO
    return hamiltonian_elbo(
        vae,
        x,
        vae_outputs;
        K=K,
        βₒ=βₒ,
        ϵ=ϵ,
        potential_energy=potential_energy,
        potential_energy_kwargs=potential_energy_kwargs,
        tempering_schedule=tempering_schedule,
    )
end # function

# ------------------------------------------------------------------------------ 

function loss(
    vae::VAE{JointLogEncoder,<:AbstractVariationalDecoder},
    x::AbstractMatrix{T};
    K::Int=3,
    βₒ::T=0.3f0,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(
        :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
    ),
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true)

    # Initialize value of ELBO
    elbo = zero(T)

    # Loop through each sample
    for i in axes(x, 2)

        # Compute the HMC estimate of the ELBO
        elbo += hamiltonian_elbo(
            vae,
            x[:, i],
            vae_outputs;
            K=K,
            βₒ=βₒ,
            ϵ=ϵ,
            potential_energy=potential_energy,
            potential_energy_kwargs=potential_energy_kwargs,
            tempering_schedule=tempering_schedule,
        )
    end # for

    return elbo
end # function