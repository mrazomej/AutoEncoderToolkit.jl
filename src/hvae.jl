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


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Caterini, A. L., Doucet, A. & Sejdinovic, D. Hamiltonian Variational
# Auto-Encoder. 11 (2018).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ==============================================================================
# Hamiltonian Leapfrog Integrator
# ==============================================================================
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Functions to construct decoder distribution
# ==============================================================================

"""
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
    μ = decoder(z)
    decoder_dist = Distributions.MvNormal(μ, σ)
    return decoder_dist
end # function

"""
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

    # Compute standard deviation
    σ = exp.(logσ)

    # Construct multivariate diagonal Gaussian distribution
    decoder_dist = Distributions.MvNormal(μ, σ)

    # Return decoder distribution
    return decoder_dist
end # function

"""
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

"""
        SphericalPrior(z::AbstractVector{T}) where {T<:AbstractFloat}

Generates a prior distribution as a spherical Gaussian. A spherical Gaussian is
a multivariate Gaussian distribution with a diagonal covariance matrix where all
the diagonal elements (variances) are the same. This results in a distribution
that is symmetric (or "spherical") in all dimensions.

# Arguments
- `z::AbstractVector{T}`: A vector representing the latent variable, where `T`
  is a subtype of `AbstractFloat`.

# Returns
- `prior`: A `Distributions.MvNormal{T}` object representing the multivariate
  spherical Gaussian distribution, where `T` is the type of elements in `z`.

# Example
```julia
# Define a latent variable
z = rand(Float32, 2)
# Generate the spherical Gaussian prior distribution
prior = SphericalPrior(z)
```
"""
function SphericalPrior(z::AbstractVector{T}) where {T<:AbstractFloat}
    # Generate prior distribution as spherical Gaussian
    prior = Distributions.MvNormal(zeros(length(z)), ones(length(z)))
    # Return prior
    return prior
end # function

# ==============================================================================
# Function to compute potential energy
# ==============================================================================

"""
        potential_energy(decoder::JointDecoder;
                                         decoder_dist::Function = MvDiagGaussianDecoder,
                                         prior::Function = SphericalPrior)

Compute the potential energy of a variational autoencoder (VAE). In the context
of Hamiltonian Monte Carlo (HMC), the potential energy is defined as the
negative log-posterior. This function returns a function `U` that computes the
potential energy for given data `x` and latent variable `z`. It does this by
computing the log-likelihood of `x` under the distribution defined by
`decoder_dist(decoder, z)`, and the log-prior of `z` under the `prior`
distribution. The potential energy is then computed as the negative sum of the
log-likelihood and the log-prior.

# Arguments
- `decoder::JointDecoder`: A `JointDecoder` struct representing the decoder
  model used in the autoencoder.

# Optional Keyword Arguments
- `decoder_dist::Function=MvDiagGaussianDecoder`: A function representing the
  distribution function used by the decoder. Default is `MvDiagGaussianDecoder`.
- `prior::Function=SphericalPrior`: A function representing the prior
  distribution used in the autoencoder. Default is `SphericalPrior`.

# Returns
- `U`: A function that computes the potential energy given an input `x` and
  latent variable `z`.

# Example
```julia
# Define a decoder
decoder = JointDecoder(
    Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), 
    Flux.Dense(2, 2), Flux.Dense(2, 2)
)

# Compute the potential energy
U = potential_energy(decoder, MvDiagGaussianDecoder, SphericalPrior)

# Use the function U to compute the potential energy for a given x and z
x = rand(2)
z = rand(2)
energy = U(x, z)
```
"""
function potential_energy(
    decoder::AbstractVariationalDecoder;
    decoder_dist::Function=MvDiagGaussianDecoder,
    prior::Function=SphericalPrior,
)
    # Define function to compute potential energy
    function U(x, z)
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

"""
    leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, potential_energy_kwargs)

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `decoder::AbstractVariationalDecoder`: The decoder model used in the
  autoencoder.
- `x::AbstractVector{T}`: The input data.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::AbstractVector{T}`: The step size for each dimension in the HMC algorithm.
- `potential_energy::Function=potential_energy`: The potential energy function
  used in the HMC algorithm.
- `potential_energy_kwargs::Dict=Dict()`: The keyword arguments for the
  potential energy function.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define a decoder
decoder = JointDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)), Flux.Dense(2, 2), Flux.Dense(2, 2))

# Define input data, position, momentum, and step size
x = rand(2)
z = rand(2)
ρ = rand(2)
ϵ = rand(2)

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    decoder::AbstractVariationalDecoder,
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::AbstractVector{T},
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Dict=Dict(),
) where {T<:AbstractFloat}
    # Build potential energy function
    U = potential_energy(decoder; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(z) = Zygote.gradient(z -> U(x, z), z)

    # Update momentum variable with half-step
    ρ̃ = @. ρ - ϵ * ∇U(z) / 2

    # Update position variable with full-step
    z̄ = @. z + ϵ * ρ̃

    # Update momentum variable with half-step
    ρ̄ = @. ρ̃ - ϵ * ∇U(z̄) / 2

    return z̄, ρ̄
end # function

# ==============================================================================
# Tempering Functions
# ==============================================================================

function quadratic_tempering(
    β₀::AbstractFloat,
    k
)
    # Define quadratic tempering function
    β = @. β₀ + (β₁ - β₀) * t^2

    return β
end # function

# ==============================================================================
# Hamiltonian ELBO, Fixed Tempering
# ==============================================================================

function hamiltonian_elbo_fixed_tempering(
    vae::VAE,
    x::AbstractVector{Float32},
    vae_outputs::Dict{Symbol,<:AbstractArray{Float32}};
    K::Int=3,
    β₀::AbstractFloat=0.3,
    ϵ::AbstractFloat=0.001,
)
    # Unpack zₒ from vae_outputs
    zₒ = vae_outputs[:z_sample]

    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(
        Distributions.MvNormal(zeros(length(zₒ)), ones(length(zₒ)))
    )
    # Define ρₒ = γₒ / √β₀
    ρₒ = γₒ / sqrt(β₀)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step

        # Update zₖ, ρₖ
        zₖ, ρₖ = leapfrog_step(ϵ, zₖ₋₁, ρₖ₋₁,)

        # Update zₒ, ρₒ
        zₒ, ρₒ = zₖ, ρₖ

        # 2) Tempering step

    end # for
end # function