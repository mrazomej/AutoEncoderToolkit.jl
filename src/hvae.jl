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

using ..AutoEncode: Float32Array, AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractVariationalDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder,
    JointLogEncoder, SimpleDecoder, JointLogDecoder,
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
- `x::Float32Array`: The input data.
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  space. Defaults to a standard normal distribution.
- `latent::Bool`: If true, the function returns the latent variables and mutual
  information. If false, it returns the reconstructed data from the decoder.
- `n_samples::Int`: The number of samples to draw from the posterior
  distribution in the latent space.

# Returns
- If `latent` is true, a `NamedTuple` containing the latent variables and mutual
  information.
- If `latent` is false, the reconstructed data from the decoder.
"""
function (hvae::HVAE)(
    x::Float32Array;
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
- `decoder::AbstractGaussianLogDecoder`: A `AbstractGaussianLogDecoder`
    struct representing the decoder model. This assumes that the decoder maps
    the latent variables to the mean and the log of the standard deviation of a
    Gaussian distribution.
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
    decoder::AbstractGaussianLogDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute mean and log std
    μ, logσ = decoder(z)

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
- `decoder::AbstractGaussianLinearDecoder`: A
  `AbstractGaussianLinearDecoder` struct representing the decoder model. This
  assumes that the decoder maps the latent variables to the mean and the
  standard deviation of a Gaussian distribution.
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
    decoder::AbstractGaussianLinearDecoder,
    z::AbstractVector{T}
) where {T<:AbstractFloat}
    # Compute mean and log std
    μ, σ = decoder(z)

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
        potential_energy(decoder::AbstractVariationalDecoder;
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
- `decoder::AbstractVariationalDecoder`: A decoder model that maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `decoder_dist::Function=MvDiagGaussianDecoder`: A function representing the
  distribution function used by the decoder. The function must take as first
  input an `AbstractVariationalDecoder` struct and as second input a vector `z`
  representing the latent variable. Default is `MvDiagGaussianDecoder`.
- `prior::Function=SphericalPrior`: A function representing the prior
  distribution used in the autoencoder. The function must take as single input a
  vector `z` representing the latent variable. Default is `SphericalPrior`.

# Returns
- `U::Function`: A function that computes the potential energy given an input
  `x` and latent variable `z`.

# Example
```julia
# Define decoder
decoder = JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))

# Compute the potential energy
U = potential_energy(decoder)

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
        ∇potential_energy(decoder::AbstractVariationalDecoder;
                          potential_energy::Function=potential_energy,
                          potential_energy_kwargs::Union{NamedTuple,Dict}=(
                            decoder_dist=MvDiagGaussianDecoder,
                            prior=SphericalPrior
                        )
                    )

Compute the gradient of the potential energy of a Hamiltonian Variational
Autoencoder (HVAE) with respect to the latent variables `Ƶ` using `Zygote.jl`
AutoDiff. This function returns a function `∇U` that takes both `x` and `Ƶ` as
arguments, but only computes the gradient with respect to `Ƶ`. The
`potential_energy` function must generate a function that takes two arguments:
First, `x`, the data, then, `Ƶ`, the latent variables.

# Arguments
- `decoder::AbstractVariationalDecoder`: A decoder model that maps the latent
  variables to the data space.

# Optional Keyword Arguments
- `potential_energy::Function=potential_energy`: A function that generates a
  function to compute the potential energy. The potential_energy function must
  take a single argument, the HVAE, and any additional arguments must be passed
  as optional keyword arguments. The generated function must take two arguments:
  First, x, the data, then, Ƶ, the latent variables.
- `potential_energy_kwargs::Union{NamedTuple,Dict}`: A named tuple or dictionary
  of keyword arguments to pass to the potential_energy function. Default is a
  tuple with decoder_dist=MvDiagGaussianDecoder and prior=SphericalPrior.

# Returns
- `∇U::Function`: A function that takes both `x` and `Ƶ` as arguments, but only
  computes the gradient with respect to `Ƶ`.

# Example
```julia
# Define decoder
decoder = JointLogDecoder(Flux.Chain(Dense(10, 5, relu), Dense(5, 2)))

# Compute the gradient of the potential energy
∇U = ∇potential_energy(decoder)

# Define data x and latent variable z
x = rand(2)
Ƶ = rand(2)

# Use the function ∇U to compute the gradient of the potential energy
gradient = ∇U(x, Ƶ)
```
"""
function ∇potential_energy(
    decoder::AbstractVariationalDecoder;
    potential_energy::Function=potential_energy,
    potential_energy_kwargs::Union{NamedTuple,Dict}=(
        decoder_dist=MvDiagGaussianDecoder,
        prior=SphericalPrior
    )
)
    # Build potential energy function
    U = potential_energy(decoder; potential_energy_kwargs...)

    # Define gradient of potential energy function
    ∇U(x::AbstractVector{Float32}, Ƶ::AbstractVector{Float32}) = first(
        Zygote.gradient(Ƶ -> U(x, Ƶ), Ƶ)
    )

    return ∇U
end # function

# ==============================================================================
# Hamiltonian Dynamics
# ==============================================================================

@doc raw"""
    leapfrog_step(
        x::AbstractVector{T}, 
        z::AbstractVector{T}, 
        ρ::AbstractVector{T}, 
        ϵ::Union{T,AbstractVector{T}},
        ∇U::Function
    ) where {T<:AbstractFloat}

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm. The
leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The leapfrog step consists of three
parts: a half-step update of the momentum, a full-step update of the position,
and another half-step update of the momentum.

# Arguments
- `x::AbstractVector{T}`: The data that defines the potential energy function
  used in the HMC algorithm.
- `z::AbstractVector{T}`: The position variable in the HMC algorithm,
  representing the latent variable.
- `ρ::AbstractVector{T}`: The momentum variable in the HMC algorithm.
- `ϵ::Union{T,AbstractArray{T}}`: The leapfrog step size. This can be a scalar
  used for all dimensions, or an array of the same size as `z` used for each
  dimension.
- `∇U::Function`: The gradient of the potential energy function used in the HMC
  algorithm. This function must take two inputs: First, `x`, the data, then,
  `Ƶ`, the latent variables.

# Returns
- `z̄::AbstractVector{T}`: The updated position variable.
- `ρ̄::AbstractVector{T}`: The updated momentum variable.

# Example
```julia
# Define position, momentum, step size, and gradient of potential energy
z = rand(2)
ρ = rand(2)
ϵ = rand(2)
∇U = ∇potential_energy(hvae, potential_energy, (decoder_dist=MvDiagGaussianDecoder, prior=SphericalPrior))

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(z, ρ, ϵ, ∇U)
```
"""
function leapfrog_step(
    x::AbstractVector{T},
    z::AbstractVector{T},
    ρ::AbstractVector{T},
    ϵ::Union{T,<:AbstractArray{T}},
    ∇U::Function
) where {T<:AbstractFloat}
    # Update momentum variable with half-step
    ρ̃ = ρ - ϵ .* ∇U(x, z) / 2

    # Update position variable with full-step
    z̄ = z + ϵ .* ρ̃

    # Update momentum variable with half-step
    ρ̄ = ρ̃ - ϵ .* ∇U(x, z̄) / 2

    return z̄, ρ̄
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    leapfrog_step(
        x::AbstractMatrix{T}, 
        z::AbstractMatrix{T}, 
        ρ::AbstractMatrix{T}, 
        ϵ::Union{T,AbstractVector{T}},
        ∇U::Function
    ) where {T<:AbstractFloat}

Perform a single leapfrog step in Hamiltonian Monte Carlo (HMC) algorithm for
each column of the input matrices. The leapfrog step is a numerical integration
scheme used in HMC to simulate the dynamics of a physical system (the position
`z` and momentum `ρ` variables) under a potential energy function `U`. The
leapfrog step consists of three parts: a half-step update of the momentum, a
full-step update of the position, and another half-step update of the momentum.

# Arguments
- `x::AbstractMatrix{T}`: The input data. Each column is treated as a separate
  vector of data.
- `z::AbstractMatrix{T}`: The position variables in the HMC algorithm,
  representing the latent variables. Each column corresponds to the position
  variable for the corresponding column in `x`.
- `ρ::AbstractMatrix{T}`: The momentum variables in the HMC algorithm. Each
  column corresponds to the momentum variable for the corresponding column in
  `x`.
- `ϵ::Union{T,AbstractArray{T}}`: The step size for the HMC algorithm. This can
  be a scalar or an array.
- `∇U::Function`: The gradient of the potential energy function used in the HMC
  algorithm. This function must take two inputs: First, `x`, the data, then,
  `Ƶ`, the latent variables.

# Returns
- `z̄::AbstractMatrix{T}`: The updated position variables. Each column
  corresponds to the updated position variable for the corresponding column in
  `x`.
- `ρ̄::AbstractMatrix{T}`: The updated momentum variables. Each column
  corresponds to the updated momentum variable for the corresponding column in
  `x`.

# Example
```julia
# Define a decoder
decoder = JointDecoder(
    Flux.Chain(Dense(10, 5, relu), Dense(5, 2)),
    Flux.Dense(2, 2), Flux.Dense(2, 2)
)

# Define input data, position, momentum, and step size
x = rand(2, 100)
z = rand(2, 100)
ρ = rand(2, 100)
ϵ = 0.01

# Perform a leapfrog step
z̄, ρ̄ = leapfrog_step(decoder, x, z, ρ, ϵ, potential_energy, Dict())
```
"""
function leapfrog_step(
    x::AbstractMatrix{T},
    z::AbstractMatrix{T},
    ρ::AbstractMatrix{T},
    ϵ::Union{T,<:AbstractArray{T}},
    ∇U::Function
) where {T<:AbstractFloat}
    # Initialize matrices for updated position and momentum variables
    z̄ = similar(z)
    ρ̄ = similar(ρ)

    # Apply leapfrog_step to each column
    for i in axes(z, 2)
        z̄[:, i], ρ̄[:, i] = leapfrog_step(x[:, i], z[:, i], ρ[:, i], ϵ, ∇U)
    end

    return z̄, ρ̄
end

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
# Combining Leapfrog and Tempering Steps
# ==============================================================================

@doc raw"""
        leapfrog_tempering_step(
            x::AbstractVector{T},
            zₒ::AbstractVector{T},
            K::Int,
            ϵ::Union{T,<:AbstractVector{T}},
            βₒ::T,
            ∇U::Function;
            tempering_schedule::Function=quadratic_tempering,
        ) where {T<:AbstractFloat}

Combines the leapfrog and tempering steps into a single function for the
Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `x::AbstractVector{T}`: The data to be processed.
- `zₒ::AbstractVector{T}`: The initial latent variable.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.

# Optional Keyword Arguments
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys:
    - `z_init`: The initial latent variable.
    - `ρ_init`: The initial momentum variable.
    - `z_final`: The final latent variable after `K` leapfrog steps.
    - `ρ_final`: The final momentum variable after `K` leapfrog steps.

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
    x::AbstractVector{T},
    zₒ::AbstractVector{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T,
    ∇U::Function;
    tempering_schedule::Function=quadratic_tempering,
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

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(x, zₖ₋₁, ρₖ₋₁, ϵ, ∇U)

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
    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ,
        ρ_final=ρₖ,
    )
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    leapfrog_tempering_step(
        leapfrog_tempering_step(
            x::AbstractMatrix{T},
            zₒ::AbstractMatrix{T},
            K::Int,
            ϵ::Union{T,<:AbstractVector{T}},
            βₒ::T,
            ∇U::Function;
            tempering_schedule::Function=quadratic_tempering,
        ) where {T<:AbstractFloat}

Perform a sequence of leapfrog steps followed by a tempering step in the
Hamiltonian Monte Carlo (HMC) algorithm for each column of the input matrices.
The leapfrog step is a numerical integration scheme used in HMC to simulate the
dynamics of a physical system (the position `z` and momentum `ρ` variables)
under a potential energy function `U`. The tempering step adjusts the momentum
variable according to a tempering schedule.

# Arguments
- `x::AbstractMatrix{T}`: The input data. Each column is treated as a separate
  vector of data.
- `zₒ::AbstractMatrix{T}`: The initial position variables in the HMC algorithm,
  representing the latent variables. Each column corresponds to the position
  variable for the corresponding column in `x`.
- `K::Int`: The number of leapfrog steps to perform.
- `ϵ::Union{T,AbstractArray{T}}`: The step size for the HMC algorithm. This can
  be a scalar or an array.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.

# Optional Keyword Arguments
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys:
    - `z_init`: The initial latent variable.
    - `ρ_init`: The initial momentum variable.
    - `z_final`: The final latent variable after `K` leapfrog steps.
    - `ρ_final`: The final momentum variable after `K` leapfrog steps.

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
    x::AbstractMatrix{T},
    zₒ::AbstractMatrix{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T,
    ∇U::Function;
    tempering_schedule::Function=quadratic_tempering,
) where {T<:AbstractFloat}
    # Sample γₒ ~ N(0, I)
    γₒ = Random.rand(SphericalPrior(zₒ[:, 1]), size(zₒ, 2))
    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value of z and ρ before loop
    zₖ₋₁, ρₖ₋₁ = zₒ, ρₒ

    # Initialize variables to have them available outside the for loop
    zₖ = Matrix{T}(undef, size(zₒ)...)
    ρₖ = Matrix{T}(undef, size(ρₒ)...)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ = leapfrog_step(x, zₖ₋₁, ρₖ₋₁, ϵ, ∇U)

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

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ,
        ρ_final=ρₖ,
    )
end # function

# ==============================================================================
# Forward pass methods for HVAE with Hamiltonian steps
# ==============================================================================

@doc raw"""
        (hvae::HVAE{VAE{JointLogEncoder,SimpleDecoder}})(
            x::AbstractVector{T},
            K::Int,
            ϵ::Union{T,<:AbstractVector{T}},
            βₒ::T;
            tempering_schedule::Function=quadratic_tempering,
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
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to `quadratic_tempering`..
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  variables. Defaults to a standard normal distribution.  
- `latent::Bool`: Whether to return the latent variables. If `true`, the
  function returns a dictionary containing the encoder mean and log standard
  deviation, the initial and final latent variables, and the decoder mean. If
  `false`, the function returns the decoder mean.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.

# Returns
- If `latent` is `true`, returns a `Dict` with the following keys: 
    - `:encoder_µ`: The mean of the encoder's output distribution. 
    - `:encoder_logσ`: The log standard deviation of the encoder's output
      distribution. 
    - `:z_init`: The initial latent variable. 
    - `:ρ_init`: The initial momentum variable. 
    - `:z_final`: The final latent variable after `K` leapfrog steps. 
    - `:ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `:decoder_µ`: The mean of the decoder's output distribution.
- If `latent` is `false`, returns the mean of the decoder's output distribution.

# Description
The function first runs the input data through the encoder to obtain the mean
and log standard deviation. It then uses the reparameterization trick to
generate an initial latent variable. Next, it performs `K` leapfrog steps, each
followed by a tempering step, to generate a new sample from the latent space.
Finally, it runs the final latent variable through the decoder to obtain the
output data.

# Notes
- Ensure the input data `x` matches the expected input dimensionality for the
HVAE model.
- The leapfrog steps are ignored by Zygote.jl, so the gradients of the HVAE do
  not include the gradients of the leapfrog steps.
"""
function (hvae::HVAE{VAE{JointLogEncoder,SimpleDecoder}})(
    x::AbstractVecOrMat{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T,
    ∇U::Function;
    tempering_schedule::Function=quadratic_tempering,
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
    step_dict = Zygote.ignore() do
        leapfrog_tempering_step(
            x, zₒ, K, ϵ, βₒ, ∇U;
            tempering_schedule=tempering_schedule,
        )
    end # Zygote.ignore

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder and collect outputs in NamedTuple
        return merge(
            (
                encoder_µ=encoder_µ,
                encoder_logσ=encoder_logσ,
                decoder_µ=hvae.vae.decoder(step_dict.z_final),
            ),
            step_dict
        )
    else
        # Run latent sample through decoder
        return hvae.vae.decoder(step_dict.z_final)
    end # if
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        (hvae::HVAE{VAE{JointLogEncoder,D}})(
            x::AbstractVector{T},
            K::Int,
            ϵ::Union{T,<:AbstractVector{T}},
            βₒ::T;
            tempering_schedule::Function=quadratic_tempering,
            prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
            latent::Bool=false,
        ) where {D<:AbstractGaussianLogDecoder,T<:Float32}

Processes the input data `x` through a Hamiltonian Variational Autoencoder
(HVAE), consisting of a `JointLogEncoder` and an
`AbstractGaussianLogDecoder`.

# Arguments
- `hvae::HVAE{VAE{JointLogEncoder,D}}`: The HVAE model.
- `x::AbstractVector{T}`: The data to be processed. This should be a vector of
  type `T`.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.

# Optional Keyword Arguments
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`..  
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
    - `:ρ_init`: The initial momentum variable. 
    - `:z_final`: The final latent variable after `K` leapfrog steps. 
    - `:ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `:decoder_µ`: The mean of the decoder's output distribution.
    - `:decoder_logσ`: The log standard deviation of the decoder's output
      distribution.
- If `latent` is `false`, returns the mean and log standard deviation the
  decoder's output distribution.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution. Then, it performs `K` steps of HMC with tempering to
generate a new sample from the latent space, which is then decoded.

# Notes
- Ensure the input data `x` matches the expected input dimensionality for the
HVAE model.
- The leapfrog steps are ignored by Zygote.jl, so the gradients of the HVAE do
  not include the gradients of the leapfrog steps.
"""
function (hvae::HVAE{VAE{JointLogEncoder,D}})(
    x::AbstractVecOrMat{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T,
    ∇U::Function;
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
) where {D<:AbstractGaussianLogDecoder,T<:Float32}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = hvae.vae.encoder(x)

    # Run reparametrization trick to generate latent variable zₒ
    zₒ = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=1, log=true
    )

    # Run leapfrog and tempering steps
    step_dict = Zygote.ignore() do
        leapfrog_tempering_step(
            x, zₒ, K, ϵ, βₒ, ∇U;
            tempering_schedule=tempering_schedule,
        )
    end # Zygote.ignore

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder to obtain mean and log std
        decoder_µ, decoder_logσ = hvae.vae.decoder(step_dict.z_final)
        # Collect outputs in NamedTuple
        return merge(
            (
                encoder_µ=encoder_µ,
                encoder_logσ=encoder_logσ,
                decoder_µ=decoder_µ,
                decoder_logσ=decoder_logσ
            ),
            step_dict
        )
    else
        # Run latent sample through decoder
        return hvae.vae.decoder(step_dict.z_final)
    end # if
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        ((hvae::HVAE{VAE{JointLogEncoder,D}})(
            x::AbstractVector{T},
            K::Int,
            ϵ::Union{T,<:AbstractVector{T}},
            βₒ::T;
            tempering_schedule::Function=quadratic_tempering,
            prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
            latent::Bool=false,
        ) where {D<:AbstractGaussianLinearDecoder,T<:Float32}

Processes the input data `x` through a Hamiltonian Variational Autoencoder
(HVAE), consisting of a `JointLogEncoder` and an
`AbstractGaussianLinearDecoder`.

# Arguments
- `hvae::HVAE{VAE{JointLogEncoder,D}}`: The HVAE model.
- `x::AbstractVector{T}`: The data to be processed. This should be a vector of
  type `T`.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.

# Optional Keyword Arguments
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`..  
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
    - `:ρ_init`: The initial momentum variable. 
    - `:z_final`: The final latent variable after `K` leapfrog steps. 
    - `:ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `:decoder_µ`: The mean of the decoder's output distribution.
    - `:decoder_σ`: The standard deviation of the decoder's output distribution.
- If `latent` is `false`, returns the mean and standard deviation the decoder's
  output distribution.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution. Then, it performs `K` steps of HMC with tempering to
generate a new sample from the latent space, which is then decoded.

# Notes
- Ensure the input data `x` matches the expected input dimensionality for the
HVAE model.
- The leapfrog steps are ignored by Zygote.jl, so the gradients of the HVAE do
  not include the gradients of the leapfrog steps.
"""
function (hvae::HVAE{VAE{JointLogEncoder,D}})(
    x::AbstractVecOrMat{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T,
    ∇U::Function;
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
) where {D<:AbstractGaussianLinearDecoder,T<:Float32}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = hvae.vae.encoder(x)

    # Run reparametrization trick to generate latent variable zₒ
    zₒ = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=1, log=false
    )

    # Run leapfrog and tempering steps
    step_dict = Zygote.ignore() do
        leapfrog_tempering_step(
            x, zₒ, K, ϵ, βₒ, ∇U;
            tempering_schedule=tempering_schedule,
        )
    end # Zygote.ignore

    # Check if latent variables should be returned
    if latent
        # Run latent sample through decoder to obtain mean and log std
        decoder_µ, decoder_σ = hvae.vae.decoder(step_dict.z_final)
        # Collect outputs in NamedTuple
        return merge(
            (
                encoder_µ=encoder_µ,
                encoder_logσ=encoder_logσ,
                decoder_µ=decoder_µ,
                decoder_σ=decoder_σ
            ),
            step_dict
        )
    else
        # Run latent sample through decoder
        return hvae.vae.decoder(step_dict.z_final)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    (hvae::HVAE{VAE{JointLogEncoder,D}})(
        x::AbstractVecOrMat{T},
        K::Int,
        ϵ::Union{T,<:AbstractVector{T}},
        βₒ::T;
        tempering_schedule::Function=quadratic_tempering,
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
        latent::Bool=false,
    ) where {D<:AbstractVariationalDecoder,T<:Float32}

Processes the input data `x` through a Hamiltonian Variational Autoencoder
(HVAE), consisting of a `JointLogEncoder` and an `AbstractVariationalDecoder`.

# Arguments
- `hvae::HVAE{VAE{JointLogEncoder,D}}`: The HVAE model.
- `x::AbstractVecOrMat{T}`: The data to be processed. This should be a vector or
  matrix of type `T`.
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm.
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog steps in the
  HMC algorithm. This can be a scalar or a vector.
- `βₒ::T`: The initial inverse temperature for the tempering schedule.

# Optional Keyword Arguments
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`.
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
    - `:ρ_init`: The initial momentum variable. 
    - `:z_final`: The final latent variable after `K` leapfrog steps. 
    - `:ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `:decoder_µ`: The mean of the decoder's output distribution.
    - `:decoder_(log)σ`: (depends on decoder) The (log) standard deviation of
      the decoder's output distribution.
- If `latent` is `false`, returns the decoder's output.

# Description
The function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. Using the reparametrization trick, it samples
from this distribution. Then, it performs `K` steps of HMC with tempering to
generate a new sample from the latent space, which is then decoded. If the
gradient function `∇U` is not provided, a default one is created using the
`∇potential_energy` function with the decoder of the `hvae` as argument.

# Notes
- Ensure the input data `x` matches the expected input dimensionality for the
  HVAE model.
- The leapfrog steps are ignored by Zygote.jl, so the gradients of the HVAE do
  not include the gradients of the leapfrog steps.
"""
function (hvae::HVAE{VAE{JointLogEncoder,D}})(
    x::AbstractVecOrMat{T},
    K::Int,
    ϵ::Union{T,<:AbstractVector{T}},
    βₒ::T;
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
) where {D<:AbstractVariationalDecoder,T<:Float32}
    # Build gradient function with default arguments
    ∇U = ∇potential_energy(hvae.vae.decoder)

    # Call previously defined methods as needed
    return hvae(
        x, K, ϵ, βₒ, ∇U;
        tempering_schedule=tempering_schedule,
        prior=prior,
        latent=latent
    )
end # function

# ==============================================================================
# Hamiltonian ELBO
# ==============================================================================

@doc raw"""
    hamiltonian_elbo(
        hvae::HVAE,
        ∇U::Function,
        x::AbstractVector{T};
        K::Int=3,
        ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
        βₒ::T=0.3f0,
        potential_energy::Function=potential_energy,
        potential_energy_kwargs::Dict=Dict(
            :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
        ),
        tempering_schedule::Function=quadratic_tempering,
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0)
    ) where {T<:AbstractFloat}

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the log evidence
estimate `log p̄` and the log variational estimate `log q̄`.

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `U::Function`: The potential energy. This function must takes both `x` and `z`
  as arguments.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.
- `x::AbstractVector{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `potential_energy::Function`: The potential energy function used in the HMC
  (default is `potential_energy`).
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments for the
  potential energy function (default is `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  variables. Defaults to a standard normal distribution.
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
  `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.

# Returns
- `elbo::T`: The HMC estimate of the ELBO.

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

# Define an HVAE
hvae = HVAE(vae)

# Define input data
x = rand(Float32, 784)

# Compute the Hamiltonian ELBO
elbo = hamiltonian_elbo(hvae, x, K=3, ϵ=0.001f0, βₒ=0.3f0)
```
"""
function hamiltonian_elbo(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    U::Function,
    ∇U::Function,
    x::AbstractVector{T};
    K::Int=3,
    ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
    βₒ::T=0.3f0,
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    return_outputs::Bool=false,
) where {T<:AbstractFloat}
    # Forward Pass (run input through reconstruct function)
    hvae_outputs = hvae(
        x, K, ϵ, βₒ, ∇U;
        tempering_schedule=tempering_schedule,
        prior=prior,
        latent=true
    )

    # Unpack position and momentum variables
    zₒ = hvae_outputs.z_init
    zₖ = hvae_outputs.z_final
    ρₒ = hvae_outputs.ρ_init
    ρₖ = hvae_outputs.ρ_final

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    # log p̄ = - U(x, zₖ) + log p(ρₖ)
    log_p̄ = -U(x, zₖ) + Distributions.logpdf(SphericalPrior(ρₖ), ρₖ)

    # log q̄ = log q(zₒ) + log p(ρₒ)
    log_q̄ = Distributions.logpdf(
        Distributions.MvNormal(
            hvae_outputs.encoder_µ, hvae_outputs.encoder_logσ
        ),
        zₒ
    ) + Distributions.logpdf(SphericalPrior(ρₒ, βₒ^-1), ρₒ) -
             0.5f0 * length(zₒ) * log(βₒ)

    if return_outputs
        return log_p̄ - log_q̄, hvae_outputs
    else
        return log_p̄ - log_q̄
    end # if
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    hamiltonian_elbo(
        hvae::HVAE,
        U::Function,
        ∇U::Function,
        x::AbstractVector{T};
        K::Int=3,
        ϵ::Union{T,<:AbstractVector{T}}=0.001f0,
        βₒ::T=0.3f0,
        potential_energy::Function=potential_energy,
        potential_energy_kwargs::Dict=Dict(
            :decoder_dist => MvDiagGaussianDecoder, :prior => SphericalPrior
        ),
        tempering_schedule::Function=quadratic_tempering,
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0)
    ) where {T<:AbstractFloat}

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the log evidence
estimate `log p̄` and the log variational estimate `log q̄`.

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `U::Function`: The potential energy. This function must takes both `x` and `z`
  as arguments.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.
- `x::AbstractMatrix{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`. Each column represents a single data point.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{T,<:AbstractVector{T}}`: The step size for the leapfrog integrator
  (default is 0.001).
- `βₒ::T`: The initial inverse temperature (default is 0.3).
- `potential_energy::Function`: The potential energy function used in the HMC
  (default is `potential_energy`).
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments for the
  potential energy function (default is `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  variables. Defaults to a standard normal distribution.
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults
  to `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.

# Returns
- `elbo::T`: The HMC estimate of the ELBO.

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

# Define an HVAE
hvae = HVAE(vae)

# Define input data
x = rand(Float32, 784)

# Compute the Hamiltonian ELBO
elbo = hamiltonian_elbo(hvae, x, K=3, ϵ=0.001f0, βₒ=0.3f0)
```
"""
function hamiltonian_elbo(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    U::Function,
    ∇U::Function,
    x::AbstractMatrix{Float32};
    K::Int=3,
    ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
    βₒ::Float32=0.3f0,
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    return_outputs::Bool=false,
)
    # Forward Pass (run input through reconstruct function)
    hvae_outputs = hvae(
        x, K, ϵ, βₒ, ∇U;
        tempering_schedule=tempering_schedule,
        prior=prior,
        latent=true
    )

    # Unpack position and momentum variables
    zₒ = hvae_outputs.z_init
    zₖ = hvae_outputs.z_final
    ρₒ = hvae_outputs.ρ_init
    ρₖ = hvae_outputs.ρ_final

    # Initialize value to save ELBO
    elbo = zero(Float32)

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # Loop through each column of input data
    for i in axes(x, 2)
        # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
        # log p̄ = - U(x, zₖ) + log p(ρₖ)
        log_p̄ = -U(x[:, i], zₖ[:, i]) + Distributions.logpdf(
            SphericalPrior(ρₖ[:, i]), ρₖ[:, i]
        )

        # log q̄ = log q(zₒ) + log p(ρₒ)
        log_q̄ = Distributions.logpdf(
            Distributions.MvNormal(
                hvae_outputs.encoder_µ[:, i],
                hvae_outputs.encoder_logσ[:, i]
            ),
            zₒ[:, i]
        ) + Distributions.logpdf(SphericalPrior(ρₒ[:, i], βₒ^-1), ρₒ[:, i]) -
                 0.5f0 * size(zₒ, 1) * log(βₒ)

        # Update ELBO
        elbo += log_p̄ - log_q̄
    end # for

    if return_outputs
        return elbo / size(x, 2), hvae_outputs
    else
        # Return ELBO normalized by number of samples
        return elbo / size(x, 2)
    end # if
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
        hamiltonian_elbo(
            hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
            x::AbstractVecOrMat{Float32};
            K::Int=3,
            ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
            βₒ::Float32=0.3f0,
            tempering_schedule::Function=quadratic_tempering,
            prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
            return_outputs::Bool=false,
        )

Compute the Hamiltonian Monte Carlo (HMC) estimate of the evidence lower bound
(ELBO) for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the log evidence
estimate `log p̄` and the log variational estimate `log q̄`.

If no potential energy `U` and its gradient `∇U` functions are provided, it
creates default ones using the decoder of the `hvae`.

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
    space.
- `x::AbstractVecOrMat{Float32}`: The input data. If a vector is provided, that
  represents a single datum. If a column is provided, each column represents a
  single data point.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{Float32,<:AbstractVector{Float32}}`: The step size for the leapfrog
    integrator (default is 0.001).
- `βₒ::Float32`: The initial inverse temperature (default is 0.3).
- `tempering_schedule::Function`: The tempering schedule function used in the
    HMC (default is `quadratic_tempering`).
- `prior::Distributions.Sampleable`: The prior distribution for the latent
    variables. Defaults to a standard normal distribution.
- `return_outputs::Bool`: Whether to return the outputs of the HVAE. Defaults to
    `false`. NOTE: This is necessary to avoid computing the forward pass twice
    when computing the loss function with regularization.

# Returns
- `elbo::Float32`: The HMC estimate of the ELBO.

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

# Define an HVAE
hvae = HVAE(vae)

# Define input data
x = rand(Float32, 784)

# Compute the Hamiltonian ELBO
elbo = hamiltonian_elbo(hvae, x, K=3, ϵ=0.001f0, βₒ=0.3f0)
```
"""
function hamiltonian_elbo(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    x::AbstractVecOrMat{Float32};
    K::Int=3,
    ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
    βₒ::Float32=0.3f0,
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    return_outputs::Bool=false,
)
    # Define default potential energy
    U = potential_energy(hvae.vae.decoder)
    # Define default gradient of potential energy
    ∇U = ∇potential_energy(hvae.vae.decoder)

    # Call previously defined methods as needed
    return hamiltonian_elbo(
        hvae, U, ∇U, x;
        K=K, ϵ=ϵ, βₒ=βₒ,
        tempering_schedule=tempering_schedule,
        prior=prior,
        return_outputs=return_outputs
    )
end # function

# ==============================================================================
# Define HVAE loss function
# ==============================================================================

"""
    loss(
        hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
        U::Function,
        ∇U::Function,
        x::AbstractMatrix{Float32};
        K::Int=3,
        ϵ::Union{T,<:AbstractVector{Float32}}=0.001f0,
        βₒ::Float32=0.3f0,
        potential_energy::Function=potential_energy,
        potential_energy_kwargs::Union{NamedTuple,Dict}=(
            decoder_dist=MvDiagGaussianDecoder, prior=SphericalPrior
        ),
        tempering_schedule::Function=quadratic_tempering,
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::Float32=1.0f0
    )

Compute the loss for a Hamiltonian Variational Autoencoder (HVAE).

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
  space.
- `U::Function`: The potential energy. This function must takes both `x` and `z`
  as arguments.
- `∇U::Function`: The gradient function of the potential energy. This function
  must takes both `x` and `z` as arguments, but only computes the gradient with
  respect to `z`.
- `x::AbstractVecOrMat{T}`: The input data, where `T` is a subtype of
  `AbstractFloat`. If vector, the function assumes a single data point. If
  matrix, the function assumes a batch of data points.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{Float32,<:AbstractVector{Float32}}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Float32`: The initial inverse temperature (default is 0.3).
- `potential_energy::Function`: The potential energy function used in the HMC
  (default is `potential_energy`).
- `potential_energy_kwargs::Dict`: A dictionary of keyword arguments for the
  potential energy function (default is `Dict(:decoder_dist =>
  MvDiagGaussianDecoder, :prior => SphericalPrior)`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `prior::Distributions.Sampleable`: The prior distribution for the latent
  variables. Defaults to a standard normal distribution.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. Should return a Float32. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- The computed loss.
"""
function loss(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    U::Function,
    ∇U::Function,
    x::AbstractVecOrMat{Float32};
    K::Int=3,
    ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
    βₒ::Float32=0.3f0,
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0
)
    # Check if there is regularization
    if reg_function !== nothing
        # Compute ELBO and regularization term
        elbo, hvae_outputs = hamiltonian_elbo(
            hvae, U, ∇U, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            tempering_schedule=tempering_schedule,
            prior=prior,
            return_outputs=true
        )

        # Compute regularization
        reg_term = reg_function(hvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -hamiltonian_elbo(
            hvae, U, ∇U, x;
            K=K, ϵ=ϵ, βₒ=βₒ,
            tempering_schedule=tempering_schedule,
            prior=prior
        )
    end # if
end # function

@doc raw"""
        loss(
            hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
            x::AbstractVecOrMat{Float32};
            K::Int=3,
            ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
            βₒ::Float32=0.3f0,
            tempering_schedule::Function=quadratic_tempering,
            prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
            reg_function::Union{Function,Nothing}=nothing,
            reg_kwargs::Union{NamedTuple,Dict}=Dict(),
            reg_strength::Float32=1.0f0
        )

Compute the loss for a Hamiltonian Variational Autoencoder (HVAE).

This function takes as input an HVAE and a vector of input data `x`. It performs
`K` HMC steps with a leapfrog integrator and a tempering schedule to estimate
the ELBO. The ELBO is computed as the difference between the log evidence
estimate `log p̄` and the log variational estimate `log q̄`.

If no potential energy `U` and its gradient `∇U` functions are provided, it creates default ones using the decoder of the `hvae`.

# Arguments
- `hvae::HVAE`: The HVAE used to encode the input data and decode the latent
    space.
- `x::AbstractVecOrMat{Float32}`: The input data. Each column represents a single data point.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{Float32,<:AbstractVector{Float32}}`: The step size for the leapfrog integrator
    (default is 0.001).
- `βₒ::Float32`: The initial inverse temperature (default is 0.3).
- `tempering_schedule::Function`: The tempering schedule function used in the
    HMC (default is `quadratic_tempering`).
- `prior::Distributions.Sampleable`: The prior distribution for the latent
    variables. Defaults to a standard normal distribution.
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
    regularization term based on the VAE outputs. Should return a Float32. This
    function must take as input the VAE outputs and the keyword arguments provided
    in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
    regularization function.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- The computed loss.
"""
function loss(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    x::AbstractVecOrMat{Float32};
    K::Int=3,
    ϵ::Union{Float32,<:AbstractVector{Float32}}=0.001f0,
    βₒ::Float32=0.3f0,
    tempering_schedule::Function=quadratic_tempering,
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Float32=1.0f0
)
    # Define default potential energy
    U = potential_energy(hvae.vae.decoder)
    # Define default gradient of potential energy
    ∇U = ∇potential_energy(hvae.vae.decoder)

    # Call previously defined methods as needed
    return loss(
        hvae, U, ∇U, x;
        K=K, ϵ=ϵ, βₒ=βₒ,
        tempering_schedule=tempering_schedule,
        prior=prior,
        reg_function=reg_function,
        reg_kwargs=reg_kwargs,
        reg_strength=reg_strength
    )
end # function

# ==============================================================================
# HVAE training
# ==============================================================================

@doc raw"""
        `train!(hvae, x, opt; loss_function, loss_kwargs)`

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}}`: A struct
  containing the elements of a Hamiltonian Variational Autoencoder.
- `x::AbstractVecOrMat{Float32}`: Data on which to evaluate the loss function.
  Columns represent individual samples.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Train.setup`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Union{NamedTuple,Dict} = Dict()`: Arguments for the loss
  function. These might include parameters like `K`, `ϵ`, `βₒ`,
  `potential_energy`, `potential_energy_kwargs`, `tempering_schedule`, `prior`,
  `reg_function`, `reg_kwargs`, `reg_strength`, depending on the specific loss
  function in use.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.

# Examples
```julia
opt = Flux.setup(Optax.adam(1e-3), hvae)
for x in dataloader
        train!(hvae, x, opt; loss_fn, loss_kwargs=Dict(:K => 3, :ϵ => 0.001f0, :βₒ => 0.3f0))
end
```
"""
function train!(
    hvae::HVAE{<:VAE{<:JointLogEncoder,<:AbstractVariationalDecoder}},
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Dict=Dict()
)
    # Compute VAE gradient
    ∇loss_ = Flux.gradient(hvae) do hvae_model
        loss_function(hvae_model, x; loss_kwargs...)
    end # do block
    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇loss_[1])
end # function