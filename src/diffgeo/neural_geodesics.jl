
# Import library for automatic differentiation
import Zygote
import TaylorDiff

# Import ML library
import Flux
import NNlib

# Import library to use Ellipsis Notation
using EllipsisNotation

# Import abstract types from RHVAEs
using ..RHVAEs: RHVAE

# Import functions from RHVAEs module
using ..RHVAEs: G_inv

# Import functions from utils module
using ..utils: taylordiff_gradient

# ==============================================================================
# > Chen, N. et al. Metrics for Deep Generative Models. in Proceedings of the
# > Twenty-First International Conference on Artificial Intelligence and
# > Statistics 1540–1550 (PMLR, 2018).
# ==============================================================================

@doc raw"""
    NeuralGeodesic

Type to define a neural network that approximates a geodesic curve on a
Riemanian manifold. If a curve γ̲(t) represents a geodesic curve on a manifold,
i.e.,

    L(γ̲) = min_γ ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where M̲̲ is the Riemmanian metric, then this type defines a neural network
g_ω(t) such that

    γ̲(t) ≈ g_ω(t).

This neural network must have a single input (1D). The dimensionality of the
output must match the dimensionality of the manifold.

# Fields
- `mlp::Flux.Chain`: Neural network that approximates the geodesic curve. The
  dimensionality of the input must be one.
- `z_init::AbstractVector`: Initial position of the geodesic curve on the latent
  space.
- `z_end::AbstractVector`: Final position of the geodesic curve on the latent
  space.

# Citation
> Chen, N. et al. Metrics for Deep Generative Models. in Proceedings of the
> Twenty-First International Conference on Artificial Intelligence and
> Statistics 1540–1550 (PMLR, 2018).
"""
struct NeuralGeodesic
    mlp::Flux.Chain
    z_init::AbstractVector
    z_end::AbstractVector
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor NeuralGeodesic (mlp,)

# ------------------------------------------------------------------------------

@doc raw"""
    (g::NeuralGeodesic)(t)

Computes the output of the NeuralGeodesic at a given time `t` by scaling and
shifting the output of the neural network.

# Arguments
- `t`: The time at which the output of the NeuralGeodesic is to be computed.

# Returns
- `output::Array`: The computed output of the NeuralGeodesic at time `t`.

# Description
The function computes the output of the NeuralGeodesic at a given time `t`. The
steps are:

1. Compute the output of the neural network at time t.
2. Compute the output of the neural network at time 0 and 1.
3. Compute scale and shift parameters based on the initial and end points of the
   geodesic and the neural network outputs at times 0 and 1.
4. Scale and shift the output of the neural network at time t according to these
parameters. The result is the output of the NeuralGeodesic at time t.

Scale and shift parameters are defined as:

- scale = (z_init - z_end) / (ẑ_init - ẑ_end)
- shift = (z_init * ẑ_end - z_end * ẑ_init) / (ẑ_init - ẑ_end)

where z_init and z_end are the initial and end points of the geodesic, and
ẑ_init and ẑ_end are the outputs of the neural network at times 0 and 1,
respectively.

# Note
Ensure that `t ∈ [0, 1]`.
"""
function (g::NeuralGeodesic)(t)
    # Check that t is within the interval [0, 1]
    if t < 0.0f0 - 2 * cbrt(eps(Float32)) || t > 1.0f0 + 2 * cbrt(eps(Float32))
        throw(ArgumentError("t must be within the interval [0, 1]."))
    end

    # Compute the output of the neural network at time t
    g_t = g.mlp([t])

    # Compute the output of the neural network at time 0 and 1
    ẑ_init = g.mlp([zero(Float32)])
    ẑ_end = g.mlp([one(Float32)])

    # Compute scale and shift parameters
    scale = (g.z_init - g.z_end) ./ (ẑ_init - ẑ_end)
    shift = (g.z_init .* ẑ_end - g.z_end .* ẑ_init) ./ (ẑ_init - ẑ_end)

    # Return shifted and scaled output
    return scale .* g_t .- shift
end

# ------------------------------------------------------------------------------

@doc raw"""
        (g::NeuralGeodesic)(t::AbstractArray)

Computes the output of the NeuralGeodesic at each given time in `t` by scaling
and shifting the output of the neural network.

# Arguments
- `t::AbstractArray`: An array of times at which the output of the
  NeuralGeodesic is to be computed. This must be within the interval [0, 1].

# Returns
- `output::Array`: The computed output of the NeuralGeodesic at each time in
  `t`.

# Description
The function computes the output of the NeuralGeodesic at each given time in
`t`. The steps are:

1. Compute the output of the neural network at each time in `t`.
2. Compute the output of the neural network at time 0 and 1.
3. Compute scale and shift parameters based on the initial and end points of the
   geodesic and the neural network outputs at times 0 and 1.
4. Scale and shift the output of the neural network at each time in `t`
   according to these parameters. The result is the output of the NeuralGeodesic
   at each time in `t`.

Scale and shift parameters are defined as:

- scale = (z_init - z_end) / (ẑ_init - ẑ_end)
- shift = (z_init * ẑ_end - z_end * ẑ_init) / (ẑ_init - ẑ_end)

where z_init and z_end are the initial and end points of the geodesic, and
ẑ_init and ẑ_end are the outputs of the neural network at times 0 and 1,
respectively.

# Note
Ensure that each `t` in the array is within the interval [0, 1].
"""
function (g::NeuralGeodesic)(t::AbstractVector{<:Number})
    # Check that every t is within the interval [0, 1]
    if any(t .< 0.0f0 - 2 * cbrt(eps(Float32))) ||
       any(t .> 1 + 2 * cbrt(eps(Float32)))
        throw(ArgumentError("t must be within the interval [0, 1]."))
    end

    # compute the output of the neural network at each time t. This is done by
    # transposing the input and feeding it to the neural network.
    g_t = g.mlp(t')

    # Compute the output of the neural network at time 0 and 1
    ẑ_init = g.mlp([zero(Float32)])
    ẑ_end = g.mlp([one(Float32)])

    # Compute scale and shift parameters
    scale = (g.z_init .- g.z_end) ./ (ẑ_init .- ẑ_end)
    shift = (g.z_init .* ẑ_end .- g.z_end .* ẑ_init) ./ (ẑ_init .- ẑ_end)

    # Return shifted and scaled output
    return scale .* g_t .- shift
end

# ------------------------------------------------------------------------------
# Curve velocity computation
# ------------------------------------------------------------------------------

@doc raw"""
    curve_velocity_TaylorDiff(
        curve::NeuralGeodesic,
        t
    )

Compute the velocity of a neural geodesic curve at a given time using Taylor
differentiation.

This function takes a `NeuralGeodesic` instance and a time `t`, and computes the
velocity of the curve at that time using Taylor differentiation. The computation
is performed for each dimension of the latent space.

# Arguments
- `curve::NeuralGeodesic`: The neural geodesic curve.
- `t`: The time at which to compute the velocity.

# Returns
A vector representing the velocity of the curve at time `t`.

# Notes
This function uses the `TaylorDiff` package to compute derivatives. Please note
that `TaylorDiff` has limited support for certain activation functions. If you
encounter an error while using this function, it may be due to the activation
function used in your `NeuralGeodesic` instance.
"""
function curve_velocity_TaylorDiff(
    curve::NeuralGeodesic,
    t
)
    # Extract dimensionality of latent space
    ldim = size(curve.z_init, 1)

    # Compute TaylorDiff gradient for each element in t
    return [
        begin
            TaylorDiff.derivative(t -> curve(t)[i], t, 1)
        end for i in 1:ldim
    ]
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    curve_velocity_TaylorDiff(
        curve::NeuralGeodesic,
        t::AbstractVector
    )

Compute the velocity of a neural geodesic curve at each time in a vector of
times using Taylor differentiation.

This function takes a `NeuralGeodesic` instance and a vector of times `t`, and
computes the velocity of the curve at each time using Taylor differentiation.
The computation is performed for each dimension of the latent space and each
time in `t`.

# Arguments
- `curve::NeuralGeodesic`: The neural geodesic curve.
- `t::AbstractVector`: The vector of times at which to compute the velocity.

# Returns
A matrix where each column represents the velocity of the curve at a time in
`t`.

# Notes
This function uses the `TaylorDiff` package to compute derivatives. Please note
that `TaylorDiff` has limited support for certain activation functions. If you
encounter an error while using this function, it may be due to the activation
function used in your `NeuralGeodesic` instance.
"""
function curve_velocity_TaylorDiff(
    curve::NeuralGeodesic,
    t::AbstractVector
)
    return reduce(
        hcat,
        begin
            curve_velocity_TaylorDiff(curve, tᵢ)
        end for tᵢ in t
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    curve_velocity_finitediff(
        curve::NeuralGeodesic,
        t::AbstractVector;
        fdtype::Symbol=:central,
    )

Compute the velocity of a neural geodesic curve at each time in a vector of
times using finite difference methods.

This function takes a `NeuralGeodesic` instance, a vector of times `t`, and an
optional finite difference type `fdtype` (which can be either `:forward` or
`:central`), and computes the velocity of the curve at each time using the
specified finite difference method. The computation is performed for each
dimension of the latent space and each time in `t`.

# Arguments
- `curve::NeuralGeodesic`: The neural geodesic curve.
- `t::AbstractVector`: The vector of times at which to compute the velocity.
- `fdtype::Symbol=:central`: The type of finite difference method to use. Can be
  either `:forward` or `:central`. Default is `:central`.

# Returns
A matrix where each column represents the velocity of the curve at a time in
`t`.

# Notes
This function uses finite difference methods to compute derivatives. Please note
that the accuracy of the computed velocities depends on the chosen finite
difference method and the step size used, which is determined by the machine
epsilon of the type of `t`.
"""
function curve_velocity_finitediff(
    curve::NeuralGeodesic,
    t::AbstractVector;
    fdtype::Symbol=:central,
)
    # Check that mode is either :forward or :central
    if !(fdtype in (:forward, :central))
        error("fdtype must be either :forward or :central")
    end

    # Extract dimensionality of latent space
    ldim = size(curve.z_init, 1)

    # Check fdtype
    if fdtype == :forward
        # Define step size
        ε = √(eps(eltype(t)))
        # Compute finite difference derivatives
        dγdt = (curve(t + ε) .- curve(t)) ./ ε
    elseif fdtype == :central
        # Define step size
        ε = ∛(eps(eltype(t)))
        # Compute finite difference derivatives
        dγdt = (curve(t .+ ε) .- curve(t .- ε)) ./ (2 * ε)
    end # if

    return dγdt
end # function

# ------------------------------------------------------------------------------
# Curve length computation
# ------------------------------------------------------------------------------

@doc raw"""
    curve_length(
        riemannian_metric::AbstractArray,
        curve_velocity::AbstractArray
    )

Function to compute the (discretized) integral defining the length of a curve γ̲
on a Riemmanina manifold. The length is defined as

    L(γ̲) = ∫ dt √(⟨γ̲̇(t), G̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and G̲̲ is the
Riemmanian metric tensor. For this function, we approximate the integral as

    L(γ̲) ≈ ∑ᵢ Δt √(⟨γ̲̇(tᵢ)ᵀ G̲̲ (γ̲(tᵢ+1) γ̲̇(tᵢ))⟩),

where Δt is the time step between points.

# Arguments
- `riemannian_metric::AbstractArray`: `d×d×N` tensor where `d` is the dimension
  of the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each slice of the array represents the Riemmanian
  metric tensor for the curve at the corresponding time point.
- `curve_velocity::AbstractArray`: `d×N` Matrix where `d` is the dimension of
  the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each column represents the velocity of the curve at
  the corresponding time point.

# Returns
- `Length::Number`: Approximation of the Length for the path on the manifold.
"""
function curve_length(
    riemannian_metric::AbstractArray,
    curve_velocity::AbstractArray
)
    # Compute γ̲̇ᵀ G̲̲ γ̲̇ in a broadcasted manner
    γ̲̇ᵀ_G_γ̲̇ = sum(
        curve_velocity .* Flux.batched_vec(riemannian_metric * curve_velocity),
        dims=1
    )

    # Return curve length
    return sum(sqrt.(γ̲̇ᵀ_G_γ̲̇) .* Δt)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    curve_length_loop(
        riemannian_metric::AbstractArray,
        curve_velocity::AbstractArray
    )

Function to compute the (discretized) integral defining the length of a curve γ̲
on a Riemmanina manifold. The length is defined as

    L(γ̲) = ∫ dt √(⟨γ̲̇(t), G̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and G̲̲ is the
Riemmanian metric tensor. For this function, we approximate the integral as

    L(γ̲) ≈ ∑ᵢ Δt √(⟨γ̲̇(tᵢ)ᵀ G̲̲ (γ̲(tᵢ+1) γ̲̇(tᵢ))⟩),

where Δt is the time step between points.

This method performs the product ⟨γ̲̇(tᵢ)ᵀ G̲̲ (γ̲(tᵢ+1) γ̲̇(tᵢ))⟩ using a list
comprehension. This is done to avoid the use of the LinearAlgebra.dot function,
which is not compatible with the composition of `Zygote.jl` over `TaylorDiff.jl`

# Arguments
- `riemannian_metric::AbstractArray`: `d×d×N` tensor where `d` is the dimension
  of the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each slice of the array represents the Riemmanian
  metric tensor for the curve at the corresponding time point.
- `curve_velocity::AbstractArray`: `d×N` Matrix where `d` is the dimension of
  the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each column represents the velocity of the curve at
  the corresponding time point.

# Returns
- `Length::Number`: Approximation of the Length for the path on the manifold.
"""
function curve_length_loop(
    riemannian_metric::AbstractMatrix,
    curve_velocity::AbstractVector
)
    # Compute γ̲̇ᵀ G̲̲ γ̲̇ in a loop
    return sum(
        begin
            curve_velocity[i] * riemannian_metric[i, j] * curve_velocity[j]
        end
        for i in eachindex(curve_velocity)
        for j in eachindex(curve_velocity)
    )
end # function

# ------------------------------------------------------------------------------