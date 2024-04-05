
# Import library for automatic differentiation
import Zygote
import TaylorDiff

# Import ML library
import Flux
import NNlib

# Import library to use Ellipsis Notation
using EllipsisNotation

# Import abstract types from RHVAEs
using ...RHVAEs: RHVAE

# Import functions from RHVAEs module
using ...RHVAEs: metric_tensor

# Import functions from utils module
using ...utils: vec_mat_vec_batched

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
    scale = @. (g.z_init - g.z_end) / (ẑ_init - ẑ_end)
    shift = @. ((g.z_init * ẑ_end) - (g.z_end * ẑ_init)) / (ẑ_init - ẑ_end)

    # Return shifted and scaled output
    return @. (scale * g_t) - shift
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
    scale = @. (g.z_init - g.z_end) / (ẑ_init - ẑ_end)
    shift = @. ((g.z_init * ẑ_end) - (g.z_end .* ẑ_init)) / (ẑ_init - ẑ_end)

    # Return shifted and scaled output
    return @. (scale * g_t) - shift
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

    # Check fdtype
    if fdtype == :forward
        # Define step size
        ε = √(eps(eltype(t)))
        # Compute finite difference derivatives
        dγdt = (curve(t .+ ε) .- curve(t)) ./ ε
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
        curve_velocity::AbstractArray,
        t::AbstractVector;
    )

Function to compute the (discretized) integral defining the length of a curve γ̲
on a Riemmanina manifold. The length is defined as

    L(γ̲) = ∫ dt √(⟨γ̲̇(t), G̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and G̲̲ is the
Riemmanian metric tensor. For this function, we approximate the integral as

    L(γ̲) ≈ ∑ᵢ Δt √(⟨γ̲̇(tᵢ)ᵀ G̲̲ (γ̲(tᵢ+1) γ̲̇(tᵢ))⟩),

where Δt is the time step between points. Note that this Δt is assumed to be
constant, thus, the time points `t` must be equally spaced.

# Arguments
- `riemannian_metric::AbstractArray`: `d×d×N` tensor where `d` is the dimension
  of the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each slice of the array represents the Riemmanian
  metric tensor for the curve at the corresponding time point.
- `curve_velocity::AbstractArray`: `d×N` Matrix where `d` is the dimension of
  the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each column represents the velocity of the curve at
  the corresponding time point.
- `t::AbstractVector`: Vector of time points at which the curve is sampled.

# Returns
- `Length::Number`: Approximation of the Length for the path on the manifold.
"""
function curve_length(
    riemannian_metric::AbstractArray,
    curve_velocity::AbstractArray,
    t::AbstractVector;
)
    # Compute Δt
    Δt = t[2] - t[1]

    # Compute γ̲̇ᵀ G̲̲ γ̲̇ 
    γ̲̇ᵀ_G_γ̲̇ = vec_mat_vec_batched(
        curve_velocity, riemannian_metric, curve_velocity
    )

    return sum(sqrt.(γ̲̇ᵀ_G_γ̲̇) .* Δt)
end # function

# ------------------------------------------------------------------------------
# Curve energy computation
# ------------------------------------------------------------------------------

@doc raw"""
    curve_energy(
        riemannian_metric::AbstractArray,
        curve_velocity::AbstractArray,
        t::AbstractVector;
    )

Function to compute the (discretized) integral defining the energy of a curve γ̲
on a Riemmanina manifold. The energy is defined as

        E(γ̲) = ∫ dt ⟨γ̲̇(t), G̲̲ γ̲̇(t)⟩,

where γ̲̇(t) defines the velocity of the parametric curve, and G̲̲ is the
Riemmanian metric tensor. For this function, we approximate the integral as

        E(γ̲) ≈ ∑ᵢ Δt ⟨γ̲̇(tᵢ)ᵀ G̲̲ (γ̲(tᵢ+1) γ̲̇(tᵢ))⟩,

where Δt is the time step between points. Note that this Δt is assumed to be
constant, thus, the time points `t` must be equally spaced.

# Arguments
- `riemannian_metric::AbstractArray`: `d×d×N` tensor where `d` is the dimension
  of the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each slice of the array represents the Riemmanian
  metric tensor for the curve at the corresponding time point.
- `curve_velocity::AbstractArray`: `d×N` Matrix where `d` is the dimension of
  the manifold on which the curve lies and `N` is the number of sampled time
  points along the curve. Each column represents the velocity of the curve at
  the corresponding time point.
- `t::AbstractVector`: Vector of time points at which the curve is sampled.

# Returns
- `Energy::Number`: Approximation of the Energy for the path on the manifold.
"""
function curve_energy(
    riemannian_metric::AbstractArray,
    curve_velocity::AbstractArray,
    t::AbstractVector;
)
    # Compute Δt
    Δt = t[2] - t[1]

    # Compute γ̲̇ᵀ G̲̲ γ̲̇ 
    γ̲̇ᵀ_G_γ̲̇ = vec_mat_vec_batched(
        curve_velocity, riemannian_metric, curve_velocity
    )

    return sum(γ̲̇ᵀ_G_γ̲̇ .* Δt) / 2
end # function

# ==============================================================================
# Neural Geodesic Training 
# ==============================================================================

@doc raw"""
    loss(
        curve::NeuralGeodesic,
        rhvae::RHVAE,
        t::AbstractVector;
        curve_velocity::Function=curve_velocity_TaylorDiff,
        curve_integral::Function=curve_length,
    )

Function to compute the loss for a given curve on a Riemmanian manifold. The
loss is defined as the integral over the curve, computed using the provided
`curve_integral` function (either length or energy).

# Arguments
- `curve::NeuralGeodesic`: The curve on the Riemmanian manifold.
- `rhvae::RHVAE`: The Riemmanian Hamiltonian Variational AutoEncoder used to
  compute the Riemmanian metric tensor.
- `t::AbstractVector`: Vector of time points at which the curve is sampled.

## Optional Keyword Arguments
- `curve_velocity::Function=curve_velocity_TaylorDiff`: Function to compute the
  velocity of the curve. Default is `curve_velocity_TaylorDiff`. Also accepts
  `curve_velocity_finitediff`.
- `curve_integral::Function=curve_length`: Function to compute the integral over
  the curve. Default is `curve_length`. Also accepts `curve_energy`.

# Returns
- `Loss::Number`: The computed loss for the given curve.

# Notes
This function first computes the geodesic curve using the provided `curve`
function. It then computes the Riemmanian metric tensor using the
`metric_tensor` function from the `RHVAE` module with the computed curve and the
provided `rhvae`. The velocity of the curve is then computed using the provided
`curve_velocity` function. Finally, the integral over the curve is computed
using the provided `curve_integral` function and returned as the loss.
"""
function loss(
    curve::NeuralGeodesic,
    rhvae::RHVAE,
    t::AbstractVector;
    curve_velocity::Function=curve_velocity_finitediff,
    curve_integral::Function=curve_length,
)
    # Compute the geodesic curve
    z_mat = curve(t)

    # Compute the Riemmanian metric tensor
    G = metric_tensor(z_mat, rhvae)

    # Compute the curve velocity
    γ̇ = curve_velocity(curve, t)

    # Compute and return the integral over the curve
    return curve_integral(G, γ̇, t)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    train!(
        curve::NeuralGeodesic,
        rhvae::RHVAE,
        t::AbstractVector,
        opt::NamedTuple;
        loss::Function=loss,
        loss_kwargs::Dict=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Function to train a NeuralGeodesic model using a Riemmanian Hamiltonian
Variational AutoEncoder (RHVAE). The training process involves computing the
gradient of the loss function and updating the model parameters accordingly.

# Arguments
- `curve::NeuralGeodesic`: The curve on the Riemmanian manifold.
- `rhvae::RHVAE`: The Riemmanian Hamiltonian Variational AutoEncoder used to
  compute the Riemmanian metric tensor.
- `t::AbstractVector`: Vector of time points at which the curve is sampled.
  These must be equally spaced.
- `opt::NamedTuple`: The optimization parameters.

## Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function to be minimized during
  training. Default is `loss`.
- `loss_kwargs::Dict=Dict()`: Additional keyword arguments to be passed to the
  loss function.
- `verbose::Bool=false`: If `true`, the loss value is printed at each iteration.
- `loss_return::Bool=false`: If `true`, the function returns the loss value.

# Returns
- `Loss::Number`: The computed loss for the given curve. This is only returned
  if `loss_return` is `true`.

# Notes
This function first computes the gradient of the loss function with respect to
the model parameters. It then updates the model parameters using the computed
gradient and the provided optimization parameters. If `verbose` is `true`, the
loss value is printed at each iteration. If `loss_return` is `true`, the
function returns the loss value.
"""
function train!(
    curve::NeuralGeodesic,
    rhvae::RHVAE,
    t::AbstractVector,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Dict=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(curve) do curve_nn
        loss_function(curve_nn, rhvae, t; loss_kwargs...)
    end # do block

    # Update parameters
    Flux.Optimisers.update!(opt, curve, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function