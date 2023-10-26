# Import basic math
import LinearAlgebra

# Import library to perform Einstein summation
using TensorOperations: @tensor

# Import library for automatic differentiation
import Zygote

# Import ML library
import Flux

# Import Abstract Types

using ..AutoEncode: JointEncoder, SimpleDecoder, JointDecoder, SplitDecoder

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Differential Geometry on Riemmanian Manifolds
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    riemannian_metric(manifold, val)

Compute the Riemannian metric `M = JᵀJ` of a manifold using numerical
differentiation with `Zygote.jl`. The metric is computed at a specific point on
the manifold.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the point on the manifold
  where the metric should be evaluated.

# Returns
- `M::Matrix{<:AbstractFloat}`: The Riemannian metric matrix evaluated at `val`.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
M = riemannian_metric(manifold, val)
```
"""
function riemannian_metric(
    manifold::Function, val::Vector{T}
)::Matrix{T} where {T<:AbstractFloat}
    # Compute Jacobian
    jac = first(Zygote.jacobian(manifold, val))
    # Compute the metric
    return jac' * jac
end # function

@doc raw"""
    riemannian_metric(manifold, val)

Compute the Riemannian metric `M = JᵀJ` of a manifold defined by a neural
network using numerical differentiation with `Zygote.jl`. The metric is
evaluated based on the outputs of the network with respect to its inputs.

# Arguments
- `manifold::Flux.Chain`: A neural network model (as a chain) defining the
  manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the manifold
  (neural network) where the metric should be evaluated.

# Returns
- `M::Matrix{<:AbstractFloat}`: The Riemannian metric matrix evaluated at `val`.

# Example
```julia-repl
model = Chain(Dense(2, 3, relu), Dense(3, 2))
val = [1.0, 2.0]
M = riemannian_metric(model, val)
```
"""
function riemannian_metric(
    manifold::Flux.Chain, val::Vector{T}
)::Matrix{T} where {T<:AbstractFloat}
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(manifold, val))
    # Compute the metric
    return jac' * jac
end # function

@doc raw"""
    riemannian_metric(manifold, val)

Compute the Riemannian metric `M = JᵀJ` of a manifold defined by the decoder
structure of a variational autoencoder (VAE) using numerical differentiation
with `Zygote.jl`. The metric is evaluated based on the outputs of the decoder
with respect to its inputs.

# Arguments
- `manifold::SimpleDecoder`: A VAE decoder structure containing a neural network
  model that defines the manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the decoder
  (neural network) where the metric should be evaluated.

# Returns
- `M::Matrix{<:AbstractFloat}`: The Riemannian metric matrix evaluated at `val`.

# Example
```julia-repl
decoder_model = Chain(Dense(2, 3, relu), Dense(3, 2))
decoder = SimpleDecoder(decoder_model)
val = [1.0, 2.0]
M = riemannian_metric(decoder, val)
```
"""
function riemannian_metric(
    manifold::SimpleDecoder, val::Vector{T}
)::Matrix{T} where {T<:AbstractFloat}
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(manifold.decoder, val))
    # Compute the metric
    return jac' * jac
end # function

@doc raw"""
    riemannian_metric(manifold::JointDecoder, val::Vector{T}) where {T<:AbstractFloat}

Compute the Riemannian metric `M̲̲` for a stochastic manifold defined by a
`JointDecoder` using numerical differentiation with `Zygote.jl`. The metric is
evaluated based on the outputs of the neural network with respect to its inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ Where J̲̲_µ and
J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect to the input.
Given that we compute the Jacobian of `logσ` directly, the Jacobian of `σ` is
obtained using the chain rule.

# Arguments
- `manifold::JointDecoder`: A VAE decoder structure that has separate paths for
  determining both the mean and log standard deviation of the latent space.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the manifold
  (neural network) where the metric should be evaluated.

# Returns
- `M̲̲::Matrix{<:AbstractFloat}`: The Riemannian metric matrix evaluated at
  `val`.

# Example
```julia-repl
model = JointDecoder(
    Chain(Dense(2, 3, relu), Dense(3, 2)),
    Dense(2, 1),
    Dense(2, 1)
)
val = [1.0, 2.0]
M = riemannian_metric(model, val)
```
"""
function riemannian_metric(
    manifold::JointDecoder, val::Vector{T}
)::Matrix{T} where {T<:AbstractFloat}
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(
        Zygote.jacobian(Flux.Chain(manifold.decoder..., manifold.µ), val)
    )

    # Compute Jacobian with respect to the input for the log standard deviation
    jac_logσ = first(
        Zygote.jacobian(Flux.Chain(manifold.decoder..., manifold.logσ), val)
    )

    # Convert jac_logσ to jac_σ using the chain rule:
    # 1. Compute σ by exponentiating logσ
    σ_val = exp.(Flux.Chain(manifold.decoder..., manifold.logσ)(val))
    # 2. Use the chain rule
    jac_σ = σ_val .* jac_logσ

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

@doc raw"""
    riemannian_metric(manifold::SplitDecoder, val::Vector{T}) where {T<:AbstractFloat}

Compute the Riemannian metric `M̲̲` for a stochastic manifold defined by a
`SplitDecoder` using numerical differentiation with `Zygote.jl`. The metric is
evaluated based on the outputs of the individual neural networks with respect to
their inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ Where J̲̲_µ and
J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect to the input.
Given that we compute the Jacobian of `logσ` directly, the Jacobian of `σ` is
obtained using the chain rule.

# Arguments
- `manifold::SplitDecoder`: A VAE decoder structure that has separate neural
  networks for determining both the mean and log standard deviation of the
  latent space.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the manifold
  (neural networks) where the metric should be evaluated.

# Returns
- `M̲̲::Matrix{<:AbstractFloat}`: The Riemannian metric matrix evaluated at
  `val`.

# Example
```julia-repl
model = SplitDecoder(
    Chain(Dense(2, 3, relu), Dense(3, 1)),
    Chain(Dense(2, 3, relu), Dense(3, 1))
)
val = [1.0, 2.0]
M = riemannian_metric(model, val)
```
"""
function riemannian_metric(
    manifold::SplitDecoder, val::Vector{T}
)::Matrix{T} where {T<:AbstractFloat}
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(Zygote.jacobian(manifold.µ, val))

    # Compute Jacobian with respect to the input for the log standard deviation
    jac_logσ = first(Zygote.jacobian(manifold.logσ, val))

    # Convert jac_logσ to jac_σ using the chain rule
    σ_val = exp.(manifold.logσ(val))
    jac_σ = σ_val .* jac_logσ

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ==============================================================================

@doc raw"""
    ∂M̲̲∂γ̲(manifold, val, out_dim)

Compute the derivative of the Riemannian metric `M̲̲` with respect to the
coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the point on the manifold
  where the derivative should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `∂M̲̲::Array{<:AbstractFloat}`: A rank-3 tensor evaluating the derivative of
  the Riemannian manifold metric.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
∂M = ∂M̲̲∂γ̲(manifold, val, 2)
```
"""
function ∂M̲̲∂γ̲(
    manifold::Function, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Compute the manifold Jacobian. This should be a D×d matrix where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we use first() to extract the object we care about from the
    # Zygote output.
    J̲̲ = first(Zygote.jacobian(manifold, val))

    # Compute Hessian tensor, i.e., the tensor with the manifold second
    # derivatives. This should be a D×d×d third-order tensor, where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we have to manually evaluate the hessian on each dimension of
    # the output.
    H̲̲ = permutedims(
        cat(
            [Zygote.hessian(v -> manifold(v)[D], val) for D = 1:out_dim]...,
            dims=3
        ),
        (3, 1, 2)
    )

    # Compute the derivative of the Riemmanian metric. This should be a d×d×d
    # third-order tensor.
    return @tensor ∂M̲̲[i, j, k] := H̲̲[l, i, k] * J̲̲[l, j] +
                                    H̲̲[l, k, j] * J̲̲[l, i]
end # function

@doc raw"""
    ∂M̲̲∂γ̲(manifold, val, out_dim)

Compute the derivative of the Riemannian metric `M̲̲` with respect to the
coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Flux.Chain`: A neural network model (as a chain) defining the
  manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the manifold
  (neural network) where the derivative should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `∂M̲̲::Array{<:AbstractFloat}`: A rank-3 tensor evaluating the derivative of
  the Riemannian manifold metric.

# Example
```julia-repl
model = Chain(Dense(2, 3, relu), Dense(3, 2))
val = [1.0, 2.0]
∂M = ∂M̲̲∂γ̲(model, val, 2)
```
"""
function ∂M̲̲∂γ̲(
    manifold::Flux.Chain, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Compute the Jacobian of the manifold with respect to its input
    J̲̲ = first(Zygote.jacobian(manifold, val))
    # Compute Hessian tensor for each dimension of the output, then permute
    # dimensions to get the desired third-order tensor structure
    H̲̲ = permutedims(
        cat(
            [Zygote.hessian(v -> manifold(v)[D], val) for D = 1:out_dim]...,
            dims=3
        ),
        (3, 1, 2)
    )

    # Compute the derivative of the Riemannian metric using the tensor product of 
    # the Hessian tensor and the Jacobian matrix
    return @tensor ∂M̲̲[i, j, k] := H̲̲[l, i, k] * J̲̲[l, j] +
                                    H̲̲[l, k, j] * J̲̲[l, i]
end # function

# ==============================================================================

@doc raw"""
    christoffel_symbols(manifold, val, out_dim)

Compute the Christoffel symbols of the first kind, which are derived from the
Riemannian metric `M̲̲` of a manifold. The Christoffel symbols represent the
connection on the manifold and are used in the geodesic equation to determine
the shortest paths between points.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the point on the manifold
  where the Christoffel symbols should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `Γᵏᵢⱼ::Array{<:AbstractFloat}`: A rank-3 tensor (dimensions: k, i, j)
  representing the Christoffel symbols for the given Riemannian manifold metric.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
Γ = christoffel_symbols(manifold, val, 2)
```
"""
function christoffel_symbols(
    manifold::Function, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Evaluate metric inverse
    M̲̲⁻¹ = LinearAlgebra.inv(riemmanian_metric(manifold, val))

    # Evaluate metric derivative
    ∂M̲̲ = ∂M̲̲∂γ̲(manifold, val, out_dim)

    # Compute Christoffel Symbols
    return @tensor Γᵏᵢⱼ[i, j, k] := (1 / 2) * M̲̲⁻¹[k, h] *
                                    (∂M̲̲[i, h, j] + ∂M̲̲[j, h, i] - ∂M̲̲[i, j, h])
end # function

@doc raw"""
    christoffel_symbols(manifold, val, out_dim)

Compute the Christoffel symbols of the first kind, derived from the Riemannian
metric `M̲̲` of a manifold represented by a neural network (as a `Flux.Chain`).
The Christoffel symbols represent the connection on the manifold and are used in
the geodesic equation to determine the shortest paths between points.

# Arguments
- `manifold::Flux.Chain`: A neural network model (as a chain) defining the
  manifold.
- `val::Vector{<:AbstractFloat}`: A vector specifying the input to the manifold
  (neural network) where the Christoffel symbols should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `Γᵏᵢⱼ::Array{<:AbstractFloat}`: A rank-3 tensor (dimensions: k, i, j)
  representing the Christoffel symbols for the given Riemannian manifold metric.

# Example
```julia-repl
model = Chain(Dense(2, 3, relu), Dense(3, 2))
val = [1.0, 2.0]
Γ = christoffel_symbols(model, val, 2)
```
"""
function christoffel_symbols(
    manifold::Flux.Chain, val::Vector{T}, out_dim::Int
)::Array{T} where {T<:AbstractFloat}
    # Compute the inverse of the Riemannian metric
    M̲̲⁻¹ = LinearAlgebra.inv(riemannian_metric(manifold, val))

    # Compute the derivative of the Riemannian metric
    ∂M̲̲ = ∂M̲̲∂γ̲(manifold, val, out_dim)

    # Use the metric inverse and its derivative to compute the Christoffel
    # symbols
    return @tensor Γᵏᵢⱼ[i, j, k] := (1 / 2) * M̲̲⁻¹[k, h] *
                                    (∂M̲̲[i, h, j] + ∂M̲̲[j, h, i] - ∂M̲̲[i, j, h])
end # function

# ==============================================================================

@doc raw"""
    geodesic_system!(du, u, param, t)

Define the right-hand side of the geodesic system of ODEs. This function
evaluates in-place by making `du` the first input, which is intended to
accelerate the integration process. To represent the 2nd order ODE system, we
use a system of coupled 1st order ODEs.

# Arguments
- `du::Array{<:AbstractFloat}`: Derivatives of state variables. The first `end ÷
  2` entries represent the velocity in the latent space (i.e., dγ/dt), while the
  latter half represents the curve's acceleration (i.e., d²γ/dt²).
- `u::Array{<:AbstractFloat}`: State variables. The first `end ÷ 2` entries
  specify the coordinates in the latent space (i.e., γ), and the second half
  represents the curve's velocity (i.e., dγ/dt).
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `out_dim::Int`: Dimensionality of the output space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that 
    defines the manifold.
  - `γ_init::Vector{<:AbstractFloat}`: Initial position in the latent space.
  - `γ_end::Vector{<:AbstractFloat}`: Final position in the latent space.
- `t::AbstractFloat`: Time at which to evaluate the right-hand side.
"""
function geodesic_system!(du, u, param, t)
    # Extract dimensions d and D
    in_dim, out_dim = param[:in_dim], param[:out_dim]
    # Extract manifold function
    manifold = param[:manifold]

    # Set curve coordinates 
    γ = u[1:in_dim]
    # Set curve velocities
    dγ = u[in_dim+1:end]

    # Compute Christoffel symbols
    Γᵏᵢⱼ = christoffel_symbols(manifold, γ, out_dim)

    # Define the geodesic system of 2nd order ODEs
    @tensor d²γ[k] := -Γᵏᵢⱼ[i, j, k] * dγ[i] * dγ[j]

    # Update derivative values
    du .= [dγ; d²γ]
end # function

# ==============================================================================

@doc raw"""
    bc_collocation!(residual, u, param, t)

Evaluates the residuals between the current position in latent space and the
desired boundary conditions. This function is crucial when implementing
collocation methods for boundary value problems in differential equations.

# Arguments
- `residual::Vector{<:AbstractFloat}`: A vector to store the residuals between
  the desired boundary conditions and the current state.
- `u::Array{Array{<:AbstractFloat}}`: A sequence of state vectors at each time
  step, where each state vector represents a position in latent space.
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that 
    defines the manifold.
  - `γ_init::Vector{<:AbstractFloat}`: Initial position in latent space.
  - `γ_end::Vector{<:AbstractFloat}`: Final position in latent space.
- `t::AbstractFloat`: Time at which to evaluate the residuals. Note: this
  argument is present for compatibility with solvers but isn't used directly in
  this function.

# Returns
- The function modifies the `residual` vector in-place, updating it with the
  calculated residuals between the desired boundary conditions and the current
  state at each boundary.
"""
function bc_collocation!(residual, u, param, t)
    # Extract parameters
    in_dim = param[:in_dim]

    # Compute residual for initial position
    @. residual[1:in_dim] = u[1][1:in_dim] - param[:γ_init]

    # Compute residual for final position
    @. residual[in_dim+1:end] = u[end][1:in_dim] - param[:γ_end]
end # function

@doc raw"""
    bc_shooting!(residual, u, param, t)

Evaluates the residuals between the current position in latent space and the
desired boundary conditions for solving boundary value problems using the
shooting method.

# Arguments
- `residual::Vector{<:AbstractFloat}`: A vector to store the residuals between
  the desired boundary conditions and the current state.
- `u::Function`: A function that, when evaluated at a specific time `t`, returns
  the state vector representing a position in latent space.
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that 
    defines the manifold.
  - `γ_init::Vector{<:AbstractFloat}`: Initial position in latent space.
  - `γ_end::Vector{<:AbstractFloat}`: Final position in latent space.
- `t::AbstractFloat`: Time at which to evaluate the residuals. Note: this
  argument is present for compatibility with solvers but isn't used directly in
  this function.

# Returns
- The function modifies the `residual` vector in-place, updating it with the
  calculated residuals between the desired boundary conditions and the current
  state at each boundary.
"""
function bc_shooting!(residual, u, param, t)
    # Extract parameters
    in_dim = param[:in_dim]

    # Compute residual for initial position
    @. residual[1:in_dim] = u(0.0)[1:in_dim] - param[:γ_init]
    # Compute residual for final position
    @. residual[in_dim+1:end] = u(1.0)[1:in_dim] - param[:γ_end]
end # function

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Approximating geodesics via Splines
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    min_energy_spline(γ_init, γ_end, manifold, n_points; step_size, stop, tol)

Function that produces an energy-minimizing spline between two points `γ_init`
and `γ_end` on a Riemmanian manifold using the algorithm by Hoffer & Pottmann.

# Arguments
- `γ_init::AbstractVector{T}`: Initial position of curve `γ` on the parameter
  space.
- `γ_end::AbstractVector{T}`: Final position of curve `γ` on the parameter
  space.
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.
- `n_points::Int`: Number of points to use for the interpolation.

## Optional arguments
- `step_size::AbstractFloat=1E-3`: Step size used to update curve parameters.
- `stop::Boolean=true`: Boolean indicating if iterations should stop if the
  current point is within certain tolerance value `tol` from the desired final
  point.
- `tol::AbstractFloat=1E-6`: Tolerated difference between desired and current
  final point for an early stop criteria.

# Returns
- `γ::Matrix{T}`: `d × N` matrix where each row represents each of the
  dimensions on the manifold and `N` is the number of points needed to
  interpolate between points.

where `T <: AbstractFloat`
"""
function min_energy_spline(
    γ_init::AbstractVector{T},
    γ_end::AbstractVector{T},
    manifold::Function,
    n_points::Int;
    step_size::AbstractFloat=1E-3,
    stop::Bool=true,
    tol::AbstractFloat=1E-3
) where {T<:AbstractFloat}
    # Initialize matrix where to store interpolation points
    γ = Matrix{T}(undef, length(γ_init), n_points)
    # Set initial value
    γ[:, 1] .= γ_init

    # Map final position to output space
    x_end = manifold(γ_end)

    # Initialize loop counter
    count = 1
    # Loop through points
    for i = 2:n_points
        # 1. Map current point to output space
        x_current = manifold(γ[:, i-1])
        # 2. Compute Jacobian. Each column represents one basis vector c̲ for
        #    the tangent space at x_current.
        J̲̲ = first(Zygote.jacobian(manifold, γ[:, i-1]))
        # 3. Compute the Riemmanian metric tensor at current point
        M̲̲ = J̲̲' * J̲̲
        # 4. Compute the difference between the current position and the desired
        #    final point
        Δx̲ = x_end .- x_current
        # 5. Compute r̲ vector where each entry is the inner product beetween
        #    Δx̲ and each of the basis vectors of the tangent space c̲. Since
        #    these vectors are stored in the Jacobian, we can simply do matrix
        #    vector multiplication.
        r̲ = J̲̲' * Δx̲
        # 6. Solve linear system to find tangential vector
        t̲ = M̲̲ \ r̲
        # 7. Update parameters given chosen step size
        γ[:, i] = γ[:, i-1] .+ (step_size .* t̲)

        # Update loop counter
        count += 1

        # Check if final point satisfies tolerance
        if (stop) .& (LinearAlgebra.norm(γ[:, i] - γ_end) ≤ tol)
            break
        end # if
    end # for

    # Return curve
    if stop
        return γ[:, 1:count]
    else
        return γ
    end # if
end # function


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Discretized curv characteristics
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    curve_energy(γ̲, manifold)

Function to compute the (discretized) integral defining the energy of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = 1 / 2 ∫ dt ⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩,

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Energy::T`: Value of the energy for the path on the manifold.
"""
function curve_energy(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return (1 / (2 * Δt)) * sum(
        [
        LinearAlgebra.dot(
            Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
        )
        for i = 1:size(Δγ, 2)
    ]
    )
end # function

@doc raw"""
    curve_length(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Length::T`: Value of the Length for the path on the manifold.
"""
function curve_length(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return sum(
        [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
    )
end # function

@doc raw"""
    curve_lengths(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Lengths::Vector{T}`: Value of the lengths for each individual path on the
  manifold.
"""
function curve_lengths(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
end # function