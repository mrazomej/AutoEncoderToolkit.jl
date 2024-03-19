# ==============================================================================
# Geodesic Equation computation via differential equations
# ==============================================================================

# ------------------------------------------------------------------------------
# Derivative of the Riemannian metric with respect to the coordinates in the
# input space
# ------------------------------------------------------------------------------

@doc raw"""
    ∂M̲̲∂γ̲(manifold, val, out_dim)

Compute the derivative of the Riemannian metric `M̲̲` with respect to the
coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `z::AbstractVector`: A vector specifying the point on the manifold where the
  derivative should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `∂M̲̲::Array{<:Number}`: A rank-3 tensor evaluating the derivative of
  the Riemannian manifold metric.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
∂M = ∂M̲̲∂γ̲(manifold, val, 2)
```
"""
function ∂M̲̲∂γ̲(
    manifold::Function, z::AbstractVector, out_dim::Int
)
    # Compute the manifold Jacobian. This should be a D×d matrix where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we use first() to extract the object we care about from the
    # Zygote output.
    J̲̲ = first(Zygote.jacobian(manifold, z))

    # Compute Hessian tensor, i.e., the tensor with the manifold second
    # derivatives. This should be a D×d×d third-order tensor, where D is the
    # dimension of the output space, and d is the dimension of the manifold.
    # Note that we have to manually evaluate the hessian on each dimension of
    # the output.
    H̲̲ = permutedims(
        cat(
            [Zygote.hessian(v -> manifold(v)[D], z) for D = 1:out_dim]...,
            dims=3
        ),
        (3, 1, 2)
    )

    # Compute the derivative of the Riemmanian metric. This should be a d×d×d
    # third-order tensor.
    return @tensor ∂M̲̲[i, j, k] := H̲̲[l, i, k] * J̲̲[l, j] +
                                    H̲̲[l, k, j] * J̲̲[l, i]
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∂M̲̲∂γ̲(manifold, z, out_dim)

Compute the derivative of the Riemannian metric `M̲̲` with respect to the
coordinates in the input space using `Zygote.jl`.

# Arguments
- `manifold::Flux.Chain`: A neural network model (as a chain) defining the
  manifold.
- `z::AbstractVector`: A vector specifying the input to the manifold (neural
  network) where the derivative should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `∂M̲̲::Array{<:Number}`: A rank-3 tensor evaluating the derivative of
  the Riemannian manifold metric.

# Example
```julia-repl
model = Chain(Dense(2, 3, relu), Dense(3, 2))
val = [1.0, 2.0]
∂M = ∂M̲̲∂γ̲(model, val, 2)
```
"""
function ∂M̲̲∂γ̲(
    manifold::Flux.Chain, z::AbstractVector, out_dim::Int
)
    # Compute the Jacobian of the manifold with respect to its input
    J̲̲ = first(Zygote.jacobian(manifold, z))
    # Compute Hessian tensor for each dimension of the output, then permute
    # dimensions to get the desired third-order tensor structure
    H̲̲ = permutedims(
        cat(
            [Zygote.hessian(v -> manifold(v)[D], z) for D = 1:out_dim]...,
            dims=3
        ),
        (3, 1, 2)
    )

    # Compute the derivative of the Riemannian metric using the tensor product
    # of the Hessian tensor and the Jacobian matrix
    return @tensor ∂M̲̲[i, j, k] := H̲̲[l, i, k] * J̲̲[l, j] +
                                    H̲̲[l, k, j] * J̲̲[l, i]
end # function

# ------------------------------------------------------------------------------
# Christoffel symbols computation
# ------------------------------------------------------------------------------

@doc raw"""
    christoffel_symbols(manifold, z, out_dim)

Compute the Christoffel symbols of the first kind, which are derived from the
Riemannian metric `M̲̲` of a manifold. The Christoffel symbols represent the
connection on the manifold and are used in the geodesic equation to determine
the shortest paths between points.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `z::AbstractVector`: A vector specifying the point on the manifold where the
  Christoffel symbols should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `Γᵏᵢⱼ::AbstractArray`: A rank-3 tensor (dimensions: k, i, j) representing the
  Christoffel symbols for the given Riemannian manifold metric.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
Γ = christoffel_symbols(manifold, val, 2)
```
"""
function christoffel_symbols(
    manifold::Function, z::AbstractVector, out_dim::Int
)
    # Evaluate metric inverse
    M̲̲⁻¹ = LinearAlgebra.inv(riemmanian_metric(manifold, z))

    # Evaluate metric derivative
    ∂M̲̲ = ∂M̲̲∂γ̲(manifold, z, out_dim)

    # Compute Christoffel Symbols
    return @tensor Γᵏᵢⱼ[i, j, k] := (1 / 2) * M̲̲⁻¹[k, h] *
                                    (∂M̲̲[i, h, j] + ∂M̲̲[j, h, i] - ∂M̲̲[i, j, h])
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    christoffel_symbols(manifold, z, out_dim)

Compute the Christoffel symbols of the first kind, derived from the Riemannian
metric `M̲̲` of a manifold represented by a neural network (as a `Flux.Chain`).
The Christoffel symbols represent the connection on the manifold and are used in
the geodesic equation to determine the shortest paths between points.

# Arguments
- `manifold::Flux.Chain`: A neural network model (as a chain) defining the
  manifold.
- `z::AbstractVector`: A vector specifying the input to the manifold (neural
  network) where the Christoffel symbols should be evaluated.
- `out_dim::Int`: Dimensionality of the output space. This is essential because
  `Zygote.jl` can't compute certain derivatives automatically. We must iterate
  over each dimension of the output space manually.

# Returns
- `Γᵏᵢⱼ::AbstractArray`: A rank-3 tensor (dimensions: k, i, j) representing the
  Christoffel symbols for the given Riemannian manifold metric.

# Example
```julia-repl
model = Chain(Dense(2, 3, relu), Dense(3, 2))
val = [1.0, 2.0]
Γ = christoffel_symbols(model, val, 2)
```
"""
function christoffel_symbols(
    manifold::Flux.Chain, z::AbstractVector, out_dim::Int
)
    # Compute the inverse of the Riemannian metric
    M̲̲⁻¹ = LinearAlgebra.inv(pullback_metric(manifold, z))

    # Compute the derivative of the Riemannian metric
    ∂M̲̲ = ∂M̲̲∂γ̲(manifold, z, out_dim)

    # Use the metric inverse and its derivative to compute the Christoffel
    # symbols
    return @tensor Γᵏᵢⱼ[i, j, k] := (1 / 2) * M̲̲⁻¹[k, h] *
                                    (∂M̲̲[i, h, j] + ∂M̲̲[j, h, i] - ∂M̲̲[i, j, h])
end # function

# ------------------------------------------------------------------------------
# Geoedesic system of ODEs
# ------------------------------------------------------------------------------

@doc raw"""
    geodesic_system!(du, u, param, t)

Define the right-hand side of the geodesic system of ODEs. This function
evaluates in-place by making `du` the first input, which is intended to
accelerate the integration process. To represent the 2nd order ODE system, we
use a system of coupled 1st order ODEs.

# Arguments
- `du::AbstractArray`: Derivatives of state variables. The first `end ÷ 2`
  entries represent the velocity in the latent space (i.e., dγ/dt), while the
  latter half represents the curve's acceleration (i.e., d²γ/dt²).
- `u::AbstractArray`: State variables. The first `end ÷ 2` entries specify the
  coordinates in the latent space (i.e., γ), and the second half represents the
  curve's velocity (i.e., dγ/dt).
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `out_dim::Int`: Dimensionality of the output space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that
    defines the manifold.
  - `γ_init::AbstractVector`: Initial position in the latent space.
  - `γ_end::AbstractVector`: Final position in the latent space.
- `t::Number`: Time at which to evaluate the right-hand side.
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

# ------------------------------------------------------------------------------
# Numerical integration of the geodesic system
# ------------------------------------------------------------------------------

@doc raw"""
    bc_collocation!(residual, u, param, t)

Evaluates the residuals between the current position in latent space and the
desired boundary conditions. This function is crucial when implementing
collocation methods for boundary value problems in differential equations.

# Arguments
- `residual::AbstractVector`: A vector to store the residuals between the
  desired boundary conditions and the current state.
- `u::AbstractArray{AbstractArray{<:Number}}`: A sequence of state vectors at
  each time step, where each state vector represents a position in latent space.
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that
    defines the manifold.
  - `γ_init::AbstractVector`: Initial position in latent space.
  - `γ_end::AbstractVector`: Final position in latent space.
- `t::Number`: Time at which to evaluate the residuals. Note: this argument is
  present for compatibility with solvers but isn't used directly in this
  function.

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
- `residual::AbstractVector`: A vector to store the residuals between
  the desired boundary conditions and the current state.
- `u::Function`: A function that, when evaluated at a specific time `t`, returns
  the state vector representing a position in latent space.
- `param::Dict`: Parameters necessary for the geodesic differential equation
  system and boundary value integration. The required parameters include:
  - `in_dim::Int`: Dimensionality of the input space.
  - `manifold::Union{Function, Flux.Chain}`: The function or Flux model that 
    defines the manifold.
  - `γ_init::AbstractVector`: Initial position in latent space.
  - `γ_end::AbstractVector`: Final position in latent space.
- `t::Number`: Time at which to evaluate the residuals. Note: this
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