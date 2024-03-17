# Import basic math
import LinearAlgebra
import Distances

# Import library to perform Einstein summation
using TensorOperations: @tensor

# Import library for automatic differentiation
import Zygote

# Import ML library
import Flux

# Import function for k-medoids clustering
using Clustering: kmedoids

# Import Abstract Types
using ..AutoEncode: JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder,
    AbstractDeterministicDecoder, AbstractVariationalDecoder,
    AbstractVariationalEncoder, AbstractDecoder

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Differential Geometry on Riemmanian Manifolds
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    pullback_metric(manifold::Function, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M = JᵀJ` of a manifold using
numerical differentiation with `Zygote.jl`. The metric is computed at a specific
point `z` on the manifold.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `z::AbstractVector`: A vector specifying the point on the manifold where
  the metric should be evaluated.

# Returns
- `M::Matrix`: The Riemannian metric matrix evaluated at `val`.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
M = pullback_metric(manifold, val)
```
"""
function pullback_metric(
    manifold::Function, z::AbstractVector
)
    # Compute Jacobian
    jac = first(Zygote.jacobian(manifold, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::AbstractDeterministicDecoder, z::AbstractVector)

Compute the Riemannian metric (pullback metric) `M = JᵀJ` at a specific point on
a manifold. The manifold is defined by the decoder structure of an autoencoder
(AE). This function uses numerical differentiation with `Zygote.jl` to compute
the metric. The metric is the squared Jacobian matrix, which encapsulates the
local geometric properties of the decoder's output space with respect to its
input space.

# Arguments
- `decoder::AbstractDeterministicDecoder`: A deterministic decoder structure
  that defines the manifold. It should be an instance of a subtype of
  `AbstractDeterministicDecoder`.
- `z::AbstractVector`: A vector specifying the point in the latent space of the
  manifold where the metric should be evaluated.

# Returns
- `M::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::AbstractDeterministicDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(decoder.decoder, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        pullback_metric(decoder::SimpleDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) of a manifold defined by the
decoder structure of a variational autoencoder (VAE) using numerical
differentiation with `Zygote.jl`. The metric is evaluated based on the outputs
of the decoder with respect to its inputs. The metric is defined as

M̲̲ = J̲̲_µᵀ J̲̲_µ

where J̲̲_µ is the Jacobian matrix of the decoder with respect to its input.

The `SimpleDecoder` is a variational decoder that assumes a constant diagonal
covariance matrix in the output. Therefore, the second term of the metric, which
would account for the variance, is not added in this case.

# Arguments
- `decoder::SimpleDecoder`: A Variational decoder structure containing a neural
  network model that defines the manifold. It assumes a constant diagonal
  covariance matrix in the output.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::SimpleDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(decoder.decoder, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::JointLogDecoder, z::AbstractVector) 

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `JointLogDecoder` using numerical differentiation with `Zygote.jl`.
The metric is evaluated based on the outputs of the neural network with respect
to its inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: 

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input. Given that we compute the Jacobian of `logσ` directly, the
Jacobian of `σ` is obtained using the chain rule.

# Arguments
- `decoder::JointLogDecoder`: A VAE decoder structure that has separate paths
  for determining both the mean and log standard deviation of the latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::JointLogDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.µ), z)
    )

    # Compute Jacobian with respect to the input for the log standard deviation
    jac_logσ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.logσ), z)
    )

    # Convert jac_logσ to jac_σ using the chain rule:
    # 1. Compute σ by exponentiating logσ
    σ_val = exp.(Flux.Chain(decoder.decoder..., decoder.logσ)(z))
    # 2. Use the chain rule
    jac_σ = σ_val .* jac_logσ

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

@doc raw"""
    pullback_metric(decoder::SplitLogDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `SplitLogDecoder` using numerical differentiation with `Zygote.jl`.
The metric is evaluated based on the outputs of the individual neural networks
with respect to their inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input. Given that we compute the Jacobian of `logσ` directly, the
Jacobian of `σ` is obtained using the chain rule.

# Arguments
- `decoder::SplitLogDecoder`: A VAE decoder structure that has separate neural
  networks for determining both the mean and log standard deviation of the
  latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::SplitLogDecoder, z::AbstractVector)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(Zygote.jacobian(decoder.µ, z))

    # Compute Jacobian with respect to the input for the log standard deviation
    jac_logσ = first(Zygote.jacobian(decoder.logσ, z))

    # Convert jac_logσ to jac_σ using the chain rule
    σ_val = exp.(decoder.logσ(z))
    jac_σ = σ_val .* jac_logσ

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::JointDecoder, z::AbstractVector) 

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `JointDecoder` using numerical differentiation with
`Zygote.jl`. The metric is evaluated based on the outputs of the neural network
with respect to its inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: 

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input.

# Arguments
- `decoder::JointDecoder`: A VAE decoder structure that has separate paths
  for determining both the mean and standard deviation of the latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::JointDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.µ), z)
    )

    # Compute Jacobian with respect to the input for the standard deviation
    jac_σ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.σ), z)
    )

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::SplitDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `SplitDecoder` using numerical differentiation with
`Zygote.jl`. The metric is evaluated based on the outputs of the individual
neural networks with respect to their inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input.

# Arguments
- `decoder::SplitDecoder`: A VAE decoder structure that has separate
  neural networks for determining both the mean and standard deviation of the
  latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::SplitDecoder, z::AbstractVector)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(Zygote.jacobian(decoder.µ, z))

    # Compute Jacobian with respect to the input for the standard deviation
    jac_σ = first(Zygote.jacobian(decoder.σ, z))

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        pullback_metric(decoder::AbstractDecoder, z::AbstractMatrix)

Compute the Riemannian metric (pull-back metric) for each column of `z` on a
manifold defined by the decoder structure of a variational autoencoder (VAE).
The metric is evaluated based on the outputs of the decoder with respect to its
inputs. 

This function applies the `pullback_metric` function to each column of `z` using
broadcasting.

# Arguments
- `decoder::AbstractDecoder`: A decoder structure containing a neural network
  model that defines the manifold.
- `z::AbstractMatrix`: A matrix where each column represents a point in the
  latent space where the metric should be evaluated.

# Returns
- `M::AbstractArray`: An array of Riemannian metric matrices, one for each
  column of `z`.

# Notes
- The Riemannian metric provides a measure of the "stretching" or "shrinking"
  effect of the decoder at a particular point in the latent space. It is crucial
  for understanding the geometric properties of the learned latent space in
  VAEs.
- The metric is computed using the Jacobian of the decoder, which is obtained
  through automatic differentiation using `Zygote.jl`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::AbstractDecoder, z::AbstractMatrix)
    # Compute pullback metric for each column of z
    return reduce(
        x -> cat(x, dims=3), pullback_metric.(Ref(decoder), eachcol(z))
    )
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
    M̲̲⁻¹ = LinearAlgebra.inv(pullback_metric(manifold, val))

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
# Discretized curve characteristics
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Encoder Riemmanian metric
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Reference
# > Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational 
# > Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).

# ------------------------------------------------------------------------------ 

@doc raw"""
    `exponential_mahalanobis_kernel(x, y, Σ; ρ=1.0f0)`

Compute the exponential Mahalanobis Kernel between two matrices `x` and `y`,
defined as 

ω(x, y) = exp(-(x - y)ᵀ Σ⁻¹ (x - y) / ρ²)

# Arguments
- `x::AbstractVector{Float32}`: First input vector for the kernel.
- `y::AbstractVector{Float32}`: Second input vector for the kernel.
- `Σ::AbstractMatrix{Float32}`: The covariance matrix used in the Mahalanobis
  distance.

# Keyword Arguments
- `ρ::Float32=1.0f0`: Kernel width parameter. Larger ρ values lead to a wider
  spread of the kernel.

# Returns
- `k::AbstractMatrix{Float32}`: Kernel matrix 

# Examples
```julia
x = rand(10, 2) 
y = rand(20, 2)
Σ = I(2)
K = exponential_mahalanobis_kernel(x, y, Σ) # 10x20 kernel matrix
```
"""
function exponential_mahalanobis_kernel(
    x::AbstractVector{Float32},
    y::AbstractVector{Float32},
    Σ::AbstractMatrix;
    ρ::Float32=1.0f0,
)
    # return Gaussian kernel
    return exp(-Distances.sqmahalanobis(x, y, Σ) / ρ^2)
end # function

# ------------------------------------------------------------------------------ 

@doc raw"""
    latent_kmedoids(encoder::AbstractVariationalEncoder,
    data::AbstractMatrix{Float32}, k::Int; distance::Distances.Metric =
    Distances.Euclidean()) -> Vector{Int}

Perform k-medoids clustering on the latent representations of data points
encoded by a Variational Autoencoder (VAE).

This function maps the input data to the latent space using the provided
encoder, computes pairwise distances between the latent representations using
the specified metric, and applies the k-medoids algorithm to partition the data
into k clusters. The function returns a vector of indices corresponding to the
medoids of each cluster.

# Arguments
- `encoder::AbstractVariationalEncoder`: An encoder from a VAE that outputs the
  mean (µ) and log variance (logσ) of the latent space distribution for input
  data.
- `data::AbstractMatrix{Float32}`: The input data matrix, where each column
  represents a data point.
- `k::Int`: The number of clusters to form.

# Optional Keyword Arguments
- `distance::Distances.Metric`: The distance metric to use when computing
  pairwise distances between latent representations, defaulting to
  Distances.Euclidean().

# Returns
- `Vector{Int}`: A vector of indices corresponding to the medoids of the clusters
  in the latent space.

# Example
```julia
# Assuming `encoder` is an instance of `AbstractVariationalEncoder` and `data` 
# is a matrix of Float32s:
medoid_indices = latent_kmedoids(encoder, data, 5)
# `medoid_indices` will hold the indices of the medoids after clustering into 
# 5 clusters.
```
"""
function latent_kmedoids(
    encoder::AbstractVariationalEncoder,
    data::AbstractMatrix{Float32},
    k::Int;
    distance::Distances.Metric=Distances.Euclidean()
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, _ = encoder(data)

    # Compute pairwise distances between elements
    dist = Distances.pairwise(distance, latent_µ, dims=2)

    # Compute kmedoids clustering with Clustering.jl and return indexes.
    return kmedoids(dist, k).medoids
end # function

# ------------------------------------------------------------------------------ 

"""
    encoder_metric_builder(encoder::JointLogEncoder, 
    data::AbstractMatrix{Float32}; λ::AbstractFloat=0.0001f0, 
    τ::AbstractFloat=eps(Float32)) -> Function

Build a metric function G(z) using a trained variational autoencoder (VAE)
model's encoder.

The metric G(z) is defined as a weighted sum of precision matrices Σ⁻¹, each
associated with a data point in the latent space, and a regularization term. The
weights are determined by the exponential Mahalanobis kernel between z and each
data point's latent mean.

# Arguments
- `encoder::JointLogEncoder`: A VAE model encoder that maps data to the latent
  space.
- `data`: The dataset in the form of a matrix, with each column representing a
  data point.

# Optional Keyword Arguments
- `ρ::Union{Nothing, Float32}=nothing`: The maximum Euclidean distance between
  any two closest neighbors in the latent space. If nothing, it will be computed
  from the data. Defaults to nothing.
- `λ::AbstractFloat=0.0001f0`: A regularization parameter that controls the
  strength of the regularization term. Defaults to 0.0001.
- `τ::AbstractFloat=eps(Float32)`: A small positive value to ensure numerical
  stability. Defaults to machine epsilon for Float32.

# Returns
- A function G that takes a vector z and returns the metric matrix.

## Mathematical Definition

For each data point i, the encoder provides the mean μᵢ and log variance logσᵢ
in the latent space. The precision matrix for each data point is given by 

Σ⁻¹ᵢ = Diagonal(exp.(-logσᵢ)). 

The metric is then constructed as follows:

G(z) = ∑ᵢ (Σ⁻¹ᵢ .* exp.(-||z - μᵢ||²_Σ⁻¹ᵢ / ρ²)) + λ * exp(-τ * ||z||²) * Iₙ

where:
- ρ is the maximum Euclidean distance between any two closest neighbors in the
  latent space.
- Iₙ is the identity matrix of size n, where n is the number of dimensions in
  the latent space.
- ||⋅||²_Σ⁻¹ᵢ denotes the squared Mahalanobis distance with respect to Σ⁻¹ᵢ.

# Examples
```julia
# Assuming `encoder` is a pre-trained JointLogEncoder and `data` is your 
# dataset:
G = encoder_metric_builder(encoder, data)
# Compute the metric for a given latent vector
metric_matrix = G(some_latent_vector)  
```

# Reference
> Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational
> Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
"""
function encoder_metric_builder(
    encoder::JointLogEncoder,
    data::AbstractMatrix{Float32};
    ρ::Union{Nothing,Float32}=nothing,
    λ::AbstractFloat=0.0001f0,
    τ::AbstractFloat=eps(Float32),
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, latent_logσ = encoder(data)

    # Generate all inverse covariance matrices (Σ⁻¹) from logσ
    Σ_inv_array = [
        LinearAlgebra.Diagonal(exp.(-latent_logσ[:, i]))
        for i in 1:size(latent_logσ, 2)
    ]

    # Compute Euclidean pairwise distance between points in latent space
    latent_euclid_dist = Distances.pairwise(
        Distances.Euclidean(), latent_µ, dims=2
    )

    # Compute ρ if not provided
    if typeof(ρ) <: Nothing
        # Define ρ as the maximum distance between two closest neighbors
        ρ = maximum([
            minimum(distances[distances.>0])
            for distances in eachrow(latent_euclid_dist)
        ])
    end # if

    # Identity matrix for regularization term, should match the dimensionality
    # of z
    I_d = LinearAlgebra.I(size(latent_µ, 1))

    # Build metric G(z) as a function
    function G(z::AbstractVector{Float32})
        # Initialize metric to zero matrix of appropriate size
        metric = zeros(Float32, size(latent_µ, 1), size(latent_µ, 1))

        # Accumulate weighted Σ_inv matrices, weighted by the kernel
        for i in 1:size(latent_µ, 2)
            # Compute ω terms with exponential mahalanobis kernel
            ω = exponential_mahalanobis_kernel(
                z, latent_µ[:, i], Σ_inv_array[i]; ρ=ρ
            )
            # Update value of metric as the precision metric multiplied
            # (elementwise for some reason) by the resulting value of the kernel
            metric .+= Σ_inv_array[i] .* ω
        end # for

        # Regularization term: λ * exp(-τ||z||²) * I_d
        regularization = λ * exp(-τ * sum(z .^ 2)) .* I_d

        return metric + regularization
    end # function

    return G
end # function

"""
    encoder_metric_builder(
        encoder::JointLogEncoder,
        data::AbstractMatrix{Float32},
        kmedoids_idx::Vector{<:Int};
        ρ::Union{Nothing, Float32}=nothing,
        λ::AbstractFloat=0.0001f0,
        τ::AbstractFloat=eps(Float32)
    ) -> Function

Build a metric function G(z) using a trained variational autoencoder (VAE)
model's encoder, sub-sampling the data points used to build the metric based on
k-medoids indices.

The metric G(z) is defined as a weighted sum of precision matrices Σ⁻¹, each
associated with a subset of data points in the latent space determined by
k-medoids clustering, and a regularization term. The weights are determined by
the exponential Mahalanobis kernel between z and each data point's latent mean
from the sub-sampled set.

# Arguments
- `encoder::JointLogEncoder`: A VAE model encoder that maps data to the latent
  space.
- `data`: The dataset in the form of a matrix, with each column representing a
  data point.
- `kmedoids_idx::Vector{<:Int}`: Indices of data points chosen as medoids from
  k-medoids clustering.

# Optional Keyword Arguments
- `ρ::Union{Nothing, Float32}=nothing`: The maximum Euclidean distance between
  any two closest neighbors in the latent space. If nothing, it will be computed
  from the data. Defaults to nothing.
- `λ::AbstractFloat=0.0001f0`: A regularization parameter that controls the
  strength of the regularization term. Defaults to 0.0001.
- `τ::AbstractFloat=eps(Float32)`: A small positive value to ensure numerical
  stability. Defaults to machine epsilon for Float32.

# Returns
- A function G that takes a vector z and returns the metric matrix.

## Mathematical Definition

For each data point i, the encoder provides the mean μᵢ and log variance logσᵢ
in the latent space. The precision matrix for each data point is given by 

Σ⁻¹ᵢ = Diagonal(exp.(-logσᵢ)). 

The metric is then constructed as follows:

G(z) = ∑ᵢ (Σ⁻¹ᵢ .* exp.(-||z - μᵢ||²_Σ⁻¹ᵢ / ρ²)) + λ * exp(-τ * ||z||²) * Iₙ

where:
- ρ is the maximum Euclidean distance between any two closest neighbors in the
  latent space.
- Iₙ is the identity matrix of size n, where n is the number of dimensions in
  the latent space.
- ||⋅||²_Σ⁻¹ᵢ denotes the squared Mahalanobis distance with respect to Σ⁻¹ᵢ.

# Examples
```julia
# Assuming `encoder` is a pre-trained JointLogEncoder, `data` is your dataset,
# and `kmedoids_idx` contains indices from a k-medoids clustering:
G = encoder_metric_builder(encoder, data, kmedoids_idx)
# Compute the metric for a given latent vector
metric_matrix = G(some_latent_vector)
```

# Reference
> Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational
> Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
"""
function encoder_metric_builder(
    encoder::JointLogEncoder,
    data::AbstractMatrix{Float32},
    kmedoids_idx::Vector{<:Int};
    ρ::Union{Nothing,Float32}=nothing,
    λ::AbstractFloat=0.0001f0,
    τ::AbstractFloat=eps(Float32),
)
    # Map data to latent space to get the mean (µ) and log variance (logσ)
    latent_µ, latent_logσ = encoder(data[:, kmedoids_idx])

    # Generate all inverse covariance matrices (Σ⁻¹) from logσ
    Σ_inv_array = [
        LinearAlgebra.Diagonal(exp.(-latent_logσ[:, i]))
        for i in 1:size(latent_logσ, 2)
    ]

    # Compute ρ if not provided
    if typeof(ρ) <: Nothing
        # Compute Euclidean pairwise distance between points in latent space
        latent_euclid_dist = Distances.pairwise(
            Distances.Euclidean(), latent_µ, dims=2
        )
        # Define ρ as the maximum distance between two closest neighbors
        ρ = maximum([
            minimum(distances[distances.>0])
            for distances in eachrow(latent_euclid_dist)
        ])
    end # if

    # Identity matrix for regularization term, should match the dimensionality
    # of z
    I_d = LinearAlgebra.I(size(latent_µ, 1))

    # Build metric G(z) as a function
    function G(z::AbstractVector{Float32})
        # Initialize metric to zero matrix of appropriate size
        metric = zeros(Float32, size(latent_µ, 1), size(latent_µ, 1))

        # Accumulate weighted Σ_inv matrices, weighted by the kernel
        for i in 1:size(latent_µ, 2)
            # Compute ω terms with exponential mahalanobis kernel
            ω = exponential_mahalanobis_kernel(
                z, latent_µ[:, i], Σ_inv_array[i]; ρ=ρ
            )
            # Update value of metric as the precision metric multiplied
            # (elementwise for some reason) by the resulting value of the kernel
            metric .+= Σ_inv_array[i] .* ω
        end # for

        # Regularization term: λ * exp(-τ||z||²) * I_d
        regularization = λ * exp(-τ * sum(z .^ 2)) .* I_d

        return metric + regularization
    end # function

    return G
end # function

# ------------------------------------------------------------------------------ 

# ------------------------------------------------------------------------------ 

