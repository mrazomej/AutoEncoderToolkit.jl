# Import ML libraries
import Flux

# Import AutoDiff backends
import ChainRulesCore
import Zygote

# Import basic math
import LinearAlgebra
import StatsBase

using AutoEncoderToolkit.RHVAEs

@doc raw"""
    G_inv(
        z::AbstractMatrix,
        centroids_latent::AbstractMatrix,
        M::AbstractArray{<:Number,3},
        T::Number,
        λ::Number,
    )

GPU AbstractVector version of the G_inv function.
"""
function RHVAEs._G_inv(
    ::Type{N},
    z::AbstractVector,
    centroids_latent::AbstractMatrix,
    M::AbstractArray{<:Any,3},
    T::Number,
    λ::Number,
) where {N<:CUDA.CuArray}
    # Compute L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²). Notes: 
    # - We use the reshape function to broadcast the operation over the third
    # dimension of M.
    # - The Zygote.dropgrad function is used to prevent the gradient from being
    # computed with respect to T.
    LLexp = M .*
            reshape(
        exp.(-sum((z .- centroids_latent) .^ 2 / Zygote.dropgrad(T^2), dims=1)),
        1, 1, :
    )

    # Compute the regularization term.
    Λ = ChainRulesCore.ignore_derivatives() do
        CUDA.Diagonal(CUDA.ones(eltype(z), length(z), length(z))) .* λ
    end # ignore_derivatives

    # Zygote.dropgrad(CUDA.cu(Matrix(LinearAlgebra.I(length(z)) .* λ)))

    # Return L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ as a matrix. NOTE:
    # - We divide the result by the number of centroids. This is NOT done in the
    #   original implementation, but without it, the metric tensor scales with
    #   the number of centroids.
    return dropdims(StatsBase.mean(LLexp, dims=3), dims=3) + Λ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    G_inv( 
        z::AbstractMatrix,
        centroids_latent::AbstractMatrix,
        M::AbstractArray{<:Any,3},
        T::Number,
        λ::Number,
    )

GPU AbstractMatrix version of the G_inv function.
"""
function RHVAEs._G_inv(
    ::Type{N},
    z::AbstractMatrix,
    centroids_latent::AbstractMatrix,
    M::AbstractArray{<:Any,3},
    T::Number,
    λ::Number,
) where {N<:CUDA.CuArray}
    # Find number of centroids
    n_centroid = size(centroids_latent, 2)
    # Find number of samples
    n_sample = size(z, 2)

    # Reshape arrays to broadcast subtraction
    z = reshape(z, size(z, 1), 1, n_sample)
    centroids_latent = reshape(
        centroids_latent, size(centroids_latent, 1), n_centroid, 1
    )

    # Compute exp(-‖z - cᵢ‖₂² / T²). Notes:
    # - We bradcast the operation by reshaping the input arrays.
    # - We use Zygot.dropgrad to prevent the gradient from being computed for T.
    # - The result is a 3D array of size (1, n_centroid, n_sample).
    exp_term = exp.(-sum(
        (z .- centroids_latent) .^ 2 / Zygote.dropgrad(T^2),
        dims=1
    ))

    # Reshape exp_term to broadcast multiplication
    exp_term = reshape(exp_term, 1, 1, n_centroid, n_sample)

    # Perform the multiplication
    LLexp = M .* exp_term

    # Compute the regularization term.
    Λ = ChainRulesCore.ignore_derivatives() do
        CUDA.Diagonal(CUDA.ones(eltype(z), size(z, 1), size(z, 1))) .* λ
    end # ignore_derivatives

    # Return L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ as a matrix. Note:
    # - We divide the result by the number of centroids. This is NOT done in the
    # original implementation, but without it, the metric tensor scales with the
    # number of centroids.
    return dropdims(StatsBase.mean(LLexp, dims=3), dims=3) .+ Λ
end # function

# ==============================================================================

@doc raw"""
    metric_tensor(
        z::AbstractMatrix,
        metric_param::Union{RHVAE,NamedTuple},
    )

GPU AbstractMatrix version of the metric_tensor function.
"""
function RHVAEs._metric_tensor(
    ::Type{T},
    z::AbstractMatrix,
    metric_param::Union{RHVAEs.RHVAE,NamedTuple},
) where {T<:CUDA.CuArray}
    # Compute the inverse of the metric tensor G at each point in z.
    G⁻¹ = G_inv(z, metric_param)

    # Invert each slice of G⁻¹
    G = reduce(
        (x, y) -> cat(x, y, dims=3),
        last(CUDA.CUBLAS.matinv_batched(collect(eachslice(G⁻¹, dims=3))))
    )
end # function

# ==============================================================================

@doc raw"""
    train!(
        rhvae::RHVAE, 
        x::AbstractArray, 
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Riemannian Hamiltonian
Variational Autoencoder given a specified loss function.

# Arguments
- `rhvae::RHVAE`: A struct containing the elements of a Riemannian Hamiltonian
  Variational Autoencoder.
- `x::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the RHVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the RHVAE by:
1. Computing the gradient of the loss w.r.t the RHVAE parameters.
2. Updating the RHVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function RHVAEs.train!(
    rhvae::RHVAEs.RHVAE,
    x::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=RHVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(rhvae) do rhvae_model
            loss_function(rhvae_model, x; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, rhvae, ∇L[1])

    # Update metric
    update_metric!(rhvae)

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    train!(
        rhvae::RHVAE, 
        x_in::AbstractArray,
        x_out::AbstractArray,
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Riemannian Hamiltonian
Variational Autoencoder given a specified loss function.

# Arguments
- `rhvae::RHVAE`: A struct containing the elements of a Riemannian Hamiltonian
  Variational Autoencoder.
- `x_in::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the RHVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the RHVAE by:
1. Computing the gradient of the loss w.r.t the RHVAE parameters.
2. Updating the RHVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function RHVAEs.train!(
    rhvae::RHVAEs.RHVAE,
    x_in::CUDA.CuArray,
    x_out::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=RHVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(rhvae) do rhvae_model
            loss_function(rhvae_model, x_in, x_out; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, rhvae, ∇L[1])

    # Update metric
    update_metric!(rhvae)

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function