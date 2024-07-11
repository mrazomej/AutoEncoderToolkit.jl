# Import ML libraries
import Flux

# Import ChainRulesCore to ignore functions when computing gradients
using ChainRulesCore: ignore_derivatives

# Import AutoDiff backends
import TaylorDiff
import Zygote
import ForwardDiff

# Import basic math
import LinearAlgebra
import Random
import StatsBase
import Distributions

# Import library to use Ellipsis Notation
using EllipsisNotation

# Import Abstract Types
using ..AutoEncoderToolkit: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder

# Import Concrete Encoder Types
using ..AutoEncoderToolkit: JointGaussianLogEncoder

# Import Concrete Decoder Types
using ..AutoEncoderToolkit: BernoulliDecoder, SimpleGaussianDecoder,
    JointGaussianLogDecoder, SplitGaussianLogDecoder,
    JointGaussianDecoder, SplitGaussianDecoder

# Import Concrete VAE type
using ..AutoEncoderToolkit: VAE

# Import log-probability functions
using ..AutoEncoderToolkit: decoder_loglikelihood, spherical_logprior,
    encoder_logposterior

# Import functions from other modules
using ..VAEs: reparameterize


# Import functions
using ..utils: vec_to_ltri, sample_MvNormalCanon, finite_difference_gradient,
    slogdet, taylordiff_gradient, vec_mat_vec_batched,
    storage_type

using ..HVAEs: quadratic_tempering, null_tempering

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#  Chadebec, C., Mantoux, C. & Allassonnière, S. Geometry-Aware Hamiltonian
#  Variational Auto-Encoder. Preprint at http://arxiv.org/abs/2010.11518 (2020).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Defining MetricChain to compute the Riemannian metric tensor in latent space
# ==============================================================================

@doc raw"""
    MetricChain <: AbstractMetricChain

A `MetricChain` is used to compute the Riemannian metric tensor in the latent
space of a Riemannian Hamiltonian Variational AutoEncoder (RHVAE).

# Fields
- `mlp::Flux.Chain`: A multi-layer perceptron (MLP) consisting of the hidden
  layers. The inputs are first run through this MLP.
- `diag::Flux.Dense`: A dense layer that computes the diagonal elements of a
  lower-triangular matrix. The output of the `mlp` is fed into this layer.
- `lower::Flux.Dense`: A dense layer that computes the off-diagonal elements of
  the lower-triangular matrix. The output of the `mlp` is also fed into this
  layer.

The outputs of `diag` and `lower` are used to construct a lower-triangular
matrix used to compute the Riemannian metric tensor in latent space.

# Note
If the dimension of the latent space is `n`, the number of neurons in the output
layer of `diag` must be `n`, and the number of neurons in the output layer of
`lower` must be `n * (n - 1) ÷ 2`.

# Example
```julia
mlp = Flux.Chain(Dense(10, 10, relu), Dense(10, 10, relu))
diag = Flux.Dense(10, 5)
lower = Flux.Dense(10, 15)
metric_chain = MetricChain(mlp, diag, lower)
```
"""
struct MetricChain
    mlp::Flux.Chain
    diag::Flux.Dense
    lower::Flux.Dense
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor MetricChain

# ------------------------------------------------------------------------------

@doc raw"""
    MetricChain(
        n_input::Int,
        n_latent::Int,
        metric_neurons::Vector{<:Int},
        metric_activation::Vector{<:Function},
        output_activation::Function;
        init::Function=Flux.glorot_uniform
    ) -> MetricChain

Construct a `MetricChain` for computing the Riemannian metric tensor in the
latent space.

# Arguments
- `n_input::Int`: The number of input features.
- `n_latent::Int`: The dimension of the latent space.
- `metric_neurons::Vector{<:Int}`: The number of neurons in each hidden layer of
  the MLP.
- `metric_activation::Vector{<:Function}`: The activation function for each
  hidden layer of the MLP.
- `output_activation::Function`: The activation function for the output layer.
- `init::Function`: The initialization function for the weights in the layers
  (default is `Flux.glorot_uniform`).

# Returns
- `MetricChain`: A `MetricChain` object that includes the MLP, and two dense
  layers for computing the elements of a lower-triangular matrix used to compute
  the Riemannian metric tensor in latent space.
"""
function MetricChain(
    n_input::Int,
    n_latent::Int,
    metric_neurons::Vector{<:Int},
    metric_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check that the number of activation functions matches the number of layers
    if length(metric_activation) != length(metric_neurons)
        error("Each hidden layer needs exactly one activation function")
    end

    # Initialize list to store layers
    mlp_layers = []

    # Add first layer to list
    push!(
        mlp_layers,
        Flux.Dense(
            n_input => metric_neurons[1], metric_activation[1]; init=init
        )
    )

    # Loop over hidden layers
    for i = 2:length(metric_neurons)
        # Add layer to list
        push!(
            mlp_layers,
            Flux.Dense(
                metric_neurons[i-1] => metric_neurons[i], metric_activation[i];
                init=init
            )
        )
    end # for

    # Create the MLP
    mlp = Flux.Chain(mlp_layers...)

    # Create the diag and lower layers. These layers have a number of neurons
    # equal to the number of entries in a lower triangular matrix in the latent
    # space.  
    diag = Flux.Dense(
        metric_neurons[end] => n_latent, output_activation; init=init
    )
    lower = Flux.Dense(
        metric_neurons[end] => n_latent * (n_latent - 1) ÷ 2,
        output_activation; init=init
    )

    return MetricChain(mlp, diag, lower)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    (m::MetricChain)(x::AbstractArray; matrix::Bool=false)

Perform a forward pass through the MetricChain.

# Arguments
- `x::AbstractArray`: The input data to be processed. 
- `matrix::Bool=false`: A boolean flag indicating whether to return the result
  as a lower triangular matrix (if `true`) or as a tuple of diagonal and lower
  off-diagonal elements (if `false`). Defaults to `false`.

# Returns
- If `matrix` is `true`, returns a lower triangular matrix constructed from the
  outputs of the `diag` and `lower` components of the MetricChain.
- If `matrix` is `false`, returns a `NamedTuple` with two elements: `diag`, the
  output of the `diag` component of the MetricChain, and `lower`, the output of
  the `lower` component of the MetricChain.

# Example
```julia
m = MetricChain(...)
x = rand(Float32, 100, 10)
m(x, matrix=true)  # Returns a lower triangular matrix
```
"""
function (m::MetricChain)(x::AbstractArray; matrix::Bool=false)
    # Compute the output of the MLP
    mlp_out = m.mlp(x)

    # Compute the diagonal elements of the lower-triangular matrix
    diag_out = exp.(m.diag(mlp_out))

    # Compute the off-diagonal elements of the lower-triangular matrix
    lower_out = m.lower(mlp_out)

    # Check if matrix should be returned
    if matrix
        return vec_to_ltri(diag_out, lower_out)
    else
        return (diag=diag_out, lower=lower_out,)
    end # if
end # function

# ==============================================================================
# Riemannian Hamiltonian Variational AutoEncoder (RHVAE)
# ==============================================================================

@doc raw"""
    RHVAE{
        V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
    } <: AbstractVariationalAutoEncoder

A Riemannian Hamiltonian Variational AutoEncoder (RHVAE) as described in
Chadebec, C., Mantoux, C. & Allassonnière, S. Geometry-Aware Hamiltonian
Variational Auto-Encoder. Preprint at http://arxiv.org/abs/2010.11518 (2020).

The RHVAE is a type of Variational AutoEncoder (VAE) that incorporates a
Riemannian metric in the latent space. This metric is computed by a
`MetricChain`, which is a struct that contains a multi-layer perceptron (MLP)
and two dense layers for computing the elements of a lower-triangular matrix.

The inverse metric is computed as follows:

G⁻¹(z) = ∑ᵢ₌₁ⁿ L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ

where L_ψᵢ is computed by the `MetricChain`, T is the temperature, λ is a
regularization factor, and each column of `centroids` are the cᵢ.

# Fields
- `vae::V`: The underlying VAE, where `V` is a subtype of `VAE` with an
  `AbstractVariationalEncoder` and an `AbstractVariationalDecoder`.
- `metric_chain::MetricChain`: The `MetricChain` that computes the Riemannian
  metric in the latent space.
- `centroids_data::AbstractArray`: An array where the last dimension
  represents a data point xᵢ from which the centroids cᵢ are computed by passing
  them through the encoder.
- `centroids_latent::AbstractMatrix`: A matrix where each column
  represents a centroid cᵢ in the inverse metric computation.
- `L::AbstractArray{<:Number, 3}`: A 3D array where each slice represents a L_ψᵢ
  matrix. L_ψᵢ can intuitively be seen as the triangular matrix in the Cholesky
  decomposition of G⁻¹(centroids_latentᵢ) up to a regularization factor.
- `M::AbstractArray{<:Number, 3}`: A 3D array where each slice represents a L_ψᵢ
  L_ψᵢᵀ.
- `T::Number`: The temperature parameter in the inverse metric computation.  
- `λ::Number`: The regularization factor in the inverse metric computation.
"""
struct RHVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
    metric_chain::MetricChain
    centroids_data::AbstractArray
    centroids_latent::AbstractMatrix
    L::AbstractArray{<:Number,3}
    M::AbstractArray{<:Number,3}
    T::Number
    λ::Number
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor RHVAE (vae, metric_chain,)

# ------------------------------------------------------------------------------

@doc raw"""
    Flux.gpu(rhvae::RHVAE)

Move the RHVAE model to the GPU.

# Arguments
- `rhvae::RHVAE`: The RHVAE model to be moved to the GPU.

# Returns
- A new RHVAE model where all the data has been moved to the GPU.

# Description
This function moves all the data of an RHVAE model to the GPU. This includes the
VAE model, the metric chain, the centroids data, the centroids latent, the L
matrix, and the M matrix.

# Note
This function is needed because the `@functor` macro only tags `vae` and
`metric_chain` for moving to the GPU. Other data fields of the RHVAE model need
to be manually moved to the GPU.
"""
function Flux.gpu(rhvae::RHVAE)
    return RHVAE(
        Flux.gpu(rhvae.vae),
        Flux.gpu(rhvae.metric_chain),
        Flux.gpu(rhvae.centroids_data),
        Flux.gpu(rhvae.centroids_latent),
        Flux.gpu(rhvae.L),
        Flux.gpu(rhvae.M),
        rhvae.T,
        rhvae.λ,
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    RHVAE(
        vae::VAE, 
        metric_chain::MetricChain, 
        centroids_data::AbstractArray, 
        T::Number, 
        λ::Number
    )

Construct a Riemannian Hamiltonian Variational Autoencoder (RHVAE) from a
standard VAE and a metric chain.

# Arguments
- `vae::VAE`: A standard Variational Autoencoder (VAE) model.
- `metric_chain::MetricChain`: A chain of metrics to be used for the Riemannian
  Hamiltonian Monte Carlo (RHMC) sampler.
- `centroids_data::AbstractArray`: An array of data centroids. Each column
  represents a centroid. `N` is a subtype of `Number`.
- `T::N`: The temperature parameter for the inverse metric tensor. `N` is a
  subtype of `Number`.
- `λ::N`: The regularization parameter for the inverse metric tensor. `N` is a
  subtype of `Number`.

# Returns
- A new `RHVAE` object.

# Description
The constructor initializes the latent centroids and the metric tensor `M` to
their default values. The latent centroids are initialized to a zero matrix of
the same size as `centroids_data`, and `M` is initialized to a 3D array of
identity matrices, one for each centroid.
"""
function RHVAE(
    vae::VAE,
    metric_chain::MetricChain,
    centroids_data::AbstractArray{N},
    T::N,
    λ::N
) where {N<:AbstractFloat}
    # Extract dimensionality of latent space
    ldim = size(vae.encoder.µ.weight, 1)

    # Extract number of centroids_data
    if ndims(centroids_data) == 1
        # Define batch size as 1
        n_centroids = 1
    else
        # Define batch size as the last dimension
        n_centroids = last(size(centroids_data))
    end # if

    # Initialize centroids_latent
    centroids_latent = zeros(eltype(centroids_data), ldim, n_centroids)

    # Initialize L as a 3D array of identity matrices
    L = reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            Matrix{N}(LinearAlgebra.I(ldim))
            for _ in 1:n_centroids
        ]
    )

    # Initialize M as a copy of L
    M = deepcopy(L)

    # Initialize RHVAE
    return RHVAE(
        vae, metric_chain, centroids_data, centroids_latent, L, M, T, λ,
    )
end # function

# ==============================================================================
# Riemannian Metric computations
# ==============================================================================

@doc raw"""
    update_metric(
        rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}
    )

Compute the `centroids_latent` and `M` field of a `RHVAE` instance without
modifying the instance. This method is used when needing to backpropagate
through the RHVAE during training.

# Arguments
- `rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}`:
  The `RHVAE` instance to be updated.

# Returns
- NamedTuple with the following fields:
  - `centroids_latent::Matrix`: A matrix where each column represents a
    centroid cᵢ in the inverse metric computation.
  - `L::Array{<:Number, 3}`: A 3D array where each slice represents a L_ψᵢ matrix.
  - `M::Array{<:Number, 3}`: A 3D array where each slice represents a L_ψᵢ L_ψᵢᵀ.
"""
function update_metric(
    rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}
)
    # Extract centroids_data
    centroids_data = ignore_derivatives() do
        rhvae.centroids_data
    end
    # Run centroids_data through encoder and update centroids_latent
    centroids_latent = rhvae.vae.encoder(centroids_data).µ
    # Run centroids_data through metric_chain and update L
    L = rhvae.metric_chain(centroids_data, matrix=true)
    # # Update M by multiplying L by its transpose
    M = reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            l * LinearAlgebra.transpose(l)
            for l in eachslice(L, dims=3)
        ]
    )

    return (centroids_latent=centroids_latent, L=L, M=M, T=rhvae.T, λ=rhvae.λ)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    update_metric!(
        rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}},
        params::NamedTuple
    )

Update the `centroids_latent` and `M` fields of a `RHVAE` instance in place.

This function takes a `RHVAE` instance and a named tuple `params` containing the
new values for `centroids_latent` and `M`. It updates the `centroids_latent`,
`L`, and `M` fields of the `RHVAE` instance with the provided values.

# Arguments
- `rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}`:
  The `RHVAE` instance to update.
- `params::NamedTuple`: A named tuple containing the new values for
  `centroids_latent` and `M`. Must have the keys `:centroids_latent`, `:L`, and
  `:M`.

# Returns
Nothing. The `RHVAE` instance is updated in place.
"""
function update_metric!(
    rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}},
    params::NamedTuple
)
    # Check that params contains centroids_latent and M
    if !(:centroids_latent in keys(params) && :M in keys(params))
        error("params must contain centroids_latent and M")
    end # if

    # Update centroid_latent values in place
    rhvae.centroids_latent .= params.centroids_latent
    # Update L values in place
    rhvae.L .= params.L
    # Update M values in place
    rhvae.M .= params.M
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    update_metric!(
        rhvae::RHVAE{
            <:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}
        }
    )

Update the `centroids_latent`, and `M` fields of a `RHVAE` instance in place.

This function takes a `RHVAE` instance as input and modifies its
`centroids_latent` and `M` fields. The `centroids_latent` field is updated by
running the `centroids_data` through the encoder of the underlying VAE and
extracting the mean (µ) of the resulting Gaussian distribution. The `M` field is
updated by running each column of the `centroids_data` through the
`metric_chain` and concatenating the results along the third dimension, then
each slice is updated by multiplying each slice of `L` by its transpose and
concating the results along the third dimension.

# Arguments
- `rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}`:
  The `RHVAE` instance to be updated.

# Notes
This function modifies the `RHVAE` instance in place, so it does not return
anything. The changes are made directly to the `centroids_latent`, `L`, and `M`
fields of the input `RHVAE` instance.
"""
function update_metric!(
    rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}
)
    # Extract centroids_data
    centroids_data = rhvae.centroids_data
    # Run centroids_data through encoder and update centroids_latent
    rhvae.centroids_latent .= rhvae.vae.encoder(centroids_data).µ
    # Run centroids_data through metric_chain and update L
    rhvae.L .= rhvae.metric_chain(centroids_data, matrix=true)
    # Update M by multiplying L by its transpose
    rhvae.M .= reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            l * LinearAlgebra.transpose(l)
            for l in eachslice(rhvae.L, dims=3)
        ]
    )
end # function

# ------------------------------------------------------------------------------
# Inverse Metric Tensor computation
# ------------------------------------------------------------------------------

@doc raw"""
    G_inv(
        z::AbstractVecOrMat,
        centroids_latent::AbstractMatrix,
        M::AbstractArray{<:Number,3},
        T::Number,
        λ::Number,
    )

Compute the inverse of the metric tensor G for a given point in the latent
space.

This function takes a point `z` in the latent space, the `centroids_latent` of
the RHVAE instance, a 3D array `M` representing the metric tensor, a temperature
`T`, and a regularization factor `λ`, and computes the inverse of the metric
tensor G at that point. The computation is based on the centroids and the
temperature, as well as a regularization term. The inverse metric is computed as
follows:

G⁻¹(z) = ∑ᵢ₌₁ⁿ L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ,

where L_ψᵢ is computed by the `MetricChain`, T is the temperature, λ is a
regularization factor, and each column of `centroids_latent` are the cᵢ.

# Arguments
- `z::AbstractVecOrMat`: The point in the latent space. If a matrix, each column
  represents a point in the latent space.
- `centroids_latent::AbstractMatrix`: The centroids in the latent space.
- `M::AbstractArray{<:Number,3}`: The 3D array containing the symmetric matrices
  used to compute the inverse metric tensor.
- `T::N`: The temperature.
- `λ::N`: The regularization factor.

# Returns
A matrix or 3D array representing the inverse of the metric tensor G at the
point `z`. If a 3D array, each slice represents the inverse metric tensor at a
different point in the latent space.

# Notes
The computation involves the squared Euclidean distance between z and each
centroid, the exponential of the negative of these distances divided by the
square of the temperature, and a regularization term proportional to the
identity matrix. The result is a matrix of the same size as the latent space.

# GPU support
This function supports CPU and GPU arrays.
"""
function G_inv(
    z::AbstractVecOrMat,
    centroids_latent::AbstractMatrix,
    M::AbstractArray,
    T::Number,
    λ::Number,
)
    return _G_inv(storage_type(z), z, centroids_latent, M, T, λ)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    G_inv( 
        z::AbstractVecOrMat,
        metric_param::Union{RHVAE,NamedTuple},
    )

Compute the inverse of the metric tensor G for a given point in the latent
space.

This function takes a `RHVAE` instance and a point `z` in the latent space, and
computes the inverse of the metric tensor G at that point. The computation is
based on the centroids and the temperature of the `RHVAE` instance, as well as a
regularization term. The inverse metric is computed as follows:

G⁻¹(z) = ∑ᵢ₌₁ⁿ L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ,

where L_ψᵢ is computed by the `MetricChain`, T is the temperature, λ is a
regularization factor, and each column of `centroids_latent` are the cᵢ.

# Arguments
- `z::AbstractVecOrMat`: The point in the latent space. If a matrix,
  each column represents a point in the latent space.
- `metric_param::Union{RHVAE,NamedTuple}`: Either an `RHVAE` instance or a named
  tuple containing the fields `centroids_latent`, `M`, `T`, and `λ`.

# Returns
A matrix representing the inverse of the metric tensor G at the point `z`.

# Notes

The computation involves the squared Euclidean distance between z and each
centroid of the RHVAE instance, the exponential of the negative of these
distances divided by the square of the temperature, and a regularization term
proportional to the identity matrix. The result is a matrix of the same size as
the latent space.
"""
function G_inv(
    z::AbstractVecOrMat,
    metric_param::Union{RHVAE,NamedTuple},
)
    return G_inv(
        z,
        metric_param.centroids_latent,
        metric_param.M,
        metric_param.T,
        metric_param.λ
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    G_inv(
        z::AbstractVector,
        centroids_latent::AbstractMatrix,
        M::AbstractArray{<:Number,3},
        T::Number,
        λ::Number,
    )

AbstractVector version of the G_inv function.
"""
function _G_inv(
    ::Type,
    z::AbstractVector,
    centroids_latent::AbstractMatrix,
    M::AbstractArray,
    T::Number,
    λ::Number,
)
    # Compute L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²). Notes: 
    # - We use the reshape function to broadcast the operation over the third
    # dimension of M.
    # - The ignore_derivatives function is used to prevent the gradient from being
    # computed with respect to T.
    LLexp = M .*
            reshape(
        exp.(-sum((z .- centroids_latent) .^ 2 / ignore_derivatives(T^2), dims=1)),
        1, 1, :
    )

    # Compute the regularization term.
    Λ = ignore_derivatives() do
        Matrix(LinearAlgebra.I(length(z))) .* λ
    end # ignore_derivatives

    # Return L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ as a matrix. Note:
    # - We divide the result by the number of centroids. This is NOT done in the
    # original implementation, but without it, the metric tensor scales with the
    # number of centroids.
    return dropdims(StatsBase.mean(LLexp, dims=3), dims=3) + Λ
end # function

# ------------------------------------------------------------------------------




# ------------------------------------------------------------------------------

@doc raw"""
    G_inv(
        z::AbstractMatrix,
        centroids_latent::AbstractMatrix,
        M::AbstractArray{<:Number,3},
        T::Number,
        λ::Number,
    )

CPU AbstractMatrix version of the G_inv function.
"""
function _G_inv(
    ::Type,
    z::AbstractMatrix,
    centroids_latent::AbstractMatrix,
    M::AbstractArray,
    T::Number,
    λ::Number,
)
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
    # - We use ignore_derivatives to prevent the gradient from being computed for T.
    # - The result is a 3D array of size (1, n_centroid, n_sample).
    exp_term = exp.(-sum(
        (z .- centroids_latent) .^ 2 / ignore_derivatives(T^2),
        dims=1
    ))

    # Reshape exp_term to broadcast multiplication
    exp_term = reshape(exp_term, 1, 1, n_centroid, n_sample)

    # Perform the multiplication
    LLexp = M .* exp_term

    # Compute the regularization term.
    Λ = ignore_derivatives() do
        Matrix(LinearAlgebra.I(size(z, 1))) .* λ
    end # ignore_derivatives

    # Return L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ as a matrix. Note:
    # - We divide the result by the number of centroids. This is NOT done in the
    # original implementation, but without it, the metric tensor scales with the
    # number of centroids.
    return dropdims(StatsBase.mean(LLexp, dims=3), dims=3) .+ Λ
end # function

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Metric Tensor computation
# ------------------------------------------------------------------------------

@doc raw"""
    metric_tensor(
        z::AbstractVecOrMat,
        metric_param::Union{RHVAE,NamedTuple},
    )

Compute the metric tensor G for a given point in the latent space. This function
is a wrapper that determines the type of the input `z` and calls the appropriate
specialized function `_metric_tensor` to perform the actual computation.

This function takes a `RHVAE` instance or a named tuple containing the fields
`centroids_latent`, `M`, `T`, and `λ`, and a point `z` in the latent space, and
computes the metric tensor G at that point. The computation is based on the
inverse of the metric tensor G, which is computed by the `G_inv` function.

# Arguments
- `z::AbstractVecOrMat`: The point in the latent space. If a matrix, each column
  represents a point in the latent space.
- `metric_param::Union{RHVAE,NamedTuple}`: Either an `RHVAE` instance or a named
  tuple containing the fields `centroids_latent`, `M`, `T`, and `λ`.

# Returns
A matrix representing the metric tensor G at the point `z`.

# Notes
The computation involves the inverse of the metric tensor G at the point z. The
result is a matrix of the same size as the latent space.

# GPU Support
This function supports CPU and GPU arrays.
"""
function metric_tensor(
    z::AbstractVecOrMat,
    metric_param::Union{RHVAE,NamedTuple},
)
    return _metric_tensor(storage_type(z), z, metric_param)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    metric_tensor(
        z::AbstractVector,
        metric_param::Union{RHVAE,NamedTuple},
    )

AbstractVector version of the metric_tensor function.
"""
function _metric_tensor(
    ::Type,
    z::AbstractVector,
    metric_param::Union{RHVAE,NamedTuple},
)
    # Compute the inverse of the metric tensor G at the point z and return its
    # inverse.
    return LinearAlgebra.inv(G_inv(z, metric_param))
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    metric_tensor(
        z::AbstractMatrix,
        metric_param::Union{RHVAE,NamedTuple},
    )

CPU AbstractMatrix version of the metric_tensor function.
"""
function _metric_tensor(
    ::Type,
    z::AbstractMatrix,
    metric_param::Union{RHVAE,NamedTuple},
)
    # Compute the inverse of the metric tensor G at each point in z.
    G⁻¹ = G_inv(z, metric_param)

    # Invert each slice of G⁻¹ and return the result.
    reduce(
        (x, y) -> cat(x, y, dims=3),
        [inv(g⁻¹) for g⁻¹ in eachslice(G⁻¹, dims=3)]
    )
end # function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generalized Hamiltonian Dynamics
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# ==============================================================================
# Functions to compute Riemannian log-prior
# ==============================================================================

@doc raw"""
    riemannian_logprior(
        ρ::AbstractVector,
        G⁻¹::AbstractMatrix,
        logdetG::Number;
    )

CPU AbstractVector version of the riemannian_logprior function.
"""
function riemannian_logprior(
    ρ::AbstractVector,
    G⁻¹::AbstractMatrix,
    logdetG::Number,
)
    # Compute ρᵀ G ρ
    ρᵀ_G_ρ = vec_mat_vec_batched(ρ, G⁻¹, ρ)

    # Return the log-prior
    return -0.5f0 * (size(G⁻¹, 1) * log(2.0f0π) + logdetG) - (0.5f0 * ρᵀ_G_ρ)
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    riemannian_logprior(
        ρ::AbstractVector,
        G⁻¹::AbstractMatrix,
        logdetG::Number,
    )

CPU AbstractMatrix version of the riemannian_logprior function.
"""
function riemannian_logprior(
    ρ::AbstractMatrix,
    G⁻¹::AbstractArray,
    logdetG::AbstractVector,
)
    # Compute ρᵀ G ρ
    ρᵀ_G_ρ = vec_mat_vec_batched(ρ, G⁻¹, ρ)

    # Return the log-prior
    return -0.5f0 .* (size(G⁻¹, 1) * log(2.0f0π) .+ logdetG) .-
           (0.5f0 .* ρᵀ_G_ρ)
end # function

# ==============================================================================
# Hamiltonian computations
# ==============================================================================

@doc raw"""
    hamiltonian(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,<:AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        decoder_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
    )

Compute the Hamiltonian for a given point in the latent space and a given
momentum.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, and a `decoder_output`
NamedTuple, and computes the Hamiltonian. The computation is based on the
log-likelihood of the decoder, the log-prior of the latent space, and the
inverse of the metric tensor G at the point `z`.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = -log p(ρ),

where p(ρ) is the log-prior of the momentum.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported, but the last dimension of the
  array should be of size 1.
- `z::AbstractVecOrMat`: The point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor. This should
  be computed elsewhere and should correspond to the given `z` value.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. This should be computed elsewhere and should
  correspond to the given `z` value.
- `decoder::AbstractVariationalDecoder`: The decoder instance. This is not used
  in the computation of the Hamiltonian, but is passed to the
  `decoder_loglikelihood` function to know which method to use.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.

# Returns
A scalar representing the Hamiltonian at the point `z` with the momentum `ρ`.

# Note
The inverse of the Riemannian metric tensor `G⁻¹` is assumed to be computed
elsewhere. The user must ensure that the provided `G⁻¹` corresponds to the given
`z` value.
"""
function hamiltonian(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,<:AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
)
    # 1. Potential energy U(z|x) = -log p(x|z) - log p(z)

    # Compute log-likelihood
    loglikelihood_x_given_z = reconstruction_loglikelihood(
        x, z, decoder, decoder_output
    )

    # Compute log-prior
    logprior_z = position_logprior(z)

    # Define potential energy
    U = -loglikelihood_x_given_z - logprior_z

    # 2. Kinetic energy K(ρ) = -log p(ρ)
    κ = -momentum_logprior(ρ, G⁻¹, logdetG)

    # Return Hamiltonian
    return U + κ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    hamiltonian(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        G_inv::Function=G_inv,
    )

Compute the Hamiltonian for a given point in the latent space and a given
momentum.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, and an instance of `RHVAE`. It computes the inverse of
the Riemannian metric tensor `G⁻¹` and the output of the decoder internally, and
then computes the Hamiltonian. The computation is based on the log-likelihood of
the decoder, the log-prior of the latent space, and the inverse of the metric
tensor G at the point `z`.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = -log p(ρ),

where p(ρ) is the log-prior of the momentum.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported, but the last dimension of the
  array should be of size 1.
- `z::AbstractVector`: The point in the latent space.
- `ρ::AbstractVector`: The momentum.
- `rhvae::RHVAE`: An instance of the RHVAE model.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`. This function must take as input the point
  `z` in the latent space and the `rhvae` instance.

# Returns
A scalar representing the Hamiltonian at the point `z` with the momentum `ρ`.

# Note
The inverse of the Riemannian metric tensor `G⁻¹`, the log determinant of the
metric tensor, and the output of the decoder are computed internally in this
function. The user does not need to provide these as inputs.
"""
function hamiltonian(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    G_inv::Function=G_inv,
)
    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    # Call hamiltonian function with metric_param as NamedTuple
    return hamiltonian(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
    )
end # function

# ==============================================================================
# Hamiltonian gradient computation
# ==============================================================================

# ------------------------------------------------------------------------------
# Finite difference method
# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_finite(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        fdtype::Symbol=:central,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using a
naive finite difference method.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, a `decoder_output` NamedTuple,
and a variable `var` (:z or :ρ), and computes the gradient of the Hamiltonian
with respect to `var` using a simple finite differences method. The computation
is based on the log-likelihood of the decoder, the log-prior of the latent
space, and `G⁻¹`.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not
  necessarily need to be a vector. Array inputs are supported. The last
  dimension is assumed to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix,
  each column represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column
  represents a momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor.
  If 3D array, each slice along the third dimension represents the inverse of
  the metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of
  the Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and `G⁻¹`.
- `fdtype::Symbol=:central`: The type of finite difference method to use. Must
  be :central or :forward. Default is :central.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.
"""
function ∇hamiltonian_finite(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    fdtype::Symbol=:central,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute gradient with respect to var
    if var == :z
        return finite_difference_gradient(
            z -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            z; fdtype=fdtype
        )
    elseif var == :ρ
        return finite_difference_gradient(
            ρ -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            ρ; fdtype=fdtype
        )
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_finite(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        G_inv::Function=G_inv,
        fdtype::Symbol=:central,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using a
naive finite difference method.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, an instance of `RHVAE`, and a variable `var` (:z or :ρ),
and computes the gradient of the Hamiltonian with respect to `var` using a
simple finite differences method. The computation is based on the log-likelihood
of the decoder, the log-prior of the latent space, and the inverse of the metric
tensor G at the point `z`.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not
  necessarily need to be a vector. Array inputs are supported. The last
  dimension is assumed to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix,
  each column represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column
  represents a momentum vector.
- `rhvae::RHVAE`: An instance of the RHVAE model.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`. This function must take as input the point
  `z` in the latent space and the `rhvae` instance.
- `fdtype::Symbol=:central`: The type of finite difference method to use. Must
  be :central or :forward. Default is :central.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.

# Note
The inverse of the Riemannian metric tensor `G⁻¹`, the log determinant of the
metric tensor, and the output of the decoder are computed internally in this
function. The user does not need to provide these as inputs.
"""
function ∇hamiltonian_finite(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    G_inv::Function=G_inv,
    fdtype::Symbol=:central,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    return ∇hamiltonian_finite(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output, var;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
        fdtype=fdtype
    )
end # function

# ------------------------------------------------------------------------------
# ForwardDiff.jl method
# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_ForwardDiff(
        x::AbstractArray,
        z::AbstractVector,
        ρ::AbstractVector,
        G⁻¹::AbstractMatrix,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using
the ForwardDiff.jl automatic differentiation library.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, a `decoder_output` NamedTuple,
and a variable `var` (:z or :ρ), and computes the gradient of the Hamiltonian
with respect to `var` using ForwardDiff.jl.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not
  necessarily need to be a vector. Array inputs are supported. The last
  dimension is assumed to have each of the data points.
- `z::AbstractVector`: The point in the latent space.
- `ρ::AbstractVector`: The momentum.
- `G⁻¹::AbstractMatrix`: The inverse of the Riemannian metric tensor.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of
  the Riemannian metric tensor.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and `G⁻¹`.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.

# Note
`ForwardDiff.jl` is not composable with `Zygote.jl.` Thus, for backpropagation
using this function one should use `ReverseDiff.jl.`
"""
function ∇hamiltonian_ForwardDiff(
    x::AbstractArray,
    z::AbstractVector,
    ρ::AbstractVector,
    G⁻¹::AbstractMatrix,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute gradient with respect to var
    if var == :z
        return ForwardDiff.gradient(
            z -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            z
        )
    elseif var == :ρ
        return ForwardDiff.gradient(
            ρ -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            ρ
        )
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_ForwardDiff(
        x::AbstractArray,
        z::AbstractMatrix,
        ρ::AbstractMatrix,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using
the ForwardDiff.jl automatic differentiation library.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, a `decoder_output` NamedTuple,
and a variable `var` (:z or :ρ), and computes the gradient of the Hamiltonian
with respect to `var` using ForwardDiff.jl.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

The Jacobian is computed with respect to `var` to compute derivatives for all
columns at once. The relevant terms for each column's gradient are then
extracted from the Jacobian.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not
  necessarily need to be a vector. Array inputs are supported. The last
  dimension is assumed to have each of the data points.
- `z::AbstractMatrix`: The point in the latent space.
- `ρ::AbstractMatrix`: The momentum.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of
  the Riemannian metric tensor.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and `G⁻¹`.

# Returns
A matrix representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.

# Note
`ForwardDiff.jl` is not composable with `Zygote.jl.` Thus, for backpropagation
using this function one should use `ReverseDiff.jl.`
"""
function ∇hamiltonian_ForwardDiff(
    x::AbstractArray,
    z::AbstractMatrix,
    ρ::AbstractMatrix,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute Jacobian with respect to var to compute derivatives for all
    # columns. In this way, we can compute all derivatives at once. Later, we
    # will extract the relevant terms for each column's gradient.
    if var == :z
        jac = ForwardDiff.jacobian(
            z -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            z
        )
    elseif var == :ρ
        jac = ForwardDiff.jacobian(
            ρ -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            ρ
        )
    end # if

    # Extract dimensionality of latent space
    ldim = size(z, 1)
    # Extract the elements of the Jacobian that represent each column's gradient
    # This is done by extracting the corresponding non-zeros entries from each
    # row and arranging them in a matrix
    grad = reduce(
        hcat,
        [jac[i, (i-1)*ldim+1:(i-1)*ldim+ldim] for i in axes(jac, 1)]
    )

    return grad
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_ForwardDiff(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        G_inv::Function=G_inv,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using
the ForwardDiff.jl automatic differentiation library.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, an instance of `RHVAE`, and a variable `var` (:z or :ρ),
and computes the gradient of the Hamiltonian with respect to `var` using
ForwardDiff.jl.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not
  necessarily need to be a vector. Array inputs are supported. The last
  dimension is assumed to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix,
  each column represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column
  represents a momentum vector.
- `rhvae::RHVAE`: An instance of the RHVAE model.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`. This function must take as input the point
  `z` in the latent space and the `rhvae` instance.

# Returns
A matrix representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.

# Note
`ForwardDiff.jl` is not composable with `Zygote.jl.` Thus, for backpropagation
using this function one should use `ReverseDiff.jl.`
"""
function ∇hamiltonian_ForwardDiff(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    G_inv::Function=G_inv,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    return ∇hamiltonian_ForwardDiff(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output, var;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
    )
end # function

# ------------------------------------------------------------------------------
# TaylorDiff.jl method
# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_TaylorDiff(
        x::AbstractArray,
        z::AbstractVector,
        ρ::AbstractVector,
        G⁻¹::AbstractMatrix,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using
the TaylorDiff.jl automatic differentiation library.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, an instance of `AbstractVariationalDecoder`, and a
variable `var` (:z or :ρ), and computes the gradient of the Hamiltonian with
respect to `var` using TaylorDiff.jl.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVector`: The point in the latent space.
- `ρ::AbstractVector`: The momentum.
- `G⁻¹::AbstractMatrix`: The inverse of the Riemannian metric tensor.
- `logdetG::Number`: The logarithm of the determinant of the Riemannian metric
  tensor.
- `decoder::AbstractVariationalDecoder`: An instance of the decoder model.
- `decoder_output::NamedTuple`: The output of the decoder model.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as
  input the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.

# Note
`TaylorDiff.jl` is composable with `Zygote.jl.` Thus, for backpropagation using
this function one should use `Zygote.jl.`
"""
function ∇hamiltonian_TaylorDiff(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute gradient with respect to var
    if var == :z
        return taylordiff_gradient(
            z -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            z
        )
    elseif var == :ρ
        return taylordiff_gradient(
            ρ -> hamiltonian(
                x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
                reconstruction_loglikelihood=reconstruction_loglikelihood,
                position_logprior=position_logprior,
                momentum_logprior=momentum_logprior,
            ),
            ρ
        )
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian_TaylorDiff(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        G_inv::Function=G_inv,
    )

Compute the gradient of the Hamiltonian with respect to a given variable using
the TaylorDiff.jl automatic differentiation library.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, an instance of `RHVAE`, and a variable `var` (:z or :ρ),
and computes the gradient of the Hamiltonian with respect to `var` using
TaylorDiff.jl.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column represents a
  momentum vector.
- `rhvae::RHVAE`: An instance of the RHVAE model.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and the inverse of the Riemannian metric tensor `G⁻¹`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`. This function must take as input the point
  `z` in the latent space and the `rhvae` instance.

# Returns
A matrix representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.
"""
function ∇hamiltonian_TaylorDiff(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    G_inv::Function=G_inv,
)
    # Check that var is a valid variable
    if var ∉ (:z, :ρ)
        error("var must be :z or :ρ")
    end # if

    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    return ∇hamiltonian_TaylorDiff(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output, var;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
    )
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

"""
    ∇Hrhvae

A `NamedTuple` mapping automatic differentiation types to their corresponding
gradient computation functions for the Hamiltonian.
"""
const ∇Hrhvae = (
    finite=∇hamiltonian_finite,
    TaylorDiff=∇hamiltonian_TaylorDiff,
)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        adtype::Symbol=:TaylorDiff,
        adkwargs::Union{NamedTuple,Dict}=Dict(),
    )

Compute the gradient of the Hamiltonian with respect to a given variable using a
specified automatic differentiation method.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, a `decoder_output` NamedTuple,
and a variable `var` (:z or :ρ), and computes the gradient of the Hamiltonian
with respect to `var` using the specified automatic differentiation method. The
computation is based on the log-likelihood of the decoder, the log-prior of the
latent space, and `G⁻¹`.

The Hamiltonian is computed as follows:

Hₓ(z, ρ) = Uₓ(z) + κ(ρ),

where Uₓ(z) is the potential energy, and κ(ρ) is the kinetic energy. The
potential energy is defined as follows:

Uₓ(z) = -log p(x|z) - log p(z),

where p(x|z) is the log-likelihood of the decoder and p(z) is the log-prior in
latent space. The kinetic energy is defined as follows:

κ(ρ) = 0.5 * log((2π)ᴰ det G(z)) + 0.5 * ρᵀ G⁻¹ ρ

where D is the dimension of the latent space, and G(z) is the metric tensor at
the point `z`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column represents a
  momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor.  If 3D
  array, each slice along the third dimension represents the inverse of the
  metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and `G⁻¹`.
- `adtype::Symbol`=:finite`: The type of automatic differentiation method to
  use. Must be `:finite`, `:ForwardDiff`, or `:TaylorDiff`. Default is
  `:finite`.
- `adkwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments to
  pass to the automatic differentiation method.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.
"""
function ∇hamiltonian(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    adtype::Symbol=:finite,
    adkwargs::Union{NamedTuple,Dict}=Dict(),
)
    # Check that the provided adtype is valid
    if adtype ∉ keys(∇Hrhvae)
        # If automatic differentiation method is specified, check if it is valid
        error("adtype must be one of $(keys(∇Hrhvae))")
    end

    # Compute gradient with respect to var
    return ∇Hrhvae[adtype](
        x, z, ρ, G⁻¹, logdetG, decoder, decoder_output, var;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
        adkwargs...
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    ∇hamiltonian(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE,
        var::Symbol;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        G_inv::Function=G_inv,
        adtype::Symbol=:TaylorDiff,
        adkwargs::Union{NamedTuple,Dict}=Dict(),
    )

Compute the gradient of the Hamiltonian with respect to a given variable using a
specified automatic differentiation method.

This function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, an instance of `RHVAE`, and a variable `var` (:z or :ρ),
and computes the gradient of the Hamiltonian with respect to `var` using the
specified automatic differentiation method. The computation is based on the
log-likelihood of the decoder, the log-prior of the latent space, and `G_inv`.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrix, each column represents a
  momentum vector.
- `rhvae::RHVAE`: An instance of the RHVAE model.
- `var::Symbol`: The variable with respect to which the gradient is computed.
  Must be :z or :ρ.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the
  log-likelihood of the decoder reconstruction. Default is
  `decoder_loglikelihood`. This function must take as input the decoder, the
  point `x` in the data space, and the `decoder_output`.
- `position_logprior::Function`: The function to compute the log-prior of the
  latent space position. Default is `spherical_logprior`. This function must
  take as input the point `z` in the latent space.
- `momentum_logprior::Function`: The function to compute the log-prior of the
  momentum. Default is `riemannian_logprior`. This function must take as input
  the momentum `ρ` and `G_inv`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor.  Default is `G_inv`.
- `adtype::Symbol`=:finite`: The type of automatic differentiation method to
  use. Must be `:finite`, `:ForwardDiff`, or `:TaylorDiff`. Default is
  `:finite`.
- `adkwargs::Union{NamedTuple,Dict}=Dict()`: Additional keyword arguments to
  pass to the automatic differentiation method.

# Returns
A vector representing the gradient of the Hamiltonian at the point `(z, ρ)` with
respect to variable `var`.
"""
function ∇hamiltonian(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE,
    var::Symbol;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    G_inv::Function=G_inv,
    adtype::Symbol=:finite,
    adkwargs::Union{NamedTuple,Dict}=Dict(),
)
    # Check that the provided adtype is valid
    if adtype ∉ keys(∇Hrhvae)
        # If automatic differentiation method is specified, check if it is valid
        error("adtype must be one of $(keys(∇Hrhvae))")
    end

    # Compute gradient with respect to var
    return ∇Hrhvae[adtype](
        x, z, ρ, rhvae, var;
        reconstruction_loglikelihood=reconstruction_loglikelihood,
        position_logprior=position_logprior,
        momentum_logprior=momentum_logprior,
        G_inv=G_inv,
        adkwargs...
    )
end # function

# ==============================================================================
# Generalized Leapfrog Integrator
# ==============================================================================

@doc raw"""
    _leapfrog_first_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
        ),
    )

Perform the first step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ * ∇z_H(z(t), ρ(t + ϵ/2)).

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, the output of the decoder
`decoder_output`, a step size `ϵ`, and optionally the number of fixed-point
iterations to perform (`steps`), a function to compute the gradient of the
Hamiltonian (`∇H`), and a set of keyword arguments for `∇H` (`∇H_kwargs`).

The function performs the following update for `steps` times:

ρ̃ = ρ̃ - 0.5 * ϵ * ∇hamiltonian(x, z, ρ̃, G⁻¹, decoder, decoder_output, :z;
∇H_kwargs...)

where `∇H` is the gradient of the Hamiltonian with respect to the position
variables `z`. The result is returned as ρ̃.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor. If 3D
  array, each slice along the third dimension represents the inverse of the
  metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The leapfrog step size. Default
  is 0.01f0.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  `momentum_logprior`, and `G_inv`.

# Returns
A vector representing the updated momentum after performing the first step of
the generalized leapfrog integrator.
"""
function _leapfrog_first_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
)
    # Copy ρ to iterate over it
    ρ̃ = deepcopy(ρ)

    # Loop over steps
    for _ in 1:steps
        # Update momentum variable into a new temporary variable
        ρ̃_ = ρ̃ - (0.5f0 * ϵ) .* ∇hamiltonian(
            x, z, ρ̃, G⁻¹, logdetG, decoder, decoder_output, :z; ∇H_kwargs...
        )
        # Update momentum variable for next cycle
        ρ̃ = ρ̃_
    end # for

    return ρ̃
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _leapfrog_first_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
    )

Perform the first step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ * ∇z_H(z(t), ρ(t + ϵ/2)).

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a `RHVAE` instance, a point `x` in the data space, a point
`z` in the latent space, a momentum `ρ`, a step size `ϵ`, and optionally the
number of fixed-point iterations to perform (`steps`), a function to compute the
gradient of the Hamiltonian (`∇H`), and a set of keyword arguments for `∇H`
(`∇H_kwargs`).

The function performs the following update for `steps` times:

ρ̃ = ρ̃ - 0.5 * ϵ * ∇hamiltonian(rhvae, x, z, ρ̃, :z; ∇H_kwargs...)

where `∇H` is the gradient of the Hamiltonian with respect to the position
variables `z`. The result is returned as ρ̃.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `rhvae::RHVAE`: The `RHVAE` instance.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The leapfrog step size. Default
  is 0.01f0.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  and `momentum_logprior`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`.

# Returns
A vector representing the updated momentum after performing the first step of
the generalized leapfrog integrator.
"""
function _leapfrog_first_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
)
    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    # Call _leapfrog_first_step function with metric_param as NamedTuple
    return _leapfrog_first_step(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        ϵ=ϵ,
        steps=steps,
        ∇H_kwargs=∇H_kwargs,
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _leapfrog_second_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
        ),
    )

Perform the second step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

z(t + ϵ) = z(t) + 0.5 * ϵ * [∇ρ_H(z(t), ρ(t+ϵ/2)) + ∇ρ_H(z(t + ϵ), ρ(t+ϵ/2))].

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, the output of the decoder
`decoder_output`, a step size `ϵ`, and optionally the number of fixed-point
iterations to perform (`steps`), a function to compute the gradient of the
Hamiltonian (`∇H`), and a set of keyword arguments for `∇H` (`∇H_kwargs`).

The function performs the following update for `steps` times:

z̄ = z̄ + 0.5 * ϵ * ( 
    ∇hamiltonian(x, z̄, ρ, G⁻¹, decoder, decoder_output, :ρ; ∇H_kwargs...) + 
    ∇hamiltonian(x, z, ρ, G⁻¹, decoder, decoder_output, :ρ; ∇H_kwargs...) 
)

where `∇H` is the gradient of the Hamiltonian with respect to the momentum
variables `ρ`. The result is returned as z̄.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor. If 3D
  array, each slice along the third dimension represents the inverse of the
  metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The step size. Default is 0.01.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  `momentum_logprior`.

# Returns
A vector representing the updated position after performing the second step of
the generalized leapfrog integrator.
"""
function _leapfrog_second_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
)
    # Compute Hamiltonian gradient for initial point not to repeat it at each
    # iteration 
    ∇Hₒ = ∇hamiltonian(
        x, z, ρ, G⁻¹, logdetG, decoder, decoder_output, :ρ; ∇H_kwargs...
    )

    # Copy z to iterate over it
    z̄ = deepcopy(z)

    # Loop over steps
    for _ in 1:steps
        # Update position variable into a new temporary variable
        z̄_ = z̄ + (0.5f0 * ϵ) .*
                   (∇Hₒ +
                    ∇hamiltonian(
            x, z̄, ρ, G⁻¹, logdetG, decoder, decoder_output, :ρ;
            ∇H_kwargs...
        ))

        # Update position variable for next cycle
        z̄ = z̄_
    end # for

    return z̄
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _leapfrog_second_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
    )

Perform the second step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

z(t + ϵ) = z(t) + 0.5 * ϵ * [∇ρ_H(z(t), ρ(t+ϵ/2)) + ∇ρ_H(z(t + ϵ), ρ(t+ϵ/2))].

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a `RHVAE` instance, a point `x` in the data space, a point
`z` in the latent space, a momentum `ρ`, a step size `ϵ`, and optionally the
number of fixed-point iterations to perform (`steps`), a function to compute the
gradient of the Hamiltonian (`∇H`), and a set of keyword arguments for `∇H`
(`∇H_kwargs`).

The function performs the following update for `steps` times:

z̄ = z̄ + 0.5 * ϵ * ( ∇hamiltonian(rhvae, x, z̄, ρ, :ρ; ∇H_kwargs...) +
        ∇hamiltonian(rhvae, x, z, ρ, :ρ; ∇H_kwargs...) )

where `∇H` is the gradient of the Hamiltonian with respect to the momentum
variables `ρ`. The result is returned as z̄.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `rhvae::RHVAE`: The `RHVAE` instance.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The leapfrog step size. Default
  is 0.01f0.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
  Typically, 3 iterations are sufficient.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  and `momentum_logprior`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`.

# Returns
A vector representing the updated position after performing the second step of
the generalized leapfrog integrator.
"""
function _leapfrog_second_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
)
    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    # Call _leapfrog_first_step function with metric_param as NamedTuple
    return _leapfrog_second_step(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        ϵ=ϵ,
        steps=steps,
        ∇H_kwargs=∇H_kwargs,
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _leapfrog_third_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
        ),
    )

Perform the third step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ * ∇z_H(z(t + ϵ), ρ(t + ϵ/2)).

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a point `x` in the data space, a point `z` in the latent
space, a momentum `ρ`, the inverse of the Riemannian metric tensor `G⁻¹`, a
`decoder` of type `AbstractVariationalDecoder`, the output of the decoder
`decoder_output`, a step size `ϵ`, a function to compute the gradient of the
Hamiltonian (`∇H`), and a set of keyword arguments for `∇H` (`∇H_kwargs`).

The function performs the following update:

ρ̃ = ρ - 0.5 * ϵ * ∇hamiltonian( x, z, ρ, G⁻¹, decoder, decoder_output, :z;
    ∇H_kwargs... )

where `∇H` is the gradient of the Hamiltonian with respect to the position
variables `z`. The result is returned as ρ̃.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor. If 3D
  array, each slice along the third dimension represents the inverse of the
  metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The step size. Default is
  0.01f0.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  `momentum_logprior`.

# Returns
A vector representing the updated momentum after performing the third step of
the generalized leapfrog integrator.
"""
function _leapfrog_third_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    ∇H::Function=∇hamiltonian_TaylorDiff,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
)
    # Update momentum variable with half step. No fixed-point iterations are
    # needed.
    return ρ - (0.5f0 * ϵ) .* ∇hamiltonian(
        x, z, ρ, G⁻¹, logdetG, decoder, decoder_output, :z; ∇H_kwargs...
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _leapfrog_third_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
    )

Perform the third step of the generalized leapfrog integrator for Hamiltonian
dynamics, defined as

ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ * ∇z_H(z(t + ϵ), ρ(t + ϵ/2)).

This function is part of the generalized leapfrog integrator used in Hamiltonian
dynamics. Unlike the standard leapfrog integrator, the generalized leapfrog
integrator is implicit, which means it requires the use of fixed-point
iterations to be solved.

The function takes a `RHVAE` instance, a point `x` in the data space, a point
`z` in the latent space, a momentum `ρ`, a step size `ϵ`, the number of
fixed-point iterations to perform (`steps`), a function to compute the gradient
of the Hamiltonian (`∇H`), and a set of keyword arguments for `∇H`
(`∇H_kwargs`).

The function performs the following update:

ρ̃ = ρ - 0.5 * ϵ * ∇hamiltonian(rhvae, x, z, ρ, :z; ∇H_kwargs...)

where `∇H` is the gradient of the Hamiltonian with respect to the position
variables `z`. The result is returned as ρ̃.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `rhvae::RHVAE`: The `RHVAE` instance.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The leapfrog step size. Default is
  0.01f0.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `reconstruction_loglikelihood`, `position_logprior`,
  and `momentum_logprior`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`.

# Returns
A vector representing the updated momentum after performing the third step of
the generalized leapfrog integrator.
"""
function _leapfrog_third_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
)
    # Compute inverse of the metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    # Call _leapfrog_first_step function with metric_param as NamedTuple
    return _leapfrog_third_step(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        ϵ=ϵ,
        ∇H_kwargs=∇H_kwargs,
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    general_leapfrog_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        G⁻¹::AbstractArray,
        logdetG::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        metric_param::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
    )

Perform a full step of the generalized leapfrog integrator for Hamiltonian
dynamics.

The leapfrog integrator is a numerical integration scheme used to simulate
Hamiltonian dynamics. It consists of three steps:

1. Half update of the momentum variable: 

    ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ * ∇z_H(z(t), ρ(t + ϵ/2)).

2. Full update of the position variable: 

z(t + ϵ) = z(t) + 0.5 * ϵ * [∇ρ_H(z(t), ρ(t+ϵ/2)) + ∇ρ_H(z(t + ϵ), ρ(t+ϵ/2))].

3. Half update of the momentum variable: 

    ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ * ∇z_H(z(t + ϵ), ρ(t + ϵ/2)).

This function performs these three steps in sequence, using the
`_leapfrog_first_step`, `_leapfrog_second_step` and `_leapfrog_third_step`
helper functions.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `G⁻¹::AbstractArray`: The inverse of the Riemannian metric tensor. If 3D
  array, each slice along the third dimension represents the inverse of the
  metric tensor at the corresponding column of `z`.
- `logdetG::Union{<:Number,AbstractVector}`: The log determinant of the
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `z`.
- `decoder::AbstractVariationalDecoder`: The decoder instance.
- `decoder_output::NamedTuple`: The output of the decoder.
- `metric_param::NamedTuple`: The parameters for the metric tensor.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The step size. Default is 0.01.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
  Typically, 3 iterations are sufficient.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `decoder_loglikelihood`, `position_logprior`,
  `momentum_logprior`, and `G_inv`.
- `G_inv::Function=G_inv`: The function to compute the inverse of the Riemannian
  metric tensor.

# Returns
A tuple `(z̄, ρ̄, Ḡ⁻¹, logdetḠ, decoder_update)` representing the updated
position, momentum, the inverse of the updated Riemannian metric tensor, the log
of the determinant of the metric tensor and the updated decoder outputs after
performing the full leapfrog step.
"""
function general_leapfrog_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    G⁻¹::AbstractArray,
    logdetG::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    metric_param::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
)
    # Update momentum variable with half step. This step peforms fixed-point
    # iterations
    ρ̃ = _leapfrog_first_step(
        x, z, ρ, G⁻¹, logdetG, decoder, decoder_output;
        ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs
    )

    # Update position variable with full step. This step peforms fixed-point
    # iterations
    z̄ = _leapfrog_second_step(
        x, z, ρ̃, G⁻¹, logdetG, decoder, decoder_output;
        ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs
    )

    # Update decoder output
    decoder_output_z̄ = decoder(z̄)

    # Update Riemannian metric tensor
    Ḡ⁻¹ = G_inv(z̄, metric_param)

    # Update log determinant of the metric tensor
    logdetḠ = -slogdet(Ḡ⁻¹)

    # Update momentum variable with half step. No fixed-point iterations needed
    ρ̄ = _leapfrog_third_step(
        x, z̄, ρ̃, Ḡ⁻¹, logdetḠ, decoder, decoder_output_z̄;
        ϵ=ϵ, ∇H_kwargs=∇H_kwargs
    )

    return z̄, ρ̄, Ḡ⁻¹, logdetḠ, decoder_output_z̄
end # function 

# ------------------------------------------------------------------------------

@doc raw"""
    general_leapfrog_step(
        x::AbstractArray,
        z::AbstractVecOrMat,
        ρ::AbstractVecOrMat,
        rhvae::RHVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
                G_inv=G_inv,
        ),
    )

Perform a full step of the generalized leapfrog integrator for Hamiltonian
dynamics.

The leapfrog integrator is a numerical integration scheme used to simulate
Hamiltonian dynamics. It consists of three steps:

1. Half update of the momentum variable: ρ(t + ϵ/2) = ρ(t) - 0.5 * ϵ *
    ∇z_H(z(t), ρ(t + ϵ/2)).

2. Full update of the position variable: z(t + ϵ) = z(t) + 0.5 * ϵ * [∇ρ_H(z(t),
ρ(t+ϵ/2)) + ∇ρ_H(z(t + ϵ), ρ(t+ϵ/2))].

3. Half update of the momentum variable: ρ(t + ϵ) = ρ(t + ϵ/2) - 0.5 * ϵ *
    ∇z_H(z(t + ϵ), ρ(t + ϵ/2)).

This function performs these three steps in sequence, using the
`_leapfrog_first_step` and `_leapfrog_second_step` helper functions.

# Arguments
- `x::AbstractArray`: The point in the data space. This does not necessarily
  need to be a vector. Array inputs are supported. The last dimension is assumed
  to have each of the data points.
- `z::AbstractVecOrMat`: The point in the latent space. If matrix, each column
  represents a point in the latent space.
- `ρ::AbstractVecOrMat`: The momentum. If matrux, each column represents a
  momentum vector.
- `rhvae::RHVAE`: The `RHVAE` instance.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The leapfrog step size. Default
  is 0.01f0.
- `steps::Int=3`: The number of fixed-point iterations to perform. Default is 3.
  Typically, 3 iterations are sufficient.
- `∇H_kwargs::Union{NamedTuple,Dict}`: The keyword arguments for `∇hamiltonian`.
  Default is a tuple with `decoder_loglikelihood`, `position_logprior`, and
  `momentum_logprior`
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Default is `G_inv`.

  A tuple `(z̄, ρ̄, Ḡ⁻¹, logdetḠ, decoder_update)` representing the updated
  position, momentum, the inverse of the updated Riemannian metric tensor, the
  log of the determinant of the metric tensor, and the updated decoder outputs
  after performing the full leapfrog step.
"""
function general_leapfrog_step(
    x::AbstractArray,
    z::AbstractVecOrMat,
    ρ::AbstractVecOrMat,
    rhvae::RHVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
)
    # Compute the riemannian metric tensor
    G⁻¹ = G_inv(z, rhvae)

    # Compute the log determinant of the metric tensor
    logdetG = -slogdet(G⁻¹)

    # Compute the output of the decoder
    decoder_output = rhvae.vae.decoder(z)

    # Update momentum variable with half step. This step peforms fixed-point
    # iterations
    ρ̃ = _leapfrog_first_step(
        x, z, ρ, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs
    )

    # Update position variable with full step. This step peforms fixed-point
    # iterations
    z̄ = _leapfrog_second_step(
        x, z, ρ̃, G⁻¹, logdetG, rhvae.vae.decoder, decoder_output;
        ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs
    )

    # Update decoder output
    decoder_output_z̄ = rhvae.vae.decoder(z̄)

    # Update Riemannian metric tensor
    Ḡ⁻¹ = G_inv(z̄, rhvae)

    # Update the log determinant of the metric tensor
    logdetḠ = -slogdet(Ḡ⁻¹)

    # Update momentum variable with half step. No fixed-point iterations needed
    ρ̄ = _leapfrog_third_step(
        x, z̄, ρ̃, Ḡ⁻¹, logdetḠ, rhvae.vae.decoder, decoder_output_z̄;
        ϵ=ϵ, ∇H_kwargs=∇H_kwargs
    )

    return z̄, ρ̄, Ḡ⁻¹, logdetḠ, decoder_output_z̄
end # function

# ==============================================================================
# Combining Leapfrog and Tempering Steps
# ==============================================================================

@doc raw"""
    general_leapfrog_tempering_step(
        x::AbstractArray,
        zₒ::AbstractVecOrMat,
        Gₒ⁻¹::AbstractArray,
        logdetGₒ::Union{<:Number,AbstractVector},
        decoder::AbstractVariationalDecoder,
        decoder_output::NamedTuple,
        metric_param::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
            G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
    )

Combines the leapfrog and tempering steps into a single function for the
Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractArray`: The data to be processed. If `Array`, the last dimension
  must be of size 1.
- `zₒ::AbstractVector`: The initial latent variable. 
- `Gₒ⁻¹::AbstractArray`: The initial inverse of the Riemannian metric tensor.
- `logdetGₒ::Union{<:Number,AbstractVector}`: The log determinant of the initial
  Riemannian metric tensor. If vector, each element represents the log
  determinant of the metric tensor at the corresponding column of `zₒ`.
- `decoder::AbstractVariationalDecoder`: The decoder of the RHVAE model.
- `decoder_output::NamedTuple`: The output of the decoder.
- `metric_param::NamedTuple`: The parameters of the metric tensor.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog steps in
  the HMC algorithm. This can be a scalar or an array. Default is 0.01f0.  
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm. Default is 3.
- `βₒ::Number`: The initial inverse temperature for the tempering schedule.
  Default is 0.3f0.
- `steps::Int`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Default is a NamedTuple with
  `reconstruction_loglikelihood`, `position_logprior`, and `momentum_logprior`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys: 
    - `z_init`: The initial latent variable. 
    - `ρ_init`: The initial momentum variable. 
    - `Ginv_init`: The initial inverse of the Riemannian metric tensor. 
    - `logdetG_init`: The initial log determinant of the Riemannian metric
      tensor.
    - `z_final`: The final latent variable after `K` leapfrog steps. 
    - `ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `Ginv_final`: The final inverse of the Riemannian metric tensor after `K`
      leapfrog steps.
    - `logdetG_final`: The final log determinant of the Riemannian metric tensor
      after `K` leapfrog steps.
- The decoder output at the final latent variable is also returned. Note: This
  is not in the same named tuple as the other outputs, but as a separate output.

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the RHVAE model.
"""
function general_leapfrog_tempering_step(
    x::AbstractArray,
    zₒ::AbstractVecOrMat,
    Gₒ⁻¹::AbstractArray,
    logdetGₒ::Union{<:Number,AbstractVector},
    decoder::AbstractVariationalDecoder,
    decoder_output::NamedTuple,
    metric_param::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
)
    # Sample γₒ ~ N(0, Gₒ⁻¹). 
    γₒ = ignore_derivatives() do
        sample_MvNormalCanon(Gₒ⁻¹)
    end

    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value before loop
    zₖ₋₁ = deepcopy(zₒ)
    ρₖ₋₁ = deepcopy(ρₒ)
    Gₖ₋₁⁻¹ = deepcopy(Gₒ⁻¹)
    logdetGₖ₋₁ = deepcopy(logdetGₒ)
    decoderₖ₋₁ = deepcopy(decoder_output)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ, Gₖ⁻¹, logdetGₖ, decoderₖ = general_leapfrog_step(
            x, zₖ₋₁, ρₖ₋₁, Gₖ₋₁⁻¹, logdetGₖ₋₁, decoder, decoderₖ₋₁, metric_param;
            ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs, G_inv=G_inv
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering Update zₖ₋₁, ρₖ₋₁ for next
        # iteration. The momentum variable is updated with tempering. Also, note
        # this is the last step as well, thus we return zₖ₋₁, ρₖ₋₁ as the final
        # points.
        zₖ₋₁ = zₖ
        ρₖ₋₁ = ρₖ .* √(βₖ₋₁ / βₖ)
        # Update Gₖ₋₁⁻¹ for next iteration
        Gₖ₋₁⁻¹ = Gₖ⁻¹
        # Update logdetGₖ₋₁ for next iteration
        logdetGₖ₋₁ = logdetGₖ
        # Update decoderₖ₋₁ for next iteration
        decoderₖ₋₁ = decoderₖ
    end # for

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ₋₁,
        ρ_final=ρₖ₋₁,
        Ginv_init=Gₒ⁻¹,
        Ginv_final=Gₖ₋₁⁻¹,
        logdetG_init=logdetGₒ,
        logdetG_final=logdetGₖ₋₁,
    ), decoderₖ₋₁
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    general_leapfrog_tempering_step(
        x::AbstractArray,
        zₒ::AbstractVecOrMat,
        rhvae::RHVAE;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
        tempering_schedule::Function=quadratic_tempering,
    )

Combines the leapfrog and tempering steps into a single function for the
Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `x::AbstractArray`: The data to be processed. If `Array`, the last dimension
  must be of size 1.
- `zₒ::AbstractVecOrMat`: The initial latent variable. 

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog steps in
  the HMC algorithm. This can be a scalar or an array. Default is 0.01f0.  
- `K::Int`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) algorithm. Default is 3.
- `βₒ::Number`: The initial inverse temperature for the tempering schedule.
  Default is 0.3f0.
- `steps::Int`: The number of fixed-point iterations to perform. Default is 3.
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Default is a NamedTuple with
  `reconstruction_loglikelihood`, `position_logprior`, and `momentum_logprior`.
- `tempering_schedule::Function`: The function to compute the inverse
  temperature at each step in the HMC algorithm. Defaults to
  `quadratic_tempering`. This function must take three arguments: First, `βₒ`,
  an initial inverse temperature, second, `k`, the current step in the tempering
  schedule, and third, `K`, the total number of steps in the tempering schedule.

# Returns
- A `NamedTuple` with the following keys: 
    - `z_init`: The initial latent variable. 
    - `ρ_init`: The initial momentum variable. 
    - `Ginv_init`: The initial inverse of the Riemannian metric tensor. 
    - `z_final`: The final latent variable after `K` leapfrog steps. 
    - `ρ_final`: The final momentum variable after `K` leapfrog steps. 
    - `Ginv_final`: The final inverse of the Riemannian metric tensor after `K`
      leapfrog steps.
- The decoder output at the final latent variable is also returned. Note: This
  is not in the same named tuple as the other outputs, but as a separate output.

# Description
The function first samples a random momentum variable `γₒ` from a standard
normal distribution and scales it by the inverse square root of the initial
inverse temperature `βₒ` to obtain the initial momentum variable `ρₒ`. Then, it
performs `K` leapfrog steps, each followed by a tempering step, to generate a
new sample from the latent space.

# Note
Ensure the input data `x` and the initial latent variable `zₒ` match the
expected input dimensionality for the RHVAE model.
"""
function general_leapfrog_tempering_step(
    x::AbstractArray,
    zₒ::AbstractVecOrMat,
    rhvae::RHVAE;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
)
    # Compute metric param
    metric_param = update_metric(rhvae)
    # Compute inverse metric for initial point
    Gₒ⁻¹ = G_inv(zₒ, rhvae)
    # Compute log determinant of the metric tensor for initial point
    logdetGₒ = -slogdet(Gₒ⁻¹)

    # Sample γₒ ~ N(0, Gₒ⁻¹).
    γₒ = ignore_derivatives() do
        sample_MvNormalCanon(Gₒ⁻¹)
    end

    # Define ρₒ = γₒ / √βₒ
    ρₒ = γₒ ./ √(βₒ)

    # Define initial value before loop
    zₖ₋₁ = deepcopy(zₒ)
    ρₖ₋₁ = deepcopy(ρₒ)
    Gₖ₋₁⁻¹ = deepcopy(Gₒ⁻¹)
    logdetGₖ₋₁ = deepcopy(logdetGₒ)
    decoderₖ₋₁ = rhvae.vae.decoder(zₒ)

    # Loop over K steps
    for k = 1:K
        # 1) Leapfrog step
        zₖ, ρₖ, Gₖ⁻¹, logdetGₖ, decoderₖ = general_leapfrog_step(
            x, zₖ₋₁, ρₖ₋₁, Gₖ₋₁⁻¹, logdetGₖ₋₁,
            rhvae.vae.decoder, decoderₖ₋₁, metric_param;
            ϵ=ϵ, steps=steps, ∇H_kwargs=∇H_kwargs, G_inv=G_inv
        )

        # 2) Tempering step
        # Compute previous step's inverse temperature
        βₖ₋₁ = tempering_schedule(βₒ, k - 1, K)
        # Compute current step's inverse temperature
        βₖ = tempering_schedule(βₒ, k, K)

        # Update momentum variable with tempering Update zₖ₋₁, ρₖ₋₁ for next
        # iteration. The momentum variable is updated with tempering. Also, note
        # this is the last step as well, thus we return zₖ₋₁, ρₖ₋₁ as the final
        # points.
        zₖ₋₁ = zₖ
        ρₖ₋₁ = ρₖ .* √(βₖ₋₁ / βₖ)
        # Update Gₖ₋₁⁻¹ for next iteration
        Gₖ₋₁⁻¹ = Gₖ⁻¹
        # Update logdetGₖ₋₁ for next iteration
        logdetGₖ₋₁ = logdetGₖ
        # Update decoderₖ₋₁ for next iteration
        decoderₖ₋₁ = decoderₖ
    end # for

    return (
        z_init=zₒ,
        ρ_init=ρₒ,
        z_final=zₖ₋₁,
        ρ_final=ρₖ₋₁,
        Ginv_init=Gₒ⁻¹,
        Ginv_final=Gₖ₋₁⁻¹,
        logdetG_init=logdetGₒ,
        logdetG_final=logdetGₖ₋₁,
    ), decoderₖ₋₁
end # function

# ==============================================================================
# Forward pass methods for RHVAE with Generalized Hamiltonian steps
# ==============================================================================

@doc raw"""
    (rhvae::RHVAE{VAE{AbstractGaussianLogEncoder,D}})(
        x::AbstractArray,
        metric_param::NamedTuple;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                        reconstruction_loglikelihood=decoder_loglikelihood,
                        position_logprior=spherical_logprior,
                        momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
        tempering_schedule::Function=quadratic_tempering,
        latent::Bool=false,
    ) where {D<:AbstractVariationalDecoder}

Run the Riemannian Hamiltonian Variational Autoencoder (RHVAE) on the given
input. This method takes the parameters to compute the metric tensor as a
separate input in the form of a NamedTuple.

# Arguments
- `x::AbstractArray`: The input to the RHVAE. If `Vector`, it represents a
  single data point. If `Array`, the last dimension must contain each of the
  data points.
- `metric_param::NamedTuple`: The parameters used to compute the metric tensor.

# Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The step size for the leapfrog
  steps in the HMC part of the RHVAE. If it is a scalar, the same step size is
  used for all dimensions. If it is an array, each element corresponds to the
  step size for a specific dimension.
- `K::Int=3`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) part of the RHVAE.
- `βₒ::Number=0.3f0`: The initial inverse temperature for the tempering
  schedule.
- `steps::Int=3`: The number of fixed-point iterations to perform.  
- `∇H::Function=∇hamiltonian_finite`: The function to compute the gradient of
  the Hamiltonian in the HMC part of the RHVAE.
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Default is a NamedTuple with
  `reconstruction_loglikelihood`, `position_logprior`, and `momentum_logprior`.  
- `G_inv::Function=G_inv`: The function to compute the inverse of the Riemannian
  metric tensor.
- `tempering_schedule::Function=quadratic_tempering`: The function to compute
  the tempering schedule in the RHVAE.
- `latent::Bool=false`: If `true`, the function returns a NamedTuple containing
  the outputs of the encoder and decoder, and the final state of the phase space
  after the leapfrog and tempering steps. If `false`, the function only returns
  the output of the decoder.

# Returns
If `latent=true`, the function returns a NamedTuple with the following fields:
- `encoder`: The outputs of the encoder.
- `decoder`: The output of the decoder.
- `phase_space`: The final state of the phase space after the leapfrog and
  tempering steps.

If `latent=false`, the function only returns the output of the decoder.

# Description
This function runs the RHVAE on the given input. It first passes the input
through the encoder to obtain the mean and log standard deviation of the latent
space. It then uses the reparameterization trick to sample from the latent
space. After that, it performs the leapfrog and tempering steps to refine the
sample from the latent space. Finally, it passes the refined sample through the
decoder to obtain the output.

# Notes
Ensure that the dimensions of `x` match the input dimensions of the RHVAE, and
that the dimensions of `ϵ` match the dimensions of the latent space.
"""
function (rhvae::RHVAE{VAE{E,D}})(
    x::AbstractArray,
    metric_param::NamedTuple;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    latent::Bool=false,
) where {E<:AbstractGaussianLogEncoder,D<:AbstractVariationalDecoder}
    # Run input through encoder
    encoder_output = rhvae.vae.encoder(x)

    # Run reparametrize trick to generate latent variable zₒ
    zₒ = reparameterize(rhvae.vae.encoder, encoder_output)

    # Initial decoder output
    decoder_output = rhvae.vae.decoder(zₒ)

    # Initial inverse metric tensor
    Gₒ⁻¹ = G_inv(zₒ, metric_param)

    # Initial log determinant of the metric tensor
    logdetGₒ = -slogdet(Gₒ⁻¹)

    # Run leapfrog and tempering steps
    phase_space, decoder_update = general_leapfrog_tempering_step(
        x, zₒ, Gₒ⁻¹, logdetGₒ, rhvae.vae.decoder, decoder_output, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        G_inv=G_inv, tempering_schedule=tempering_schedule
    )

    # Check if latent variables should be returned
    if latent
        return (
            encoder=encoder_output,
            decoder=decoder_update,
            phase_space=phase_space,
        )
    else
        return decoder_update
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    (rhvae::RHVAE{VAE{E,D}})(
        x::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        ∇H::Function=∇hamiltonian_TaylorDiff,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
                reconstruction_loglikelihood=decoder_loglikelihood,
                position_logprior=spherical_logprior,
                momentum_logprior=riemannian_logprior,
                G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
        latent::Bool=false,
    ) where where {E<:AbstractGaussianLogEncoder,D<:AbstractVariationalDecoder}

Run the Riemannian Hamiltonian Variational Autoencoder (RHVAE) on the given
input.

# Arguments
- `x::AbstractArray`: The input to the RHVAE. If it is a vector, it represents a
  single data point. If `Array,` the last dimension must contain each of the
  data points.

# Optional Keyword Arguments
- `K::Int=3`: The number of leapfrog steps to perform in the Hamiltonian Monte
  Carlo (HMC) part of the RHVAE.
- `ϵ::Union{<:Number,<:AbstractVector}=0.01f0`: The step size for the leapfrog
  steps in the HMC part of the RHVAE. If it is a scalar, the same step size is
  used for all dimensions. If it is an array, each element corresponds to the
  step size for a specific dimension.
- `βₒ::Number=0.3f0`: The initial inverse temperature for the tempering
  schedule.
- `steps::Int`: The number of fixed-point iterations to perform. Default is 3.
- `∇H::Function=∇hamiltonian_finite`: The function to compute the gradient of
  the Hamiltonian in the HMC part of the RHVAE.
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Default is a NamedTuple with
  `reconstruction_loglikelihood`, `position_logprior`, and `momentum_logprior`.  
- `G_inv::Function=G_inv`: The function to compute the inverse of the Riemannian
  metric tensor.
- `tempering_schedule::Function=quadratic_tempering`: The function to compute
  the tempering schedule in the RHVAE.
- `latent::Bool=false`: If `true`, the function returns a NamedTuple containing
  the outputs of the encoder and decoder, and the final state of the phase space
  after the leapfrog and tempering steps. If `false`, the function only returns
  the output of the decoder.

# Returns
If `latent=true`, the function returns a NamedTuple with the following fields:
- `encoder`: The outputs of the encoder.
- `decoder`: The output of the decoder.
- `phase_space`: The final state of the phase space after the leapfrog and
    tempering steps.

If `latent=false`, the function only returns the output of the decoder.

# Description
This function runs the RHVAE on the given input. It first passes the input
through the encoder to obtain the mean and log standard deviation of the latent
space. It then uses the reparameterization trick to sample from the latent
space. After that, it performs the leapfrog and tempering steps to refine the
sample from the latent space. Finally, it passes the refined sample through the
decoder to obtain the output.

# Notes
Ensure that the dimensions of `x` match the input dimensions of the RHVAE, and
that the dimensions of `ϵ` match the dimensions of the latent space.
"""
function (rhvae::RHVAE{VAE{E,D}})(
    x::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    latent::Bool=false,
) where {E<:AbstractGaussianLogEncoder,D<:AbstractVariationalDecoder}
    # Compute metric_param
    metric_param = update_metric(rhvae)

    # Run input through encoder
    encoder_output = rhvae.vae.encoder(x)

    # Run reparametrize trick to generate latent variable zₒ
    zₒ = reparameterize(rhvae.vae.encoder, encoder_output)

    # Initial decoder output
    decoder_output = rhvae.vae.decoder(zₒ)

    # Initial inverse metric tensor
    Gₒ⁻¹ = G_inv(zₒ, metric_param)

    # Initial log determinant of the metric tensor
    logdetGₒ = -slogdet(Gₒ⁻¹)

    # Run leapfrog and tempering steps
    phase_space, decoder_update = general_leapfrog_tempering_step(
        x, zₒ, Gₒ⁻¹, logdetGₒ,
        rhvae.vae.decoder, decoder_output, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        G_inv=G_inv, tempering_schedule=tempering_schedule
    )

    # Check if latent variables should be returned
    if latent
        return (
            encoder=encoder_output,
            decoder=decoder_update,
            phase_space=phase_space,
        )
    else
        return decoder_update
    end # if
end # function

# ==============================================================================
# Riemannian Hamiltonian ELBO
# ==============================================================================

@doc raw"""
    _log_p̄(
        x::AbstractArray,
        rhvae::RHVAE{VAE{E,D}},
        rhvae_outputs::NamedTuple;
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        position_logprior::Function=spherical_logprior,
        momentum_logprior::Function=riemannian_logprior,
        prefactor::AbstractArray=ones(Float32, 3),
    )

This is an internal function used in `riemannian_hamiltonian_elbo` to compute
the numerator of the unbiased estimator of the marginal likelihood. The function
computes the sum of the log likelihood of the data given the latent variables,
the log prior of the latent variables, and the log prior of the momentum
variables.

    log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ(zₖ))

# Arguments
- `x::AbstractArray`: The input data. If `Array`, the last dimension must
  contain each of the data points.
- `rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractGaussianLogDecoder}}`:
  The Riemannian Hamiltonian Variational Autoencoder (RHVAE) model.
- `rhvae_outputs::NamedTuple`: The outputs of the RHVAE, including the final
  latent variables `zₖ` and the final momentum variables `ρₖ`.

# Optional Keyword Arguments
- `reconstruction_loglikelihood::Function`: The function to compute the log
  likelihood of the data given the latent variables. Default is
  `decoder_loglikelihood`.
- `position_logprior::Function`: The function to compute the log prior of the
  latent variables. Default is `spherical_logprior`.
- `momentum_logprior::Function`: The function to compute the log prior of the
  momentum variables. Default is `riemannian_logprior`.
- `prefactor::AbstractArray`: A 3-element array to scale the log likelihood,
  log prior of the latent variables, and log prior of the momentum variables.
  Default is an array of ones.

# Returns
- `log_p̄::AbstractVector`: The first term of the log of the unbiased estimator
  of the marginal likelihood for each data point.

# Note
This is an internal function and should not be called directly. It is used as
part of the `riemannian_hamiltonian_elbo` function.
"""
function _log_p̄(
    x::AbstractArray,
    rhvae::RHVAE,
    rhvae_outputs::NamedTuple;
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    position_logprior::Function=spherical_logprior,
    momentum_logprior::Function=riemannian_logprior,
    prefactor::AbstractArray=ones(Float32, 3),
)
    # Check that prefactor is the correct size
    if length(prefactor) != 3
        throw(ArgumentError("prefactor must be a 3-element array"))
    end # if

    # log p̄ = log p(x | zₖ) + log p(ρₖ | zₖ) + log p(zₖ) 

    # Compute log p(x | zₖ)
    log_p_x_given_zₖ = reconstruction_loglikelihood(
        x,
        rhvae_outputs.phase_space.z_final,
        rhvae.vae.decoder,
        rhvae_outputs.decoder
    ) .* prefactor[1]

    # Compute log p(ρₖ | zₖ)
    log_p_ρₖ_given_zₖ = momentum_logprior(
        rhvae_outputs.phase_space.ρ_final,
        rhvae_outputs.phase_space.Ginv_final,
        rhvae_outputs.phase_space.logdetG_final,
    ) .* prefactor[2]

    # Compute log p(zₖ)
    log_p_zₖ = position_logprior(
        rhvae_outputs.phase_space.z_final
    ) .* prefactor[3]


    return log_p_x_given_zₖ + log_p_ρₖ_given_zₖ + log_p_zₖ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    _log_q̄(
        rhvae::RHVAE,
        rhvae_outputs::NamedTuple,
        βₒ::Number;
        momentum_logprior::Function=riemannian_logprior,
        prefactor::AbstractArray=ones(Float32, 3),
    )

This is an internal function used in `riemannian_hamiltonian_elbo` to compute
the second term of the unbiased estimator of the marginal likelihood. The
function computes the sum of the log posterior of the initial latent variables
and the log prior of the initial momentum variables, minus a term that depends
on the dimensionality of the latent space and the initial temperature.

        log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)

# Arguments
- `rhvae::RHVAE`: The Riemannian Hamiltonian Variational Autoencoder (RHVAE)
  model.
- `rhvae_outputs::NamedTuple`: The outputs of the RHVAE, including the initial
  latent variables `zₒ` and the initial momentum variables `ρₒ`.
- `βₒ::Number`: The initial temperature for the tempering steps.

# Optional Keyword Arguments
- `momentum_logprior::Function`: The function to compute the log prior of the
  momentum variables. Default is `riemannian_logprior`.
- `prefactor::AbstractArray`: A 3-element array to scale the log posterior of
  the initial latent variables, log prior of the initial momentum variables, and
  the tempering Jacobian term. Default is an array of ones.

# Returns
- `log_q̄::Vector`: The second term of the log of the unbiased estimator of the
    marginal likelihood for each data point.

# Note
This is an internal function and should not be called directly. It is used as
part of the `riemannian_hamiltonian_elbo` function.
"""
function _log_q̄(
    rhvae::RHVAE,
    rhvae_outputs::NamedTuple,
    βₒ::Number;
    momentum_logprior::Function=riemannian_logprior,
    prefactor::AbstractArray=ones(Float32, 3),
)
    # Check that prefactor is the correct size
    if length(prefactor) != 3
        throw(ArgumentError("prefactor must be a 2-element array"))
    end # if
    # log q̄ = log q(zₒ | x) + log p(ρₒ | zₒ) - d/2 log(βₒ)

    # Compute log q(zₒ | x).
    log_q_zₒ_given_x = encoder_logposterior(
        rhvae_outputs.phase_space.z_init,
        rhvae.vae.encoder,
        rhvae_outputs.encoder
    ) .* prefactor[1]

    # Compute log p(ρₒ|zₒ)
    log_p_ρₒ_given_zₒ = momentum_logprior(
        rhvae_outputs.phase_space.ρ_init,
        rhvae_outputs.phase_space.Ginv_init,
        rhvae_outputs.phase_space.logdetG_init,
    ) .* prefactor[2]

    # Compute tempering Jacobian term
    tempering_jacobian = prefactor[3] * 0.5f0 *
                         size(rhvae_outputs.phase_space.z_init, 1) * log(βₒ)

    return log_q_zₒ_given_x + log_p_ρₒ_given_zₒ .- tempering_jacobian
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    riemannian_hamiltonian_elbo(
        rhvae::RHVAE,
        metric_param::NamedTuple,
        x::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
            G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Riemannian Hamiltonian Monte Carlo (RHMC) estimate of the evidence
lower bound (ELBO) for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

This function takes as input an RHVAE, a NamedTuple of metric parameters, and a
vector of input data `x`. It performs `K` RHMC steps with a leapfrog integrator
and a tempering schedule to estimate the ELBO. The ELBO is computed as the
difference between the `log p̄` and `log q̄` as

elbo = mean(log p̄ - log q̄),

# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `metric_param::NamedTuple`: The parameters used to compute the metric tensor.
- `x::AbstractArray`: The input data. If `Array`, the last dimension must
  contain each of the data points.

## Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.01).
- `K::Int`: The number of RHMC steps (default is 3).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of leapfrog steps (default is 3).
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Defaults to a NamedTuple with
  `:decoder_loglikelihood` set to `decoder_loglikelihood`, `:position_logprior`
  set to `spherical_logprior`, and `:momentum_logprior` set to
  `riemannian_logprior`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Defaults to `G_inv`.
- `tempering_schedule::Function`: The tempering schedule function used in the
  RHMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the RHVAE. Defaults
  to `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The RHMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the RHVAE.
"""
function riemannian_hamiltonian_elbo(
    rhvae::RHVAE,
    metric_param::NamedTuple,
    x::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Forward Pass (run input through reconstruct function)
    rhvae_outputs = rhvae(
        x, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        tempering_schedule=tempering_schedule,
        G_inv=G_inv,
        latent=true
    )
    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x, rhvae, rhvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇H_kwargs.reconstruction_loglikelihood,
        position_logprior=∇H_kwargs.position_logprior,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        rhvae, rhvae_outputs, βₒ;
        prefactor=logq_prefactor,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), rhvae_outputs
    else
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    riemannian_hamiltonian_elbo(
        rhvae::RHVAE,
        x::AbstractVector;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
            G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Riemannian Hamiltonian Monte Carlo (RHMC) estimate of the evidence
lower bound (ELBO) for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

This function takes as input an RHVAE, a NamedTuple of metric parameters, and a
vector of input data `x`. It performs `K` RHMC steps with a leapfrog integrator
and a tempering schedule to estimate the ELBO. The ELBO is computed as the
difference between the `log p̄` and `log q̄` as
    
elbo = mean(log p̄ - log q̄)
    
# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `x::AbstractVector`: The input data.

## Optional Keyword Arguments
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Defaults to a NamedTuple with
  `:decoder_loglikelihood` set to `decoder_loglikelihood`, `:position_logprior`
  set to `spherical_logprior`, `:momentum_logprior` set to
  `riemannian_logprior`, and `:G_inv` set to `G_inv`.
- `K::Int`: The number of RHMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of leapfrog steps (default is 3).
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor (default is `G_inv`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  RHMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the RHVAE. Defaults
  to `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The RHMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the RHVAE.
"""
function riemannian_hamiltonian_elbo(
    rhvae::RHVAE,
    x::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Compute metric_param
    metric_param = update_metric(rhvae)
    # Forward Pass (run input through reconstruct function)
    rhvae_outputs = rhvae(
        x, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        tempering_schedule=tempering_schedule,
        G_inv=G_inv,
        latent=true
    )

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x, rhvae, rhvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇H_kwargs.reconstruction_loglikelihood,
        position_logprior=∇H_kwargs.position_logprior,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        rhvae, rhvae_outputs, βₒ;
        prefactor=logq_prefactor,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), rhvae_outputs
    else
        # Return ELBO normalized by number of samples
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    riemannian_hamiltonian_elbo(
        rhvae::RHVAE,
        metric_param::NamedTuple,
        x_in::AbstractArray,
        x_out::AbstractArray;
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        K::Int=3,
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
            G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Riemannian Hamiltonian Monte Carlo (RHMC) estimate of the evidence
lower bound (ELBO) for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

This function takes as input an RHVAE, a NamedTuple of metric parameters, and a
vector of input data `x`. It performs `K` RHMC steps with a leapfrog integrator
and a tempering schedule to estimate the ELBO. The ELBO is computed as the
difference between the `log p̄` and `log q̄` as

elbo = mean(log p̄ - log q̄),

# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `metric_param::NamedTuple`: The parameters used to compute the metric tensor.
- `x_in::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.

## Optional Keyword Arguments
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.01).
- `K::Int`: The number of RHMC steps (default is 3).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of leapfrog steps (default is 3).
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Defaults to a NamedTuple with
  `:decoder_loglikelihood` set to `decoder_loglikelihood`, `:position_logprior`
  set to `spherical_logprior`, and `:momentum_logprior` set to
  `riemannian_logprior`.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor. Defaults to `G_inv`.
- `tempering_schedule::Function`: The tempering schedule function used in the
  RHMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the RHVAE. Defaults
  to `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The RHMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the RHVAE.
"""
function riemannian_hamiltonian_elbo(
    rhvae::RHVAE,
    metric_param::NamedTuple,
    x_in::AbstractArray,
    x_out::AbstractArray;
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    K::Int=3,
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Forward Pass (run input through reconstruct function)
    rhvae_outputs = rhvae(
        x_in, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        tempering_schedule=tempering_schedule,
        G_inv=G_inv,
        latent=true
    )
    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x_out, rhvae, rhvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇H_kwargs.reconstruction_loglikelihood,
        position_logprior=∇H_kwargs.position_logprior,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        rhvae, rhvae_outputs, βₒ;
        prefactor=logq_prefactor,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), rhvae_outputs
    else
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    riemannian_hamiltonian_elbo(
        rhvae::RHVAE,
        x_in::AbstractArray,
        x_out::AbstractArray;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
            G_inv=G_inv,
        ),
        tempering_schedule::Function=quadratic_tempering,
        return_outputs::Bool=false,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the Riemannian Hamiltonian Monte Carlo (RHMC) estimate of the evidence
lower bound (ELBO) for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

This function takes as input an RHVAE, a NamedTuple of metric parameters, and a
vector of input data `x`. It performs `K` RHMC steps with a leapfrog integrator
and a tempering schedule to estimate the ELBO. The ELBO is computed as the
difference between the `log p̄` and `log q̄` as
    
elbo = mean(log p̄ - log q̄).
    
# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `x_in::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.

## Optional Keyword Arguments
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function. Defaults to a NamedTuple with
  `:decoder_loglikelihood` set to `decoder_loglikelihood`, `:position_logprior`
  set to `spherical_logprior`, `:momentum_logprior` set to
  `riemannian_logprior`, and `:G_inv` set to `G_inv`.
- `K::Int`: The number of RHMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of leapfrog steps (default is 3).
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor (default is `G_inv`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  RHMC (default is `quadratic_tempering`).
- `return_outputs::Bool`: Whether to return the outputs of the RHVAE. Defaults
  to `false`. NOTE: This is necessary to avoid computing the forward pass twice
  when computing the loss function with regularization.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- `elbo::Number`: The RHMC estimate of the ELBO. If `return_outputs` is `true`,
  also returns the outputs of the RHVAE.
"""
function riemannian_hamiltonian_elbo(
    rhvae::RHVAE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    return_outputs::Bool=false,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Compute metric_param
    metric_param = update_metric(rhvae)
    # Forward Pass (run input through reconstruct function)
    rhvae_outputs = rhvae(
        x_in, metric_param;
        K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
        ∇H_kwargs=∇H_kwargs,
        tempering_schedule=tempering_schedule,
        G_inv=G_inv,
        latent=true
    )

    # Compute log evidence estimate log π̂(x) = log p̄ - log q̄

    # log p̄ = log p(x, zₖ) + log p(ρₖ)
    # log p̄ = log p(x | zₖ) + log p(zₖ) + log p(ρₖ)
    log_p = _log_p̄(
        x_out, rhvae, rhvae_outputs;
        prefactor=logp_prefactor,
        reconstruction_loglikelihood=∇H_kwargs.reconstruction_loglikelihood,
        position_logprior=∇H_kwargs.position_logprior,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    # log q̄ = log q(zₒ) + log p(ρₒ) - d/2 log(βₒ)
    log_q = _log_q̄(
        rhvae, rhvae_outputs, βₒ;
        prefactor=logq_prefactor,
        momentum_logprior=∇H_kwargs.momentum_logprior
    )

    if return_outputs
        return StatsBase.mean(log_p - log_q), rhvae_outputs
    else
        # Return ELBO normalized by number of samples
        return StatsBase.mean(log_p - log_q)
    end # if
end # function

# ==============================================================================
# RHVAE Loss function
# ==============================================================================

@doc raw"""
    loss(
        rhvae::RHVAE,
        x::AbstractArray;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
        tempering_schedule::Function=quadratic_tempering,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::Number=1.0f0,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the loss for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `x::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of steps in the leapfrog integrator (default is 3).
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor (default is `G_inv`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. This
  function must take as input the VAE outputs and the keyword arguments provided
  in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Number=1.0f0`: The strength of the regularization term.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- The computed loss.
"""
function loss(
    rhvae::RHVAE,
    x::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Number=1.0f0,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Update metric so that we can backpropagate through it
    metric_param = update_metric(rhvae)

    # Check if there is regularization 
    if reg_function !== nothing
        # Compute ELBO
        elbo, rhvae_outputs = riemannian_hamiltonian_elbo(
            rhvae, metric_param, x;
            K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
            ∇H_kwargs=∇H_kwargs,
            G_inv=G_inv,
            tempering_schedule=tempering_schedule,
            return_outputs=true,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )

        # Compute regularization
        reg_term = reg_function(rhvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -riemannian_hamiltonian_elbo(
            rhvae, metric_param, x;
            K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
            ∇H_kwargs=∇H_kwargs,
            G_inv=G_inv,
            tempering_schedule=tempering_schedule,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )
    end # if
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    loss(
        rhvae::RHVAE,
        x_in::AbstractArray,
        x_out::AbstractArray;
        K::Int=3,
        ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
        βₒ::Number=0.3f0,
        steps::Int=3,
        ∇H_kwargs::Union{NamedTuple,Dict}=(
            reconstruction_loglikelihood=decoder_loglikelihood,
            position_logprior=spherical_logprior,
            momentum_logprior=riemannian_logprior,
        ),
        G_inv::Function=G_inv,
        tempering_schedule::Function=quadratic_tempering,
        reg_function::Union{Function,Nothing}=nothing,
        reg_kwargs::Union{NamedTuple,Dict}=Dict(),
        reg_strength::Number=1.0f0,
        logp_prefactor::AbstractArray=ones(Float32, 3),
        logq_prefactor::AbstractArray=ones(Float32, 3),
    )

Compute the loss for a Riemannian Hamiltonian Variational Autoencoder (RHVAE).

# Arguments
- `rhvae::RHVAE`: The RHVAE used to encode the input data and decode the latent
  space.
- `x_in::AbstractArray`: Input data to the RHVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.

## Optional Keyword Arguments
- `K::Int`: The number of HMC steps (default is 3).
- `ϵ::Union{<:Number,<:AbstractVector}`: The step size for the leapfrog
  integrator (default is 0.001).
- `βₒ::Number`: The initial inverse temperature (default is 0.3).
- `steps::Int`: The number of steps in the leapfrog integrator (default is 3).
- `∇H_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be passed
  to the `∇hamiltonian` function.
- `G_inv::Function`: The function to compute the inverse of the Riemannian
  metric tensor (default is `G_inv`).
- `tempering_schedule::Function`: The tempering schedule function used in the
  HMC (default is `quadratic_tempering`).
- `reg_function::Union{Function, Nothing}=nothing`: A function that computes the
  regularization term based on the VAE outputs. This function must take as input
  the VAE outputs and the keyword arguments provided in `reg_kwargs`.
- `reg_kwargs::Union{NamedTuple,Dict}=Dict()`: Keyword arguments to pass to the
  regularization function.
- `reg_strength::Number=1.0f0`: The strength of the regularization term.
- `logp_prefactor::AbstractArray`: A 3-element array to scale the log
  likelihood, log prior of the latent variables, and log prior of the momentum
  variables. Default is an array of ones.
- `logq_prefactor::AbstractArray`: A 3-element array to scale the log posterior
  of the initial latent variables, log prior of the initial momentum variables,
  and the tempering Jacobian term. Default is an array of ones.

# Returns
- The computed loss.
"""
function loss(
    rhvae::RHVAE,
    x_in::AbstractArray,
    x_out::AbstractArray;
    K::Int=3,
    ϵ::Union{<:Number,<:AbstractVector}=Float32(1E-4),
    βₒ::Number=0.3f0,
    steps::Int=3,
    ∇H_kwargs::Union{NamedTuple,Dict}=(
        reconstruction_loglikelihood=decoder_loglikelihood,
        position_logprior=spherical_logprior,
        momentum_logprior=riemannian_logprior,
    ),
    G_inv::Function=G_inv,
    tempering_schedule::Function=quadratic_tempering,
    reg_function::Union{Function,Nothing}=nothing,
    reg_kwargs::Union{NamedTuple,Dict}=Dict(),
    reg_strength::Number=1.0f0,
    logp_prefactor::AbstractArray=ones(Float32, 3),
    logq_prefactor::AbstractArray=ones(Float32, 3),
)
    # Update metric so that we can backpropagate through it
    metric_param = update_metric(rhvae)

    # Check if there is regularization 
    if reg_function !== nothing
        # Compute ELBO
        elbo, rhvae_outputs = riemannian_hamiltonian_elbo(
            rhvae, metric_param, x_in, x_out;
            K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
            ∇H_kwargs=∇H_kwargs,
            G_inv=G_inv,
            tempering_schedule=tempering_schedule,
            return_outputs=true,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )

        # Compute regularization
        reg_term = reg_function(rhvae_outputs; reg_kwargs...)

        return -elbo + reg_strength * reg_term
    else
        # Compute ELBO
        return -riemannian_hamiltonian_elbo(
            rhvae, metric_param, x_in, x_out;
            K=K, ϵ=ϵ, βₒ=βₒ, steps=steps,
            ∇H_kwargs=∇H_kwargs,
            G_inv=G_inv,
            tempering_schedule=tempering_schedule,
            logp_prefactor=logp_prefactor,
            logq_prefactor=logq_prefactor
        )
    end # if
end # function

# ==============================================================================
# train! function for RHVAEs
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
function train!(
    rhvae::RHVAE,
    x::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(rhvae) do rhvae_model
        loss_function(rhvae_model, x; loss_kwargs...)
    end # do block

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
function train!(
    rhvae::RHVAE,
    x_in::AbstractArray,
    x_out::AbstractArray,
    opt::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = Flux.withgradient(rhvae) do rhvae_model
        loss_function(rhvae_model, x_in, x_out; loss_kwargs...)
    end # do block

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