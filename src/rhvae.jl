# Import ML libraries
import Flux
import Zygote

# Import basic math
import Distances
import LinearAlgebra
import Random
import StatsBase
import Distributions

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder,
    Float32Array

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import Concrete VAE type
using ..AutoEncode: VAE

# Import functions from other modules
using ..VAEs: reparameterize
using ..utils: vec_to_ltri

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
                metric_neurons[i-1] => metric_neurons[i], metric_activation[i]; init=init
            )
        )
    end # for

    # Create the MLP
    mlp = Flux.Chain(mlp_layers...)

    # Create the diag and lower layers. These layers have a number of neurons equal to the number of entries in a lower triangular matrix in the latent space.  
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
    (m::MetricChain)(x::AbstractArray{Float32}; matrix::Bool=false)

Perform a forward pass through the MetricChain.

# Arguments
- `x::AbstractArray{Float32}`: The input data to be processed. This should be a
  Float32 array.
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
function (m::MetricChain)(x::AbstractArray{Float32}; matrix::Bool=false)
    # Compute the output of the MLP
    mlp_out = m.mlp(x)

    # Compute the diagonal elements of the lower-triangular matrix
    diag_out = m.diag(mlp_out)

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
- `centroids_data::Matrix{Float32}`: A matrix where each column represents a
  data point xᵢ from which the centroids cᵢ are computed by passing them through
  the encoder.
- `centroids_latent::Matrix{Float32}`: A matrix where each column represents a
  centroid cᵢ in the inverse metric computation.
- `L::Array{Float32, 3}`: A 3D array where each slice represents a L_ψᵢ.
- `M::Array{Float32, 3}`: A 3D array where each slice represents a L_ψᵢ L_ψᵢᵀ.
- `T::Float32`: The temperature parameter in the inverse metric computation.  
- `λ::Float32`: The regularization factor in the inverse metric computation.

# Constructor
The constructor for `RHVAE` takes the following arguments:
- `vae`: The underlying VAE.
- `metric_chain`: The `MetricChain` that computes the Riemannian metric.
- `centroids_data`: A matrix of data points used to compute the centroids.
- `T`: The temperature parameter.
- `λ`: The regularization factor.

It initializes `centroids_latent` as a zero matrix with the same number of
columns as `centroids_data` and number of rows equal to the dimensionality of
the latent space. `L` and `M` are initialized as 3D arrays of identity matrices,
with the third dimension equal to the number of columns in `centroids_data`.
"""
struct RHVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
    metric_chain::MetricChain
    centroids_data::Matrix{Float32}
    centroids_latent::Matrix{Float32}
    L::Array{Float32,3}
    M::Array{Float32,3}
    T::Float32
    λ::Float32

    # Define default constructor
    function RHVAE(vae, metric_chain, centroids_data, T, λ)
        # Extract dimensionality of latent space
        ldim = size(vae.encoder.µ.weight, 1)

        # Initialize centroids_latent
        centroids_latent = zeros(
            Float32, ldim, size(centroids_data, 2)
        )

        # Initialize L
        L = reduce(
            (x, y) -> cat(x, y, dims=3),
            [
                Matrix{Float32}(LinearAlgebra.I(ldim))
                for _ in axes(centroids_data, 2)
            ]
        )

        # Initialize M
        M = L

        # Initialize RHVAE
        new{typeof(vae)}(
            vae, metric_chain, centroids_data, centroids_latent, L, M, T, λ,
        )
    end # function
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor RHVAE (vae, metric,)

# ==============================================================================
# Riemannian Metric computations
# ==============================================================================

@doc raw"""
    update_metric!(
        rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}
    )

Update the `centroids_latent` and `L` fields of a `RHVAE` instance in place.

This function takes a `RHVAE` instance as input and modifies its
`centroids_latent` and `L` fields. The `centroids_latent` field is updated by
running the `centroids_data` through the encoder of the underlying VAE and
extracting the mean (µ) of the resulting Gaussian distribution. The `L` field is
updated by running each column of the `centroids_data` through the
`metric_chain` and concatenating the results along the third dimension.

# Arguments
- `rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}`:
  The `RHVAE` instance to be updated.

# Usage
```julia
update_metric!(rhvae)
```
# Notes

This function modifies the `RHVAE` instance in place, so it does not return
anything. The changes are made directly to the `centroids_latent` and `L` fields
of the input `RHVAE` instance.
"""
function update_metric!(
    rhvae::RHVAE{<:VAE{<:AbstractGaussianEncoder,<:AbstractVariationalDecoder}}
)
    # Extract centroids_data
    centroids_data = rhvae.centroids_data
    # Run centroids_data through encoder and update centroids_latent
    rhvae.centroids_latent .= rhvae.vae.encoder(centroids_data).µ
    # Run centroids_data through metric_chain and update L
    rhvae.L .= reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            rhvae.metric_chain(centroid, matrix=true)
            for centroid in eachcol(centroids_data)
        ]
    )
    # Update M by multiplying L by its transpose
    rhvae.M .= reduce(
        (x, y) -> cat(x, y, dims=3),
        [
            L * LinearAlgebra.transpose(L)
            for L in eachslice(rhvae.L, dims=3)
        ]
    )
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    G_inv(
        rhvae::RHVAE,
        z::AbstractVector{Float32},
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
- `rhvae::RHVAE`: The `RHVAE` instance.
- `z::AbstractVector{Float32}`: The point in the latent space.

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
    rhvae::RHVAE,
    z::AbstractVector{Float32},
)
    # Compute Squared Euclidean distance between z and each centroid
    distances = Distances.colwise(
        Distances.SqEuclidean(), z, rhvae.centroids_latent
    )

    # Compute L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²). Note: The reshape is necessary
    # to broadcast the elemnt-wise product with each slice of M.
    LLexp = rhvae.M .* reshape(exp.(-distances ./ rhvae.T^2), 1, 1, :)

    # Return the sum of the LLexp slices plus the regularization term
    return dropdims(sum(LLexp, dims=3); dims=3) +
           rhvae.λ * LinearAlgebra.I(size(z, 1))
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    G_inv(
        rhvae::RHVAE,
        z::AbstractMatrix{Float32},
    )

Compute the inverse of the metric tensor G for a given set of points in the
latent space.

This function takes a `RHVAE` instance and a matrix `z` where each column
represents a point in the latent space, and computes the inverse of the metric
tensor G at each point. The computation is based on the centroids and the
temperature of the `RHVAE` instance, as well as a regularization term. The
inverse metric is computed as follows:

G⁻¹(z) = ∑ᵢ₌₁ⁿ L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²) + λIₗ,

where L_ψᵢ is computed by the `MetricChain`, T is the temperature, λ is a
regularization factor, and each column of `centroids_latent` are the cᵢ.

# Arguments
- `rhvae::RHVAE`: The `RHVAE` instance.
- `z::AbstractMatrix{Float32}`: The matrix where each column represents a point
  in the latent space.

# Returns
A 3D array where each slice represents the inverse of the metric tensor G at
the corresponding point `z`.

# Notes

The computation involves the squared Euclidean distance between each column of
`z` and each centroid of the RHVAE instance, the exponential of the negative of
these distances divided by the square of the temperature, and a regularization
term proportional to the identity matrix. The result is a 4D array where each 3D
array is of the same size as the latent space.
"""
function G_inv(
    rhvae::RHVAE,
    z::AbstractMatrix{Float32},
)
    # Compute Squared Euclidean distance between z and each centroid. Note: we
    # broadcast Distances.colwise over each column of z.
    distances = reduce(
        hcat,
        Distances.colwise.(
            Ref(Distances.SqEuclidean()),
            eachcol(z),
            Ref(rhvae.centroids_latent)
        )
    )

    # Compute L_ψᵢ L_ψᵢᵀ exp(-‖z - cᵢ‖₂² / T²). Note: The reshape is necessary
    # to broadcast the elemnt-wise product with each slice of M. The reduce in
    # combination with cat is used to append the resulting 3D arrays along a 4th
    # dimension.
    LLexp = reduce(
        (x, y) -> cat(x, y, dims=4),
        [
            rhvae.M .* reshape(exp.(-d ./ rhvae.T^2), 1, 1, :)
            for d in eachcol(distances)
        ]
    )

    # Return the sum of the LLexp slices plus the regularization term
    return dropdims(sum(LLexp, dims=3); dims=3) .+
           rhvae.λ * LinearAlgebra.I(size(z, 1))
end # function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generalized Hamiltonian Dynamics
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
