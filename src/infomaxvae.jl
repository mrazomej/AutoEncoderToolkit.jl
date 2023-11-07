# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

##

# Import Abstract Types

using ..AutoEncode: AbstractVariationalAutoEncoder, AbstractVariationalEncoder,
    AbstractVariationalDecoder, JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder, VAE

using ..VAEs: reparameterize

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMax-VAE
# Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
# Information in Variational Autoencoders. in 2020 IEEE International Symposium
# on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
# doi:10.1109/ISIT44484.2020.9174424.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# ==============================================================================
# `InfoMaxVAE <: AbstractVariationalAutoEncoder`
# ==============================================================================

@doc raw"""
    `InfoMaxVAE <: AbstractVariationalAutoEncoder`

Structure encapsulating an InfoMax variational autoencoder (InfoMaxVAE), an
architecture designed to enhance the VAE framework by maximizing mutual
information between the inputs and the latent representations, as per the
methods described by Rezaabad and Vishwanath (2020).

The model aims to learn representations that preserve mutual information with
the input data, arguably capturing more meaningful factors of variation.

> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
Information in Variational Autoencoders. in 2020 IEEE International Symposium on
Information Theory (ISIT) 2729–2734 (IEEE, 2020).
doi:10.1109/ISIT44484.2020.9174424.

# Fields
- `vae::VAE`: The core variational autoencoder, consisting of an encoder that
  maps input data into a latent space representation, and a decoder that
  attempts to reconstruct the input from the latent representation.
- `mlp::Flux.Chain`: A multi-layer perceptron (MLP) that is used to compute the
  mutual information between the inputs and the latent representations. The MLP
  takes as input the latent variables and outputs a scalar representing the
  estimated mutual information.

# Usage
The `InfoMaxVAE` struct is utilized in a similar manner to a standard VAE, with
the added capability of mutual information maximization as part of the training
process. This may involve an additional loss term that considers the output of
the `mlp` network to encourage latent representations that are informative about
the input data.

# Example
```julia
# Assuming definitions for `encoder`, `decoder`, and `mlp` are provided:
info_max_vae = InfoMaxVAE(VAE(encoder, decoder), mlp)

# During training, one would maximize both the variational lower bound and the 
# mutual information estimate provided by `mlp`.
```
"""
mutable struct InfoMaxVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder},
    M<:Flux.Chain
} <: AbstractVariationalAutoEncoder
    vae::V
    mlp::M
end # struct


# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor InfoMaxVAE

# ==============================================================================
# `MLP <: Flux.Chain`
# ==============================================================================

"""
    MLP(
        n_input::Int,
        n_latent::Int,
        mlp_neurons::Vector{<:Int},
        mlp_activations::Vector{<:Function},
        output_activation::Function;
        init::Function = Flux.glorot_uniform
    ) -> Flux.Chain

Construct a multi-layer perceptron (MLP) using `Flux.jl`. The MLP is composed of
a sequence of dense layers, each followed by a specified activation function,
ending with a single neuron output layer with its own activation function.

# Arguments
- `n_input::Int`: Number of input features to the MLP.
- `n_latent::Int`: The dimensionality of the latent space.
- `mlp_neurons::Vector{<:Int}`: A vector of integers where each element
  represents the number of neurons in the corresponding hidden layer of the MLP.
- `mlp_activations::Vector{<:Function}`: A vector of activation functions to be
  used in the hidden layers. Length must match that of `mlp_neurons`.
- `output_activation::Function`: Activation function for the output neuron of
  the MLP.

# Optional Keyword Arguments
- `init::Function`: Initialization function for the weights of all layers in the
  MLP. Defaults to `Flux.glorot_uniform`.

# Returns
- `Flux.Chain`: A `Flux.Chain` object representing the constructed MLP.

# Example
```julia
mlp = initialize_MLP(
    256,
    [128, 64, 32],
    [relu, relu, relu],
    sigmoid
)
```
# Notes

The function will throw an error if the number of provided activation functions
does not match the number of layers specified in mlp_neurons.
"""
function MLP(
    n_input::Int,
    n_latent::Int,
    mlp_neurons::Vector{<:Int},
    mlp_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check if there's an activation function for each layer, plus the output layer
    if length(mlp_activation) != length(mlp_neurons)
        error("Each hidden layer needs exactly one activation function")
    end # if

    mlp_layers = []

    # Add first layer (input layer to first hidden layer)
    push!(
        mlp_layers,
        Flux.Dense(
            n_input + n_latent => mlp_neurons[1], mlp_activation[1]; init=init
        )
    )

    # Add hidden layers
    for i = 2:length(mlp_neurons)
        push!(
            mlp_layers,
            Flux.Dense(
                mlp_neurons[i-1] => mlp_neurons[i], mlp_activation[i]; init=init
            )
        )
    end # for

    # Add output layer
    push!(
        mlp_layers,
        Flux.Dense(mlp_neurons[end] => 1, output_activation; init=init)
    )

    return Flux.Chain(mlp_layers...)
end # function

@doc raw"""
    (vae::InfoMaxVAE)(
        x::AbstractVecOrMat{Float32}; 
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
        latent::Bool=false, 
        n_samples::Int=1
    ) 

Processes the input data `x` through an InfoMaxVAE, which consists of an
encoder, a decoder, and a multi-layer perceptron (MLP) to estimate variational
mutual information.

# Arguments
- `x::AbstractVecOrMat{Float32}`: The data to be processed. Can be a vector or a
  matrix where each column represents a separate data sample.

# Optional Keyword Arguments
- `prior::Distributions.Sampleable`: Specifies the prior distribution for the
  latent space during the reparametrization trick. Defaults to a standard normal
  distribution.
- `latent::Bool`: If `true`, returns a dictionary with latent variables and
  mutual information estimations along with the reconstruction. Defaults to
  `false`.
- `n_samples::Int=1`: The number of samples to draw from the latent distribution
  using the reparametrization trick.

# Returns
- If `latent=false`: `Array{Float32}`, the reconstructed data after processing
  through the encoder and decoder.
- If `latent=true`: A dictionary with keys `:encoder_µ`, `:encoder_(log)σ`,
  `:z`, `:decoder_µ`, `:decoder_(log)σ`, and `:mutual_info`, containing the
  corresponding values.

# Description
This function first encodes the input `x` to obtain the mean and log standard
deviation of the latent space. It then samples from this distribution using the
reparametrization trick. The sampled latent vectors are then decoded, and the
MLP estimates the variational mutual information.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the InfoMaxVAE.
"""
function (infomaxvae::InfoMaxVAE)(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
)
    # Check if latent variables and mutual information should be returned
    if latent
        outputs = infomaxvae.vae(x, prior; latent=latent, n_samples=n_samples)

        # Compute mutual information estimate using MLP
        outputs[:mutual_info] = infomaxvae.mlp([x; outputs[:z]])

        return outputs
    else
        # or return reconstructed data from decoder
        return infomaxvae.vae(x, prior; latent=latent, n_samples=n_samples)
    end # if
end # function

# ==============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE loss function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ==============================================================================

# ==============================================================================
# Loss InfoMax.VAE{JointLogEncoder,SimpleDecoder}
# ==============================================================================

@doc raw"""
    `loss(vae, mlp, x; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
    regularization=nothing, reg_strength=1.0f0)`

Computes the loss for an InfoMax variational autoencoder (VAE) with mutual
information constraints, by averaging over `n_samples` latent space samples.

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, the variational mutual information between input and latent
representations, and possibly a regularization term, defined as:

loss = -⟨log π(x|z)⟩ + β × Dₖₗ[qᵩ(z|x) || π(z)] - α × I(x;z) + reg_strength ×
reg_term

Where:
- `⟨log π(x|z)⟩` is the expected log likelihood of the probabilistic decoder. -
`Dₖₗ[qᵩ(z|x) || π(z)]` is the KL divergence between the approximated encoder
  and the prior over the latent space.
- `I(x;z)` is the variational mutual information between the inputs `x` and the
  latent variables `z`.
- `mlp` is a multi-layer perceptron estimating the mutual information I(x;z).

# Arguments
- `vae::VAE{JointLogEncoder,SimpleDecoder}`: A VAE model with encoder and
  decoder networks.
- `mlp::Flux.Chain`: A multi-layer perceptron used to compute the mutual
  information term.
- `x::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder π(x|z).
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `α::Float32=1.0f0`: Weighting factor for the mutual information term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the VAE outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed average loss value for the input `x` and its
  reconstructed counterparts over `n_samples` samples, including possible
  regularization terms and the mutual information constraint.

# Note
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function loss(
    vae::VAE{JointLogEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Extract batch size
    batch_size = size(x, 2)

    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * batch_size * n_samples) * sum((x .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

@doc raw"""
    `loss(vae, mlp, x_in, x_out; σ=1.0f0, β=1.0f0, α=1.0f0, 
          n_samples=1, regularization=nothing, reg_strength=1.0f0)`

Computes the loss for the variational autoencoder (VAE) by averaging over
`n_samples` latent space samples, considering both input `x_in` and output
`x_out`.

This function extends the loss computation to include an additional mutual
information term between the input `x_in` and latent representation `z`, and
between `x_out` and `z`. The loss function combines the reconstruction loss, the
Kullback-Leibler (KL) divergence, the mutual information term, and possibly a
regularization term, defined as:

loss = -⟨log π(x_out|z)⟩ + β × Dₖₗ[qᵩ(z|x_in) || π(z)] - α × I(x_in; z) +
reg_strength × reg_term

Where:
- π(x_out|z) is a probabilistic decoder: π(x_out|z) = N(f(z), σ² I̲̲)) - f(z) is
the function defining the mean of the decoder π(x_out|z) - qᵩ(z|x_in) is the
approximated encoder: qᵩ(z|x_in) = N(g(x_in), h(x_in))
- g(x_in) and h(x_in) define the mean and covariance of the encoder
  respectively.
- I(x_in; z) is the mutual information between `x_in` and the latent variable
  `z`, approximated by a neural network (mlp).

# Arguments
- `vae::VAE{JointLogEncoder,SimpleDecoder}`: A VAE model with encoder and
  decoder networks.
- `mlp::Flux.Chain`: A multilayer perceptron for approximating the mutual
  information between inputs and latent variables.
- `x_in::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation input to the encoder.
- `x_out::AbstractMatrix{Float32}`: Target output Matrix for the decoder.

# Optional Keyword Arguments
- `σ::Float32=1.0f0`: Standard deviation for the probabilistic decoder
  π(x_out|z).
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `α::Float32=1.0f0`: Weighting factor for the mutual information term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the VAE outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed average loss value for the input `x_in`, the target
  `x_out`, and their reconstructed counterparts over `n_samples` samples,
  including possible regularization terms.

# Note
- Ensure that the input data `x_in` and `x_out` match the expected input and
  output dimensionality for the encoder and decoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function loss(
    vae::VAE{JointLogEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Extract batch size
    batch_size = size(x_in, 2)

    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples * batch_size) * sum((x_out .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x_in; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x_in; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x_in); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x_in); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

# ==============================================================================
# Loss InfoMaxVAE.VAE{JointLogEncoder, Union{JointLogDecoder,SplitLogDecoder}}
# ==============================================================================

@doc raw"""
    `loss(vae, mlp, x; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
    regularization=nothing, reg_strength=1.0f0)`

Computes the loss for an InfoMax variational autoencoder (VAE) which integrates
mutual information constraints, by averaging over `n_samples` samples drawn from
the latent space.

The loss function is a composite of the reconstruction loss, the
Kullback-Leibler (KL) divergence, and the variational mutual information between
the input and latent representations. A regularization term may also be
included:

    loss = -⟨log π(x|z)⟩ + β × Dₖₗ[q(z|x) || p(z)] - α × I(x;z) 
            + reg_strength × reg_term

where:
- `⟨log π(x|z)⟩` is the expected log likelihood of the probabilistic decoder,
  estimating how well the reconstruction matches the original input.
- `Dₖₗ[q(z|x) || p(z)]` is the KL divergence, measuring how the variational
  posterior q(z|x) diverges from the prior p(z) over the latent variables.
- `I(x;z)` represents the variational mutual information, quantifying the amount
  of information shared between the inputs `x` and the latent variables `z`.
- `reg_term` is an optional regularization term.

## Arguments
- `vae::VAE{JointLogEncoder, <:Union{JointLogDecoder, SplitLogDecoder}}`: The
  VAE model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation.

## Optional Keyword Arguments
- `β::Float32=1.0f0`: The scaling factor for the KL divergence term.
- `α::Float32=1.0f0`: The scaling factor for the mutual information term.
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.
- `regularization::Union{Function, Nothing}=nothing`: An optional regularization
  function, should return a scalar cost component.
- `reg_strength::Float32=1.0f0`: Strength of the regularization term.

## Returns
- `Float32`: The total computed loss, averaged over `n_samples` and inclusive of
  the mutual information constraint and any regularization.

## Notes
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function loss(
    vae::VAE{JointLogEncoder,T},
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointLogDecoder,SplitLogDecoder}}
    # Extract batch size
    batch_size = size(x, 2)

    # Run input through reconstruct function with n_samples
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples * batch_size) * (
        length(decoder_µ) * log(2 * π) +
        2.0f0 * sum(decoder_logσ) +
        sum((x .- decoder_µ) .^ 2 ./ exp.(2 * decoder_logσ))
    )

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated encoder qᵩ(z|x_in)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

@doc raw"""
    `loss(vae, mlp, x_in, x_out; σ=1.0f0, β=1.0f0, α=1.0f0, 
    n_samples=1, regularization=nothing, reg_strength=1.0f0)`

Computes the loss for an InfoMax variational autoencoder (VAE) which integrates
mutual information constraints, by averaging over `n_samples` samples drawn from
the latent space.

The loss function is a composite of the reconstruction loss, the
Kullback-Leibler (KL) divergence, and the variational mutual information between
the input and latent representations. A regularization term may also be
included:

    loss = -⟨log π(x|z)⟩ + β × Dₖₗ[q(z|x) || p(z)] - α × I(x;z) 
            + reg_strength × reg_term

where:
- `⟨log π(x|z)⟩` is the expected log likelihood of the probabilistic decoder,
  estimating how well the reconstruction matches the original input.
- `Dₖₗ[q(z|x) || p(z)]` is the KL divergence, measuring how the variational
  posterior q(z|x) diverges from the prior p(z) over the latent variables.
- `I(x;z)` represents the variational mutual information, quantifying the amount
  of information shared between the inputs `x` and the latent variables `z`.
- `reg_term` is an optional regularization term.

## Arguments
- `vae::VAE{JointLogEncoder, <:Union{JointLogDecoder, SplitLogDecoder}}`: The
  VAE model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x_in::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation input to the encoder.
- `x_out::AbstractMatrix{Float32}`: Target output Matrix for the decoder.

## Optional Keyword Arguments
- `β::Float32=1.0f0`: The scaling factor for the KL divergence term.
- `α::Float32=1.0f0`: The scaling factor for the mutual information term.
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.
- `regularization::Union{Function, Nothing}=nothing`: An optional regularization
  function, should return a scalar cost component.
- `reg_strength::Float32=1.0f0`: Strength of the regularization term.

## Returns
- `Float32`: The total computed loss, averaged over `n_samples` and inclusive of
  the mutual information constraint and any regularization.

## Notes
- Ensure that the input data `x_in` and `x_out` match the expected input and
  output dimensionality for the encoder and decoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function loss(
    vae::VAE{JointLogEncoder,T},
    mlp::Flux.Chain,
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointLogDecoder,SplitLogDecoder}}
    # Extract batch size
    batch_size = size(x_in, 2)
    # Run input through reconstruct function with n_samples
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples * batch_size) * (
        length(decoder_µ) * log(2 * π) +
        2.0f0 * sum(decoder_logσ) +
        sum((x_out .- decoder_µ) .^ 2 ./ exp.(2 * decoder_logσ))
    )

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x_in; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x_in; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x_in); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x_in); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated encoder qᵩ(z|x_in)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

# ==============================================================================
# Loss InfoMaxVAE.VAE{JointLogEncoder, Union{JointDecoder,SplitDecoder}}
# ==============================================================================

@doc raw"""
    `loss(vae, mlp, x; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
    regularization=nothing, reg_strength=1.0f0)`

Computes the loss for an InfoMax variational autoencoder (VAE) which integrates
mutual information constraints, by averaging over `n_samples` samples drawn from
the latent space.

The loss function is a composite of the reconstruction loss, the
Kullback-Leibler (KL) divergence, and the variational mutual information between
the input and latent representations. A regularization term may also be
included:

    loss = -⟨log π(x|z)⟩ + β × Dₖₗ[q(z|x) || p(z)] - α × I(x;z) 
            + reg_strength × reg_term

where:
- `⟨log π(x|z)⟩` is the expected log likelihood of the probabilistic decoder,
  estimating how well the reconstruction matches the original input.
- `Dₖₗ[q(z|x) || p(z)]` is the KL divergence, measuring how the variational
  posterior q(z|x) diverges from the prior p(z) over the latent variables.
- `I(x;z)` represents the variational mutual information, quantifying the amount
  of information shared between the inputs `x` and the latent variables `z`.
- `reg_term` is an optional regularization term.

## Arguments
- `vae::VAE{JointLogEncoder, <:Union{JointDecoder, SplitDecoder}}`: The VAE
  model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation.

## Optional Keyword Arguments
- `β::Float32=1.0f0`: The scaling factor for the KL divergence term.
- `α::Float32=1.0f0`: The scaling factor for the mutual information term.
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.
- `regularization::Union{Function, Nothing}=nothing`: An optional regularization
  function, should return a scalar cost component.
- `reg_strength::Float32=1.0f0`: Strength of the regularization term.

## Returns
- `Float32`: The total computed loss, averaged over `n_samples` and inclusive of
  the mutual information constraint and any regularization.

## Notes
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Extract batch size
    batch_size = size(x, 2)

    # Run input through reconstruct function with n_samples
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_σ = outputs[:decoder_µ], outputs[:decoder_σ]

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples * batch_size) * (
        length(decoder_µ) * log(2 * π) +
        2.0f0 * sum(log.(decoder_σ)) +
        sum((x .- decoder_µ) .^ 2 ./ decoder_σ .^ 2)
    )

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated encoder
    # qᵩ(z|x_in) and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

@doc raw"""
    `loss(vae, mlp, x_in, x_out; σ=1.0f0, β=1.0f0, α=1.0f0, 
    n_samples=1, regularization=nothing, reg_strength=1.0f0)`

Computes the loss for an InfoMax variational autoencoder (VAE) which integrates
mutual information constraints, by averaging over `n_samples` samples drawn from
the latent space.

The loss function is a composite of the reconstruction loss, the
Kullback-Leibler (KL) divergence, and the variational mutual information between
the input and latent representations. A regularization term may also be
included:

    loss = -⟨log π(x|z)⟩ + β × Dₖₗ[q(z|x) || p(z)] - α × I(x;z) 
            + reg_strength × reg_term

where:
- `⟨log π(x|z)⟩` is the expected log likelihood of the probabilistic decoder,
  estimating how well the reconstruction matches the original input.
- `Dₖₗ[q(z|x) || p(z)]` is the KL divergence, measuring how the variational
  posterior q(z|x) diverges from the prior p(z) over the latent variables.
- `I(x;z)` represents the variational mutual information, quantifying the amount
  of information shared between the inputs `x` and the latent variables `z`.
- `reg_term` is an optional regularization term.

## Arguments
- `vae::VAE{JointLogEncoder, <:Union{JointDecoder, SplitDecoder}}`: The VAE
  model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x_in::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation input to the encoder.
- `x_out::AbstractMatrix{Float32}`: Target output Matrix for the decoder.

## Optional Keyword Arguments
- `β::Float32=1.0f0`: The scaling factor for the KL divergence term.
- `α::Float32=1.0f0`: The scaling factor for the mutual information term.
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.
- `regularization::Union{Function, Nothing}=nothing`: An optional regularization
  function, should return a scalar cost component.
- `reg_strength::Float32=1.0f0`: Strength of the regularization term.

## Returns
- `Float32`: The total computed loss, averaged over `n_samples` and inclusive of
  the mutual information constraint and any regularization.

## Notes
- Ensure that the input data `x_in` and `x_out` match the expected input
  dimensionality for the encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples)..
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Extract batch size
    batch_size = size(x_in, 2)
    # Run input through reconstruct function with n_samples
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_σ = outputs[:decoder_µ], outputs[:decoder_σ]

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples * batch_size) * (
        length(decoder_µ) * log(2 * π) +
        2.0f0 * sum(log.(decoder_σ)) +
        sum((x_out .- decoder_µ) .^ 2 ./ decoder_σ .^ 2)
    )

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x_in; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x_in; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x_in); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x_in); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute Kullback-Leibler divergence between approximated encoder
    # qᵩ(z|x_in) and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    ) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

# ==============================================================================

@doc raw"""
    mlp_loss(vae, mlp, x; n_samples=1, regularization=nothing, 
            reg_strength=1.0f0)

Calculates the loss for training the multi-layer perceptron (mlp) in the
InfoMaxVAE algorithm to estimate mutual information between the input `x` and
the latent representation `z`. The loss function is based on a variational
approximation of mutual information, using the MLP's output `g(x, z)`. The
variational mutual information is then calculated as the difference between the
MLP's output for the true `x` and latent `z`, and the exponentiated average of
the MLP's output for `x` and the shuffled latent `z_shuffle`, adjusted for the
regularization term if provided.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder, <:AbstractVariationalDecoder}`: The
  variational autoencoder.
- `mlp::Flux.Chain`: The multi-layer perceptron used for estimating mutual
  information.
- `x::AbstractMatrix{Float32}`: The input vector for the VAE.

# Optional Keyword Arguments
- `n_samples::Int=1`: The number of samples to draw from the latent
  distribution.
- `regularization::Union{Function, Nothing}=nothing`: A regularization function
  applied to the MLP's output.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.

# Returns
- `Float32`: The computed loss, representing negative variational mutual
  information, adjusted by the regularization term.

# Description
The function computes the loss as follows:

loss = -sum(I(x; z)) + sum(exp(I(x; z̃) - 1)) + reg_strength * reg_term

where `I(x; z)` is the MLP's output representing an estimation of mutual
information for true `x` and latent `z`, and `z̃` represents shuffled latent
variables, meaning, the latent codes are randomly swap between data points.

The function is used to separately train the MLP to estimate mutual information,
which is a component of the larger InfoMaxVAE model.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).

# Examples
```julia
# Assuming vae, mlp, x, and x_shuffle are predefined and properly formatted:
loss_value = mlp_loss(vae, mlp, x, x_shuffle)
```
"""
function mlp_loss(
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Extract batch size
    batch_size = size(x, 2)

    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract relevant variable
    z = outputs[:z]

    # Permute latent codes for computation of mutual information
    if n_samples == 1
        z_shuffle = @view z[:, Random.shuffle(1:end)]
    else
        z_shuffle = @view z[:, Random.shuffle(1:end), :]
    end # if

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = sum(mlp([x; z]))
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp([x; z_shuffle]))
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([Ref(x); eachslice(z, dims=3)])) / n_samples

        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([Ref(x); eachslice(z_shuffle, dims=3)])) /
                    n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1)) / batch_size

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute (negative) variational mutual information as loss function
    return -info_x_z + reg_strength * reg_term
end #function

# ==============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE training functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ==============================================================================

"""
    `train!(infomaxvae, x, opt_vae, opt_mlp; loss_function=loss, loss_kwargs,
            mlp_loss_function, mpl_loss_kwargs)`

Customized training function to update parameters of an InfoMax variational
autoencoder (VAE) given a loss function of the specified form.

The InfoMax VAE loss function can be defined as:

    loss_infoMax = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z) || π(z)) -
                   α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

This function simultaneously optimizes two neural networks: the VAE itself and a
multi-layer perceptron (MLP) used to compute the mutual information between
input and latent variables.

# Arguments
- `infomaxvae::InfoMaxVAE`: Struct containing the elements of an InfoMax VAE.
- `x::AbstractVector{Float32}`: Vector containing the data on which to evaluate
  the loss function. Each element should represent a single input.
- `opt_vae::Flux.Optimise.Optimiser`: Optimizing algorithm to be used for
  updating the VAE parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: Optimizing algorithm to be used for
  updating the MLP parameters.

# Optional Keyword arguments
- `loss_function::Function`: The loss function to be used during training,
  defaulting to `loss`.
- `loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
  passed to the loss function.
- `mlp_loss_function::Function`: The loss function to be used during training
  for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
  

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function to jointly
train the VAE and MLP. The VAE parameters are updated to minimize the InfoMaxVAE
loss, while the MLP parameters are updated to maximize the estimated mutual
information. The function allows for customization of loss hyperparameters
during training.

# Examples
```julia
opt_vae = Flux.Optimise.ADAM(1e-3)
opt_mlp = Flux.Optimise.ADAM(1e-3)

for x in dataloader
    train!(infomaxvae, x, opt_vae, opt_mlp, α=100.0)
end
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractVector{Float32},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Permute data for computation of mutual information
    x_shuffle = @view x[Random.shuffle(1:end)]

    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss_function(vae, infomaxvae.mlp, x, x_shuffle; loss_kwargs...)
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss_function(infomaxvae.vae, mlp, x, x_shuffle; mlp_loss_kwargs...)
    end # do

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
end # function

@doc raw"""
    `train!(infomaxvae, x, opt_vae, opt_mlp; loss_function=loss, loss_kwargs,
            mpl_loss_function, mpl_loss_kwargs, average=true)`

Trains an InfoMax Variational Autoencoder (VAE) on a batch of data represented
as a matrix, with optional averaging of gradients across data points.

# Arguments
- `infomaxvae::InfoMaxVAE`: The InfoMax VAE model to be trained.
- `x::AbstractMatrix{Float32}`: A matrix of training data where each column
  represents a single data sample.
- `opt_vae::Flux.Optimise.Optimiser`: The optimizer to use for the VAE
  parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: The optimizer to use for the MLP
  parameters.


# Optional Keyword Arguments
- `loss_function::Function`: The loss function to be used for training. Defaults
  to a predefined `loss` function.
- `loss_kwargs::Union{NamedTuple,Dict}`: Keyword arguments to pass to the loss
  function.
- `mlp_loss_function::Function`: The loss function to be used during training
for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
- `average::Bool`: A boolean flag to determine if gradients should be averaged
  across all data points. If `true`, gradients are averaged. If `false`,
  `train!` is called on each data point separately.

# Description
This function can either average the gradients across all the data points before
updating the parameters (`average=true`) or update the parameters for each data
point separately (`average=false`). When `average` is set to `true`, it computes
the gradients by averaging the loss over all columns of the input data matrix
`x` and applies the updates to the VAE and MLP parameters accordingly. When
`average` is set to `false`, it loops over each column and applies updates
individually.

# Examples
```julia
opt_vae = Flux.Optimise.ADAM(1e-3)
opt_mlp = Flux.Optimise.ADAM(1e-3)
x = rand(Float32, 784, 100) # 100 samples of 784-dimensional data

train!(infomaxvae, x, opt_vae, opt_mlp, average=false)
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractMatrix{Float32},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Permute data for computation of mutual information
    x_shuffle = @view x[:, Random.shuffle(1:end)]

    # Decide on training approach based on 'average'
    if average
        # == VAE == #
        # Compute gradient
        ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
            StatsBase.mean(
                loss_function.(
                    Ref(vae),
                    Ref(infomaxvae.mlp),
                    eachcol(x),
                    eachcol(x_shuffle);
                    loss_kwargs...
                )
            )
        end # do

        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

        # == MLP == #
        # Compute gradient
        ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
            StatsBase.mean(
                mlp_loss_function.(
                    Ref(infomaxvae.vae),
                    Ref(mlp),
                    eachcol(x),
                    eachcol(x_shuffle);
                    mlp_loss_kwargs...
                )
            )
        end # do

        # Update the MLP network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
    else
        foreach(
            col -> train!(
                infomaxvae, col, opt_vae, opt_mlp;
                loss_function=loss_function,
                loss_kwargs=loss_kwargs,
                mlp_loss_function=mlp_loss_function,
                mlp_loss_kwargs=mlp_loss_kwargs
            ),
            eachcol(x)
        )
    end # if
end # function

@doc raw"""
        train!(infomaxvae, x, opt_vae, opt_mlp; loss_function=loss, 
        loss_kwargs, mlp_loss_function=mlp_loss, mlp_loss_kwargs, 
        average=true)

Trains an InfoMax Variational Autoencoder (VAE) on a batch of 3D data, with
optional averaging of gradients across the tensor slices.

# Arguments

# Arguments
- `infomaxvae::InfoMaxVAE`: The InfoMax VAE model to be trained.
- `x::AbstractArray{Float32,3}`: A 3D tensor of training data where the first
  dimension corresponds to features, the second to time or sequence, and the
  third to different samples or batch items.
- `opt_vae::Flux.Optimise.Optimiser`: The optimizer to use for the VAE
  parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: The optimizer to use for the MLP
  parameters.


# Optional Keyword Arguments
- `loss_function::Function`: The loss function to be used for training. Defaults
  to a predefined `loss` function.
- `loss_kwargs::Union{NamedTuple,Dict}`: Keyword arguments to pass to the loss
  function.
- `mlp_loss_function::Function`: The loss function to be used during training
for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
- `average::Bool`: A boolean flag to determine if gradients should be averaged
  across all data points. If `true`, gradients are averaged. If `false`,
  `train!` is called on each data point separately.

# Description

This function allows training on 3D data by handling each slice along the third
dimension either in an averaged manner or individually. If `average=true`, the
function computes the gradients by averaging the loss across all slices of the
3D tensor `x` and updates the VAE and MLP parameters accordingly. If
`average=false`, it loops over each slice of x and applies updates individually
to each column of the slice.

# Examples

```julia
code opt_vae = (...) opt_mlp = (...) x = rand(Float32, 10, 100, 5) 
# 5 samples of 100 timesteps with 10 features each

train!(infomaxvae, x, opt_vae, opt_mlp, average=true)
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractArray{Float32,3},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Permute the data for each slice to compute the mutual information
    x_shuffled = map(eachslice(x, dims=3)) do slice
        # Shuffling needs to be done along the second dimension for each slice
        slice[:, Random.shuffle(1:size(slice, 2)), :]
    end

    # Decide on the training approach based on 'average'
    if average
        # Compute the averaged gradient across all slices of the tensor
        ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae_model
            StatsBase.mean([
                StatsBase.mean(
                    loss_function.(
                        Ref(vae_model), eachcol(slice), Ref(loss_kwargs)...
                    )
                )
                for (slice, slice_shuffled) in
                zip(eachslice(x, dims=3), eachslice(x_shuffled, dims=3))
            ])
        end # do block
        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

        # Compute the averaged gradient across all slices of the tensor
        ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
            StatsBase.mean([
                StatsBase.mean(
                    mlp_loss_function.(
                        Ref(mlp), eachcol(slice), Ref(mlp_loss_kwargs)...
                    )
                )
                for slice in eachslice(x, dims=3)
            ])
        end # do block
        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
    else
        foreach(
            slice -> foreach(
                col -> train!(
                    infomax, col, opt_vae, opt_mlp;
                    loss_function, loss_kwargs,
                    mlp_loss_function, mlp_loss_kwargs
                ),
                eachcol(slice)
            ),
            eachslice(x, dims=3)
        )
    end # if
end # function

# ==============================================================================

"""
    `train!(infomaxvae, x_in, x_out, opt_vae, opt_mlp; loss_function=loss, 
    loss_kwargs, mlp_loss_function=mlp_loss, mlp_loss_kwargs)`

Customized training function to update parameters of an InfoMax variational
autoencoder (VAE) given a loss function of the specified form.

The InfoMax VAE loss function can be defined as:

    loss_infoMax = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z) || π(z)) -
                   α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

This function simultaneously optimizes two neural networks: the VAE itself and a
multi-layer perceptron (MLP) used to compute the mutual information between
input and latent variables.

# Arguments
- `infomaxvae::InfoMaxVAE`: Struct containing the elements of an InfoMax VAE.
- `x_in::AbstractVector{Float32}`: Input data for the loss function. Represents
  an individual sample.
- `x_out::AbstractVector{Float32}`: Target output data for the loss function.
  Represents the corresponding output for the `x_in` sample.
- `opt_vae::Flux.Optimise.Optimiser`: Optimizing algorithm to be used for
  updating the VAE parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: Optimizing algorithm to be used for
  updating the MLP parameters.

# Optional Keyword arguments
- `loss_function::Function`: The loss function to be used during training,
  defaulting to `loss`.
- `loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
  passed to the loss function.
- `mlp_loss_function::Function`: The loss function to be used during training
  for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
  

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function to jointly
train the VAE and MLP. The VAE parameters are updated to minimize the InfoMaxVAE
loss, while the MLP parameters are updated to maximize the estimated mutual
information. The function allows for customization of loss hyperparameters
during training.

# Examples
```julia
opt_vae = Flux.Optimise.ADAM(1e-3)
opt_mlp = Flux.Optimise.ADAM(1e-3)

for x in dataloader
    train!(infomaxvae, x, opt_vae, opt_mlp, α=100.0)
end
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # Permute data for computation of mutual information
    x_shuffle = @view x_in[Random.shuffle(1:end)]

    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss_function(
            vae, infomaxvae.mlp, x_in, x_out, x_shuffle; loss_kwargs...
        )
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss_function(
            infomaxvae.vae, mlp, x_in, x_shuffle; mlp_loss_kwargs...
        )
    end # do

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
end # function

@doc raw"""
    `train!(infomaxvae, x_in, x_out,  opt_vae, opt_mlp; loss_function=loss, 
    loss_kwargs, mlp_loss_function=mlp_loss, mlp_loss_kwargs, average=true)`

Trains an InfoMax Variational Autoencoder (VAE) on a batch of data represented
as a matrix, with optional averaging of gradients across data points.

# Arguments
- `infomaxvae::InfoMaxVAE`: The InfoMax VAE model to be trained.
- `x_in::AbstractMatrix{Float32}`: Matrix of input data samples on which to
  evaluate the loss function. Each column represents an individual sample.
- `x_out::AbstractMatrix{Float32}`: Matrix of target output data samples. Each
  column represents the corresponding output for an individual sample in `x_in`.
- `opt_vae::Flux.Optimise.Optimiser`: The optimizer to use for the VAE
  parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: The optimizer to use for the MLP
  parameters.


# Optional Keyword Arguments
- `loss_function::Function`: The loss function to be used for training. Defaults
  to a predefined `loss` function.
- `loss_kwargs::Union{NamedTuple,Dict}`: Keyword arguments to pass to the loss
  function.
- `mlp_loss_function::Function`: The loss function to be used during training
for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
- `average::Bool`: A boolean flag to determine if gradients should be averaged
  across all data points. If `true`, gradients are averaged. If `false`,
  `train!` is called on each data point separately.

# Description
This function can either average the gradients across all the data points before
updating the parameters (`average=true`) or update the parameters for each data
point separately (`average=false`). When `average` is set to `true`, it computes
the gradients by averaging the loss over all columns of the input data matrix
`x` and applies the updates to the VAE and MLP parameters accordingly. When
`average` is set to `false`, it loops over each column and applies updates
individually.

# Examples
```julia
opt_vae = Flux.Optimise.ADAM(1e-3)
opt_mlp = Flux.Optimise.ADAM(1e-3)
x = rand(Float32, 784, 100) # 100 samples of 784-dimensional data

train!(infomaxvae, x, opt_vae, opt_mlp, average=false)
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Permute data for computation of mutual information
    x_shuffle = @view x_in[:, Random.shuffle(1:end)]

    # Decide on training approach based on 'average'
    if average
        # == VAE == #
        # Compute gradient
        ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
            StatsBase.mean(
                loss_function.(
                    Ref(vae),
                    Ref(infomaxvae.mlp),
                    eachcol(x_in),
                    eachcol(x_out),
                    eachcol(x_shuffle);
                    loss_kwargs...
                )
            )
        end # do

        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

        # == MLP == #
        # Compute gradient
        ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
            StatsBase.mean(
                mlp_loss_function.(
                    Ref(infomaxvae.vae),
                    Ref(mlp),
                    eachcol(x_in),
                    eachcol(x_out),
                    eachcol(x_shuffle);
                    mlp_loss_kwargs...
                )
            )
        end # do
        # Update the MLP network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
    else
        foreach(
            (col_in, col_out) -> train!(
                infomaxvae, col_in, col_out, opt_vae, opt_mlp;
                loss_function=loss_function,
                loss_kwargs=loss_kwargs,
                mlp_loss_function=mlp_loss_function,
                mlp_loss_kwargs=mlp_loss_kwargs
            ),
            zip(eachcol(x_in), eachcol(x_out))
        )
    end # if
end # function

@doc raw"""
        train!(infomaxvae, x_in, x_out, opt_vae, opt_mlp; loss_function=loss, 
        loss_kwargs, mlp_loss_function=mlp_loss, mlp_loss_kwargs, 
        average=true)

Trains an InfoMax Variational Autoencoder (VAE) on a batch of 3D data, with
optional averaging of gradients across the tensor slices.

# Arguments

# Arguments
- `infomaxvae::InfoMaxVAE`: The InfoMax VAE model to be trained.
- `x_in::AbstractArray{Float32,3}`: A 3D tensor of training data where the first
  dimension corresponds to features, the second to time or sequence, and the
  third to different samples or batch items.
- `x_out::AbstractArray{Float32,3}`: A 3D tensor of training data where the
  first dimension corresponds to features, the second to time or sequence, and
  the third to different samples or batch items.
- `opt_vae::Flux.Optimise.Optimiser`: The optimizer to use for the VAE
  parameters.
- `opt_mlp::Flux.Optimise.Optimiser`: The optimizer to use for the MLP
  parameters.


# Optional Keyword Arguments
- `loss_function::Function`: The loss function to be used for training. Defaults
  to a predefined `loss` function.
- `loss_kwargs::Union{NamedTuple,Dict}`: Keyword arguments to pass to the loss
  function.
- `mlp_loss_function::Function`: The loss function to be used during training
for the MLP computing the variational free energy, defaulting to `mlp_loss`.
- `mlp_loss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
    passed to the MLP loss function.
- `average::Bool`: A boolean flag to determine if gradients should be averaged
  across all data points. If `true`, gradients are averaged. If `false`,
  `train!` is called on each data point separately.

# Description

This function allows training on 3D data by handling each slice along the third
dimension either in an averaged manner or individually. If `average=true`, the
function computes the gradients by averaging the loss across all slices of the
3D tensor `x_in` and updates the VAE and MLP parameters accordingly. If
`average=false`, it loops over each slice of x and applies updates individually
to each column of the slice.

# Examples

```julia
# Assuming the existence of a predefined loss function `loss` and `mlp_loss`
infomaxvae = InfoMaxVAE(...)  # Instantiate the model
opt_vae = ADAM(...)  # Set up the optimizer for VAE
opt_mlp = ADAM(...)  # Set up the optimizer for MLP
x_in = rand(Float32, 10, 100, 5)
x_out = rand(Float32, 10, 100, 5)  # Corresponding target data

train!(infomaxvae, x_in, x_out, opt_vae, opt_mlp, average=true)
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x_in::AbstractArray{Float32,3},
    x_out::AbstractArray{Float32,3},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    average=true
)
    # Permute the data for each slice to compute the mutual information
    x_shuffled = map(eachslice(x_in, dims=3)) do slice
        # Shuffling needs to be done along the second dimension for each slice
        slice[:, Random.shuffle(1:size(slice, 2)), :]
    end

    # Decide on the training approach based on 'average'
    if average
        # Compute the averaged gradient across all slices of the tensor
        ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae_model
            StatsBase.mean([
                StatsBase.mean(
                    loss_function.(
                        Ref(vae_model),
                        eachcol(slice_in),
                        eachcol(slice_out),
                        eachcol(slice_shuffled),
                        Ref(loss_kwargs)...
                    )
                )
                for (slice_in, slice_out, slice_shuffled) in
                zip(
                    eachslice(x_in, dims=3),
                    eachslice(x_out, dims=3),
                    eachslice(x_shuffled, dims=3)
                )
            ])
        end # do block
        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

        # Compute the averaged gradient across all slices of the tensor
        ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
            StatsBase.mean([
                StatsBase.mean(
                    mlp_loss_function.(
                        Ref(mlp), eachcol(slice_in), Ref(mlp_loss_kwargs)...
                    )
                )
                for slice in eachslice(x_in, dims=3)
            ])
        end # do block
        # Update the VAE network parameters averaging gradient from all datasets
        Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
    else
        foreach(
            ((slice_in, slice_out) -> foreach(
                (col_in, col_out) -> train!(
                    infomax, col_in, col_out, opt_vae, opt_mlp;
                    loss_function=loss_function,
                    loss_kwargs=loss_kwargs,
                    mlp_loss_function=mlp_loss_function,
                    mlp_loss_kwargs=mlp_loss_kwargs
                ),
                zip(eachcol(slice_in), eachcol(slice_out))
            ),
            zip(eachslice(x_in, dims=3), eachslice(x_out, dims=3))
        )
        )
    end # if
end # function