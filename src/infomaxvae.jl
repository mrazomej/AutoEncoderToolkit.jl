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
    AbstractVariationalDecoder, JointEncoder, SimpleDecoder, JointDecoder,
    SplitDecoder, VAE

using ..VAEs: reparameterize

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMax-VAE
# Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
# Information in Variational Autoencoders. in 2020 IEEE International Symposium
# on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
# doi:10.1109/ISIT44484.2020.9174424.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `InfoMaxVAE`

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

"""
    MLP(
        n_input::Int,
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
        Flux.Dense(n_input => mlp_neurons[1], mlp_activation[1]; init=init)
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
    (vae::InfoMaxVAE{<:AbstractVariationalEncoder,T, Flux.Chain})(
        x::AbstractVecOrMat{Float32}; 
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
        latent::Bool=false, 
        n_samples::Int=1) where {T<:Union{JointDecoder,SplitDecoder}}

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
- If `latent=true`: A dictionary with keys `:encoder_µ`, `:encoder_logσ`, `:z`,
  `:decoder_µ`, `:decoder_logσ`, and `:mutual_info`, containing the
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
function (infomaxvae::InfoMaxVAE{VAE{E,D},Flux.Chain})(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
) where {E<:AbstractVariationalEncoder,D<:SimpleDecoder}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = infomaxvae.vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=n_samples
    )

    # Run latent sample through decoder to obtain mean and log std
    decoder_µ = infomaxvae.vae.decoder(z_sample)

    # Check if latent variables and mutual information should be returned
    if latent
        # Compute mutual information estimate using MLP
        mutual_info = vae.mlp([x; z_sample])

        return Dict(
            :encoder_µ => encoder_µ,
            :encoder_logσ => encoder_logσ,
            :z => z_sample,
            :decoder_µ => decoder_µ,
            :mutual_info => mutual_info
        )
    else
        # or return reconstructed data from decoder
        return decoder_µ
    end # if
end # function

@doc raw"""
    (vae::InfoMaxVAE{<:AbstractVariationalEncoder,T, Flux.Chain})(
        x::AbstractVecOrMat{Float32}; 
        prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0), 
        latent::Bool=false, 
        n_samples::Int=1) where {T<:Union{JointDecoder,SplitDecoder}}

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
- If `latent=true`: A dictionary with keys `:encoder_µ`, `:encoder_logσ`, `:z`,
  `:decoder_µ`, `:decoder_logσ`, and `:mutual_info`, containing the
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
function (infomaxvae::InfoMaxVAE{VAE{E,D},Flux.Chain})(
    x::AbstractVecOrMat{Float32},
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0);
    latent::Bool=false,
    n_samples::Int=1
) where {E<:AbstractVariationalEncoder,D<:Union{JointDecoder,SplitDecoder}}
    # Run input through encoder to obtain mean and log std
    encoder_µ, encoder_logσ = infomaxvae.vae.encoder(x)

    # Run reparametrization trick
    z_sample = reparameterize(
        encoder_µ, encoder_logσ; prior=prior, n_samples=n_samples
    )

    # Run latent sample through decoder to obtain mean and log std
    decoder_µ, decoder_logσ = infomaxvae.vae.decoder(z_sample)

    # Check if latent variables and mutual information should be returned
    if latent
        # Compute mutual information estimate using MLP
        mutual_info = vae.mlp([x; z_sample])

        return Dict(
            :encoder_µ => encoder_µ,
            :encoder_logσ => encoder_logσ,
            :z => z_sample,
            :decoder_µ => decoder_µ,
            :decoder_logσ => decoder_logσ,
            :mutual_info => mutual_info
        )
    else
        # or return reconstructed data from decoder
        return decoder_µ, decoder_logσ
    end # if
end # function

# ==============================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE loss function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `loss(vae, mlp, x, x_shuffle; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
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
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `mlp::Flux.Chain`: A multi-layer perceptron used to compute the mutual
  information term.
- `x::AbstractVector{Float32}`: Input vector.
- `x_shuffle::AbstractVector{Float32}`: Shuffled input vector used for
  estimating mutual information.

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
- Ensure that the input data `x` and `x_shuffle` match the expected input
  dimensionality for the encoder in the VAE. The `x_shuffle` should be a
  permutation of `x` to estimate the mutual information correctly.
- For batch processing or evaluating an entire dataset, use:
  `sum(loss.(Ref(vae), eachcol(x), eachcol(x_shuffle)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

@doc raw"""
    `loss(vae, mlp, x_in, x_out, x_shuffle; σ=1.0f0, β=1.0f0, α=1.0f0, 
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
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks.
- `mlp::Flux.Chain`: A multilayer perceptron for approximating the mutual
  information between inputs and latent variables.
- `x_in::AbstractVector{Float32}`: Input vector for the encoder.
- `x_out::AbstractVector{Float32}`: Target output vector for the decoder. For
  batch processing or evaluating the entire dataset, use: `sum(loss.(Ref(vae),
  Ref(mlp), eachcol(x_in), eachcol(x_out), eachcol(x_shuffle)))`.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of `x_in` to compute
  the mutual information with permuted latent variables.

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
- For batch processing or evaluating an entire dataset, use:
    `sum(loss.(Ref(vae), eachcol(x_in), eachcol(x_out), eachcol(x_shuffle)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x_out .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x_in, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x_in, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x_in, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x_in, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

# ==============================================================================

@doc raw"""
    `loss(vae, mlp, x, x_shuffle; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
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
- `vae::VAE{<:AbstractVariationalEncoder, <:Union{JointDecoder, SplitDecoder}}`:
  The VAE model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x::AbstractVector{Float32}`: Input data vector.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data to help
  estimate mutual information.

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
- Ensure that `x` and `x_shuffle` have proper dimensions as expected by the
  encoder within the VAE. `x_shuffle` should be a permutation of `x` to
  facilitate an accurate mutual information estimate.
- For batch processing or evaluating an entire dataset, use:
`sum(loss.(Ref(vae), eachcol(x), eachcol(x_shuffle)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function with n_samples
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z_samples = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x .- decoder_µ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated encoder qᵩ(z|x_in)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

@doc raw"""
    `loss(vae, mlp, x, x_shuffle; σ=1.0f0, β=1.0f0, α=1.0f0, n_samples=1, 
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
- `vae::VAE{<:AbstractVariationalEncoder, <:Union{JointDecoder, SplitDecoder}}`:
  The VAE model consisting of encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
error.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data to help
  estimate mutual information.

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
- Ensure that `x_in` and `x_shuffle` have proper dimensions as expected by the
  encoder within the VAE. `x_shuffle` should be a permutation of `x` to
  facilitate an accurate mutual information estimate.
- For batch processing or evaluating an entire dataset, use:
  `sum(loss.(Ref(vae), eachcol(x_in), eachcol(x_out), eachcol(x_shuffle)))`.
"""
function loss(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function with n_samples
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z_samples = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x_out .- decoder_µ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x_in, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x_in, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x_in, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x_in, eachcol(z_shuffle)])) / n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated encoder qᵩ(z|x_in)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute loss function
    return -logπ_x_z + β * kl_qᵩ_π - α * info_x_z + reg_strength * reg_term
end #function

# ==============================================================================

@doc raw"""
    mlp_loss(vae, mlp, x, x_shuffle; n_samples=1, regularization=nothing, 
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
- `x::AbstractVector{Float32}`: The input vector for the VAE.
- `x_shuffle::AbstractVector{Float32}`: The shuffled input vector for
  calculating mutual information with permuted latent variables.

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
information for true `x` and latent `z`, and `z̃` represents latent variables
associated with shuffled inputs `x_shuffle`.

The function is used to separately train the MLP to estimate mutual information,
which is a component of the larger InfoMaxVAE model.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the VAE.
- For batch processing or evaluating an entire dataset, use:
`sum(mlp_loss.(Ref(vae), eachcol(x)))`.

# Examples
```julia
# Assuming vae, mlp, x, and x_shuffle are predefined and properly formatted:
loss_value = mlp_loss(vae, mlp, x, x_shuffle)
```
"""
function mlp_loss(
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32},
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Extract relevant variable
    z = outputs[:z]

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x_in, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x_in, eachcol(z_shuffle)])) / n_samples
    end

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute (negative) variational mutual information as loss function
    return -sum(@. I_xz - exp(I_xz_perm - 1)) + reg_strength * reg_term
end #function

# ==============================================================================
# Individual loss terms
# ==============================================================================

"""
    loss_terms(vae::VAE, mlp::Flux.Chain, x::AbstractVector{Float32}, 
    x_shuffle::AbstractVector{Float32}; n_samples::Int=1) -> Vector{Float32}

Calculate individual terms of the loss function for an InfoMax variational
autoencoder (VAE) without their respective scaling factors.

This function processes the input `x` through the VAE and a separate MLP used
for estimating mutual information. It then returns the array of loss components
computed from the original and shuffled input data. These components are useful
for analyzing the contribution of each term to the total loss without the
scaling factors applied.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks..
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x::AbstractVector{Float32}`: Original input data vector.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data, used to
  estimate mutual information.

# Optional Keyword Arguments
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.

# Returns
- `Vector{Float32}`: An array containing the unscaled components of the loss
  function:
    1. Average log likelihood of the probabilistic decoder ⟨log π(x|z)⟩   
    2. Kullback-Leibler (KL) divergence Dₖₗ[q(z|x) || p(z)]
    3. Variational mutual information I(x;z)

# Notes
- The function assumes a Gaussian decoder in the VAE.
- The input `x` and `x_shuffle` must have proper dimensions as expected by the
  encoder within the VAE.
- `x_shuffle` should be a permutation of `x` to facilitate an accurate mutual
  information estimate.
- The return values are components of the loss function before scaling and do
  not represent the final loss value.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    n_samples::Int=1
)
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Return array of terms without their multiplication constants
    return [logπ_x_z, kl_qᵩ_π, info_x_z]
end # function

"""
    loss_terms(vae::VAE, mlp::Flux.Chain, x_in::AbstractVector{Float32}, 
        x_out::AbstractVector{Float32} x_shuffle::AbstractVector{Float32}; 
        n_samples::Int=1) -> Vector{Float32}

Calculate individual terms of the loss function for an InfoMax variational
autoencoder (VAE) without their respective scaling factors.

This function processes the input `x` through the VAE and a separate MLP used
for estimating mutual information. It then returns the array of loss components
computed from the original and shuffled input data. These components are useful
for analyzing the contribution of each term to the total loss without the
scaling factors applied.

# Arguments
- `vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder}`: A VAE model with
  encoder and decoder networks..
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
  error.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data, used to
  estimate mutual information.

# Optional Keyword Arguments
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.

# Returns
- `Vector{Float32}`: An array containing the unscaled components of the loss
  function:
    1. Average log likelihood of the probabilistic decoder ⟨log π(x|z)⟩   
    2. Kullback-Leibler (KL) divergence Dₖₗ[q(z|x) || p(z)]
    3. Variational mutual information I(x;z)

# Notes
- The function assumes a Gaussian decoder in the VAE.
- The input `x_in` and `x_shuffle` must have proper dimensions as expected by
  the encoder within the VAE.
- `x_shuffle` should be a permutation of `x_in` to facilitate an accurate mutual
  information estimate.
- The return values are components of the loss function before scaling and do
  not represent the final loss value.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,SimpleDecoder},
    mlp::Flux.Chain,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    n_samples::Int=1
)
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Unpack outputs
    µ, logσ, z, x̂ = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z],
        outputs[:decoder_µ]
    )

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    logπ_x_z = -1 / (2 * σ^2 * n_samples) * sum((x_out .- x̂) .^ 2)

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x_in, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x_in, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x_in, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x_in, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. exp(2.0f0 * logσ) + µ^2 - 1.0f0 - 2.0f0 * logσ
    )

    # Return array of terms without their multiplication constants
    return [logπ_x_z, kl_qᵩ_π, info_x_z]
end # function

# ==============================================================================

"""
    loss_terms(vae::VAE, mlp::Flux.Chain, x_in::AbstractVector{Float32}, 
        x_out::AbstractVector{Float32}, x_shuffle::AbstractVector{Float32}; 
        n_samples::Int=1) -> Vector{Float32

Calculate individual terms of the loss function for an InfoMax variational
autoencoder (VAE) without their respective scaling factors.

This function processes the input `x` through the VAE and a separate MLP used
for estimating mutual information. It then returns the array of loss components
computed from the original and shuffled input data. These components are useful
for analyzing the contribution of each term to the total loss without the
scaling factors applied.

# Arguments
- `vae::VAE`: The VAE model comprising encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x::AbstractVector{Float32}`: Original input data vector.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data, used to
  estimate mutual information.

# Optional Keyword Arguments
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.

# Returns
- `Vector{Float32}`: An array containing the unscaled components of the loss
  function:
    1. Average log likelihood of the probabilistic decoder ⟨log π(x|z)⟩   
    2. Kullback-Leibler (KL) divergence Dₖₗ[q(z|x) || p(z)]
    3. Variational mutual information I(x;z)

# Notes
- The function assumes a Gaussian decoder in the VAE.
- The input `x` and `x_shuffle` must have proper dimensions as expected by the
  encoder within the VAE.
- `x_shuffle` should be a permutation of `x` to facilitate an accurate mutual
  information estimate.
- The return values are components of the loss function before scaling and do
  not represent the final loss value.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function with n_samples
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z_samples = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x .- decoder_µ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x, eachcol(z_shuffle)])) / n_samples
    end

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated encoder
    # qᵩ(z|x_in) and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Return array of terms without their multiplication constants
    return [logπ_x_z, kl_qᵩ_π, info_x_z]
end # function

"""
    loss_terms(vae::VAE, mlp::Flux.Chain, x_in::AbstractVector{Float32}, 
        x_out::AbstractVector{Float32}, x_shuffle::AbstractVector{Float32}; 
        n_samples::Int=1) -> Vector{Float32}

Calculate individual terms of the loss function for an InfoMax variational
autoencoder (VAE) without their respective scaling factors.

This function processes the input `x` through the VAE and a separate MLP used
for estimating mutual information. It then returns the array of loss components
computed from the original and shuffled input data. These components are useful
for analyzing the contribution of each term to the total loss without the
scaling factors applied.

# Arguments
- `vae::VAE`: The VAE model comprising encoder and decoder components.
- `mlp::Flux.Chain`: A multi-layer perceptron for estimating mutual information
  I(x;z).
- `x_in::AbstractVector{Float32}`: Input vector to the VAE encoder.
- `x_out::AbstractVector{Float32}`: Target vector to compute the reconstruction
  error.
- `x_shuffle::AbstractVector{Float32}`: Shuffled version of input data, used to
  estimate mutual information.

# Optional Keyword Arguments
- `n_samples::Int=1`: Number of latent samples to average over for loss
  computation.

# Returns
- `Vector{Float32}`: An array containing the unscaled components of the loss
  function:
    1. Average log likelihood of the probabilistic decoder ⟨log π(x|z)⟩   
    2. Kullback-Leibler (KL) divergence Dₖₗ[q(z|x) || p(z)]
    3. Variational mutual information I(x;z)

# Notes
- The function assumes a Gaussian decoder in the VAE.
- The input `x` and `x_shuffle` must have proper dimensions as expected by the
  encoder within the VAE.
- `x_shuffle` should be a permutation of `x` to facilitate an accurate mutual
  information estimate.
- The return values are components of the loss function before scaling and do
  not represent the final loss value.
"""
function loss_terms(
    vae::VAE{<:AbstractVariationalEncoder,T},
    mlp::Flux.Chain,
    x_in::AbstractVector{Float32},
    x_out::AbstractVector{Float32},
    x_shuffle::AbstractVector{Float32};
    n_samples::Int=1
) where {T<:Union{JointDecoder,SplitDecoder}}
    # Run input through reconstruct function with n_samples
    outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Extract encoder-related terms
    encoder_µ, encoder_logσ, z_samples = (
        outputs[:encoder_µ],
        outputs[:encoder_logσ],
        outputs[:z]
    )

    # Extract decoder-related terms
    decoder_µ, decoder_logσ = outputs[:decoder_µ], outputs[:decoder_logσ]

    # Run shuffle input through reconstruct function
    outputs_shuffle = vae(x_shuffle; latent=true, n_samples=n_samples)
    # Extract relevant variable
    z_shuffle = outputs_shuffle[:z]

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute average reconstruction loss ⟨log π(x|z)⟩ for a Gaussian decoder
    # averaged over all samples
    logπ_x_z = -1 / (2.0f0 * n_samples) * length(decoder_µ) * log(2 * π) -
               1 / n_samples * sum(decoder_logσ) -
               1 / (2.0f0 * n_samples) * sum((x_out .- decoder_µ) .^ 2 ./
                                             exp.(2 * decoder_logσ))

    # Mutual Information Calculation
    if n_samples == 1
        # Compute mutual information for real input
        I_xz = mlp([x_in, z])
        # Compute mutual information for shuffled input
        I_xz_perm = mlp([x_in, z_shuffle])
    else
        # Compute mutual information for real input
        I_xz = sum(mlp.([x_in, eachcol(z)])) / n_samples
        # Compute mutual information for shuffled input
        I_xz_perm = sum(mlp.([x_in, eachcol(z_shuffle)])) / n_samples
    end # if

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated encoder qᵩ(z|x_in)
    # and latent prior distribution π(z)
    kl_qᵩ_π = 1 / 2.0f0 * sum(
        @. (exp(2.0f0 * encoder_logσ) + encoder_μ^2 - 1.0f0) -
           2.0f0 * encoder_logσ
    )

    # Return array of terms without their multiplication constants
    return [logπ_x_z, kl_qᵩ_π, info_x_z]
end # function

# ==============================================================================

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE training functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `train!(vae, x, vae_opt, mlp_opt; loss_kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z) || π(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables.

# Arguments
- `vae::InfoMaxVAE`: Struct containint the elements of an InfoMax variational
  autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to
  evaluate the loss function. NOTE: Every column should represent a single
  input.
- `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the variational autoencoder parameters. This should be fed already with
  the corresponding parameters. For example, one could feed: 
  ⋅ Flux.AMSGrad(η)
- `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the multi-layered perceptron parameters. This should be fed already
  with the corresponding parameters. For example, one could feed: 
  ⋅ Flux.AMSGrad(η)

## Optional arguments
- `loss_kwargs::NamedTuple`: Tuple containing arguments for the loss function.
    For `InfoMaxVAEs.loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function, which
trains the VAE and MLP portions jointly.

The VAE parameters are updated to minimize the InfoMaxVAE loss. The MLP
parameters are updated to maximize the estimated mutual information.

Allows customization of loss hyperparameters during training.

# Examples
```julia
opt_vae = Flux.ADAM(1e-3)
opt_mlp = Flux.ADAM(1e-3)

for x in dataloader
train!(infomaxvae, x, x_true, opt_vae, opt_mlp, α=100.0)
end
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractVecOrMat{Float32},
    vae_opt::NamedTuple,
    mlp_opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0, :α => 1.0f0)
)

    # Permute data for computation of mutual information
    x_shuffle = @view x[:, Random.shuffle(1:end)]

    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss(vae, infomaxvae.mlp, x, x_shuffle; loss_kwargs...)
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        vae_opt,
        infomaxvae.vae,
        ∇vae_loss_[1]
    )

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss(infomaxvae.vae, mlp, x, x_shuffle)
    end # do

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        mlp_opt,
        infomaxvae.mlp,
        ∇mlp_loss_[1]
    )
end # function

@doc raw"""
    `train!(vae, x, x_true, vae_opt, mlp_opt; loss_kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log π(x|z)⟩ + β Dₖₗ(qᵩ(z) || π(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables.

# Arguments
- `vae::InfoMaxVAE`: Struct containint the elements of an InfoMax variational
  autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to
  evaluate the loss function. NOTE: Every column should represent a single
  input.
- `x_true;:AbstractVecOrMat{Float32}`: Array containing the data used to compare
  the reconstruction for the loss function. This can be used to train denoising
  VAE, for exmaple.
- `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
  needed to compute the mutual information term. This term is used to obtain an
  encoding `z_shuffle` that represents a random sample from the marginal π(z).
- `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the variational autoencoder parameters. This should be fed already with
  the corresponding parameters. For example, one could feed: ⋅ Flux.AMSGrad(η)
- `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
  update the multi-layered perceptron parameters. This should be fed already
  with the corresponding parameters. For example, one could feed: ⋅
  Flux.AMSGrad(η)

## Optional arguments
- `loss_kwargs::NamedTuple`: Tuple containing arguments for the loss function.
    For `InfoMaxVAEs.loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder π(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function, which
trains the VAE and MLP portions jointly.

The VAE parameters are updated to minimize the InfoMaxVAE loss. The MLP
parameters are updated to maximize the estimated mutual information.

Allows customization of loss hyperparameters during training. The main
difference with the method that only takes `x` as input is that the comparison
at the output layer does not need to necessarily match that of the input. Useful
for data augmentation training schemes.

# Examples
```julia
opt_vae = Flux.ADAM(1e-3)
opt_mlp = Flux.ADAM(1e-3)

for x in dataloader
  train!(infomaxvae, x, opt_vae, opt_mlp, α=100.0)
end
```
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32},
    vae_opt::NamedTuple,
    mlp_opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0, :α => 1.0f0)
)

    # Permute data for computation of mutual information
    x_shuffle = @view x[:, Random.shuffle(1:end)]

    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss(vae, infomaxvae.mlp, x, x_true, x_shuffle; loss_kwargs...)
    end # do

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss(infomaxvae.vae, mlp, x, x_shuffle)
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        vae_opt,
        infomaxvae.vae,
        ∇vae_loss_[1]
    )

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        mlp_opt,
        infomaxvae.mlp,
        ∇mlp_loss_[1]
    )
end # function