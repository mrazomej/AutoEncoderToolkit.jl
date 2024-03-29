# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

# Import library to use Ellipsis Notation
using EllipsisNotation

# Import ChainRulesCore to ignore functions when computing gradients
using ChainRulesCore: @ignore_derivatives

##

# Import Abstract Types

# Import Abstract Types
using ..AutoEncode: AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractGaussianEncoder,
    AbstractGaussianLogEncoder,
    AbstractVariationalDecoder, AbstractGaussianDecoder,
    AbstractGaussianLogDecoder, AbstractGaussianLinearDecoder

# Import Concrete Encoder Types
using ..AutoEncode: JointLogEncoder

# Import Concrete Decoder Types
using ..AutoEncode: BernoulliDecoder, SimpleDecoder,
    JointLogDecoder, SplitLogDecoder,
    JointDecoder, SplitDecoder

# Import Concrete VAE type
using ..AutoEncode: VAE

# Import log-probability functions
using ..AutoEncode: decoder_loglikelihood, spherical_logprior,
    encoder_logposterior, encoder_kl, Flatten

# Import functions from other modules
using ..VAEs: reparameterize

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMax-VAE
# Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
# Information in Variational Autoencoders. in 2020 IEEE International Symposium
# on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
# doi:10.1109/ISIT44484.2020.9174424.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# ==============================================================================
# Defining MutualInfoChain to compute the variational mutual information
# ==============================================================================

@doc raw"""
    MutualInfoChain

A `MutualInfoChain` is used to compute the variational mutual information when
training an InfoMaxVAE. The chain is composed of a series of layers that must
end with a single output: the mutual information between the latent variables
and the input data.

# Arguments
- `data::Union{Flux.Dense,Flux.Chain}`: The data layer of the MutualInfoChain.
  This layer is used to input the data.
- `latent::Union{Flux.Dense,Flux.Chain}`: The latent layer of the
  MutualInfoChain. This layer is used to input the latent variables.
- `mlp::Flux.Chain`: A multi-layer perceptron (MLP) that is used to compute the
  mutual information between the inputs and the latent representations. The MLP
  takes as input the latent variables and outputs a scalar representing the
  estimated variational mutual information.

# Citation
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. in 2020 IEEE International Symposium
> on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
> doi:10.1109/ISIT44484.2020.9174424.

# Note
If the input data is not a flat array, make sure to include a flattening layer
within `data`.
"""
struct MutualInfoChain
    data::Union{Flux.Dense,Flux.Chain}
    latent::Union{Flux.Dense,Flux.Chain}
    mlp::Flux.Chain
end

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor MutualInfoChain

# ------------------------------------------------------------------------------

@doc raw"""
    (mi::MutualInfoChain)(x::AbstractArray)

Forward pass function for the MutualInfoChain, which applies the MLP to an input
x.

# Arguments
- `x::AbstractArray`: The input array to be processed. The last dimension
  represents each data sample.
- `z::AbstractVecOrMat`: The latent representation of the input data. The last
  dimension represents each data sample.

# Returns
- The result of applying the MutualInfoChain to the input data and the latent
  representation simultaneously.

# Description
This function applies the MLP (Multilayer Perceptron) of a MutualInfoChain
instance to an input array. The MLP is a type of neural network used in the
MutualInfoChain for processing the input data.
"""
function (mi::MutualInfoChain)(x::AbstractArray, z::AbstractVecOrMat)
    # Arrange data and latent variables on top of each other to be fed to mlp.
    mlp_input = Flux.Parallel(vcat, mi.data, mi.latent)(x, z)
    # Then feed the data.
    return vec(mi.mlp(mlp_input))
end

# ------------------------------------------------------------------------------

"""
    MutualInfoChain(
        size_input::Union{Int,Vector{<:Int}},
        n_latent::Int,
        mlp_neurons::Vector{<:Int},
        mlp_activations::Vector{<:Function},
        output_activation::Function;
        init::Function = Flux.glorot_uniform
    )

Constructs a default `MutualInfoChain`. 

# Arguments
- `n_input::Int`: Number of input features to the `MutualInfoChain`.
- `n_latent::Int`: The dimensionality of the latent space.
- `mlp_neurons::Vector{<:Int}`: A vector of integers where each element
  represents the number of neurons in the corresponding hidden layer of the MLP.
- `mlp_activations::Vector{<:Function}`: A vector of activation functions to be
  used in the hidden layers. Length must match that of `mlp_neurons`.
- `output_activation::Function`: Activation function for the output neuron of
  the MLP.

# Optional Keyword Arguments
- `init::Function`: Initialization function for the weights of all layers in the
  `MutualInfoChain`. Defaults to `Flux.glorot_uniform`.

# Returns
- `MutualInfoChain`: A `MutualInfoChain` instance with the specified MLP
  architecture.

# Notes
The function will throw an error if the number of provided activation functions
does not match the number of layers specified in mlp_neurons.
"""
function MutualInfoChain(
    size_input::Union{Int,Vector{<:Int}},
    n_latent::Int,
    mlp_neurons::Vector{<:Int},
    mlp_activation::Vector{<:Function},
    output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check if there's an activation function for each layer, plus the output
    # layer
    if length(mlp_activation) != length(mlp_neurons)
        error("Each hidden layer needs exactly one activation function")
    end # if

    # Initialize MLP layers
    mlp_layers = []

    # Define number of input features
    n_input = prod(size_input)

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

    # Define data layer
    if isa(size_input, Number)
        # Define data layer as simple Dense layer with identity activation
        data_layer = Flux.Dense(n_input => n_input, Flux.identity; init=init)
    else
        # Define data layer as a chain of layers including a Flatten layer and
        # a Dense layer with identity activation
        data_layer = Flux.Chain(
            AutoEncode.Flatten,
            Flux.Dense(n_input => n_input, Flux.identity; init=init)
        )
    end # if

    # Define latent layer as a simple Dense layer with identity activation
    latent_layer = Flux.Dense(n_latent => n_latent, Flux.identity; init=init)

    return MutualInfoChain(
        data_layer,
        latent_layer,
        Flux.Chain(mlp_layers...)
    )
end # function

# ==============================================================================
# `InfoMaxVAE <: AbstractVariationalAutoEncoder`
# ==============================================================================

@doc raw"""
    `InfoMaxVAE <: AbstractVariationalAutoEncoder`

`struct` encapsulating an InfoMax variational autoencoder (InfoMaxVAE), an
architecture designed to enhance the VAE framework by maximizing mutual
information between the inputs and the latent representations, as per the
methods described by Rezaabad and Vishwanath (2020).

The model aims to learn representations that preserve mutual information with
the input data, arguably capturing more meaningful factors of variation.

# Fields
- `vae::VAE`: The core variational autoencoder, consisting of an encoder that
  maps input data into a latent space representation, and a decoder that
  attempts to reconstruct the input from the latent representation.
- `mi::MutualInfoChain`: A multi-layer perceptron (MLP) that estimates the
  mutual information between the input data and the latent representations.

# Usage
The `InfoMaxVAE` struct is utilized in a similar manner to a standard VAE, with
the added capability of mutual information maximization as part of the training
process. This involves an additional loss term that considers the output of the
`mi` network to encourage latent representations that are informative about the
input data.

# Example
```julia
# Assuming definitions for `encoder`, `decoder`, and `mi` are provided:
info_max_vae = InfoMaxVAE(VAE(encoder, decoder), mi)

# During training, one would maximize both the variational lower bound and the 
# mutual information estimate provided by `mlp`.
```

# Citation
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. in 2020 IEEE International Symposium
> on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
> doi:10.1109/ISIT44484.2020.9174424.
"""
struct InfoMaxVAE{
    V<:VAE{<:AbstractVariationalEncoder,<:AbstractVariationalDecoder}
} <: AbstractVariationalAutoEncoder
    vae::V
    mi::MutualInfoChain
end # struct

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor InfoMaxVAE

# ==============================================================================
# Variational Mutual Information Computation
# ==============================================================================

# ------------------------------------------------------------------------------
# Shuffle latent codes between data samples
# ------------------------------------------------------------------------------

@doc raw"""
    shuffle_latent(z::AbstractMatrix, seed::Int=Random.seed!())

Shuffle the elements of the second dimension of a matrix representing latent
space points.

# Arguments
- `z::AbstractMatrix`: A matrix representing latent codes. Each column
  corresponds to a single latent code.

# Optional Keyword Arguments
- `seed::Union{Nothing, Int}`: Optional argument. The seed for the random number
  generator. If not provided, a random seed will be used.

# Returns
- `AbstractMatrix`: A new matrix with the second dimension shuffled.
"""
function shuffle_latent(
    z::AbstractMatrix; seed::Union{Nothing,<:Int}=nothing
)
    # Check if seed is provided
    if seed === nothing
        # Initialize the random number generator
        rng = Random.GLOBAL_RNG
    else
        # Initialize the random number generator with the provided seed
        rng = MersenneTwister(seed)
    end # if
    # Define shuffle indexes
    shuffled_indices = Random.shuffle(rng, 1:size(z, 2))
    # Return shuffled data
    return z[:, shuffled_indices]
end # function

# Set ChainRulesCore to ignore the function when computing gradients
@ignore_derivatives shuffle_latent

# ------------------------------------------------------------------------------
# Variational Mutual information
# ------------------------------------------------------------------------------

"""
    variational_mutual_info(mi, x, z, z_shuffle)

Compute a variational approximation of the mutual information between the input
`x` and the latent code `z` using a `MutualInfoChain`. Note that this estimate
requires shuffling the latent codes between data samples. Therefore, it only
applies to batch data cases. A single sample will not provide a meaningful
estimate.

# Arguments
- `mi::MutualInfoChain`: A MutualInfoChain instance used to estimate mutual
  information.
- `x::AbstractArray`: Array of input data. The last dimension represents each
  data sample.
- `z::AbstractMatrix`: Matrix of corresponding latent representations of the
  input data.
- `z_shuffle::AbstractMatrix`: Matrix of latent representations where the second
  dimension has been shuffled.

# Returns
- `Float32`: An approximation of the mutual information between the input data
  and its corresponding latent representation.

# References
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).
"""
function variational_mutual_info(
    mi::MutualInfoChain,
    x::AbstractArray,
    z::AbstractVecOrMat,
    z_shuffle::AbstractVecOrMat,
)
    # Run input and real latent code through MutualInfoChain
    I_xz = StatsBase.mean(mi(x, z))
    # Run input and shuffled latent code through MutualInfoChain
    I_xz_perm = StatsBase.mean(exp.(mi(x, z_shuffle) .- 1))

    # Compute variational mutual information
    return I_xz - I_xz_perm
end # function

# ------------------------------------------------------------------------------

"""
    variational_mutual_info(infomaxvae, x, z, z_shuffle)

Compute a variational approximation of the mutual information between the input
`x` and the latent code `z` using an `InfoMaxVAE` instance. Note that this
estimate requires shuffling the latent codes between data samples. Therefore, it
only applies to batch data cases. A single sample will not provide a meaningful
estimate.

# Arguments
- `infomaxvae::InfoMaxVAE`: An InfoMaxVAE instance used to estimate mutual
  information.
- `x::AbstractArray`: Array of input data. The last dimension represents each
  data sample.
- `z::AbstractMatrix`: Matrix of corresponding latent representations of the
  input data.
- `z_shuffle::AbstractMatrix`: Matrix of latent representations where the second
  dimension has been shuffled.

# Returns
- `Float32`: An approximation of the mutual information between the input data
  and its corresponding latent representation.

# References
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).
"""
function variational_mutual_info(
    infomaxvae::InfoMaxVAE,
    x::AbstractArray,
    z::AbstractVecOrMat,
    z_shuffle::AbstractVecOrMat,
)
    # Run input and real latent code through MutualInfoChain
    I_xz = StatsBase.mean(infomaxvae.mi(x, z))
    # Run input and shuffled latent code through MutualInfoChain
    I_xz_perm = StatsBase.mean(exp.(infomaxvae.mi(x, z_shuffle) .- 1))

    # Compute variational mutual information
    return I_xz - I_xz_perm
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    variational_mutual_info(
        infomaxvae::InfoMaxVAE,
        x::AbstractArray;
        seed::Union{Nothing,Int}=nothing
    )

Compute a variational approximation of the mutual information between the input
`x` and the latent code `z` using an `InfoMaxVAE` instance. This function also
shuffles the latent codes between data samples to provide a meaningful estimate
even for a single data sample.

# Arguments
- `infomaxvae::InfoMaxVAE`: An InfoMaxVAE instance used to estimate mutual
  information.
- `x::AbstractArray`: Array of input data. The last dimension represents each
  data sample.

# Optional Keyword Arguments
- `seed::Union{Nothing,Int}`: Optional argument. The seed for the random number
  generator used for shuffling the latent codes. If not provided, a random seed
  will be used.

# Returns
- `Float32`: An approximation of the mutual information between the input data
  and its corresponding latent representation.

# References
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).
"""
function variational_mutual_info(
    infomaxvae::InfoMaxVAE,
    x::AbstractArray;
    seed::Union{Nothing,Int}=nothing
)
    # Obtain latent variables from VAE
    z = infomaxvae.vae(x; latent=true).z
    # Shuffle latent codes between data samples
    z_shuffle = @ignore_derivatives shuffle_latent(z; seed=seed)

    return variational_mutual_info(infomaxvae, x, z, z_shuffle)
end # function

# ==============================================================================
# InfoMaxVAE Forward Pass
# ==============================================================================

@doc raw"""
    (vae::InfoMaxVAE)(x::AbstractArray; latent::Bool=false) 

Processes the input data `x` through an InfoMaxVAE, which consists of an
encoder, a decoder, and a multi-layer perceptron (MLP) to estimate variational
mutual information.

# Arguments
- `x::AbstractArray`: The data to be decoded. If array, the last dimension
  contains each data sample. 

# Optional Keyword Arguments
- `latent::Bool`: If `true`, returns a dictionary with latent variables and
  mutual information estimations along with the reconstruction. Defaults to
  `false`.
- `seed::Union{Nothing,Int}`: Optional argument. The seed for the random number
  generator used for shuffling the latent codes. If not provided, a random seed
  will be used.

# Returns
- If `latent=false`: The decoder output as a `NamedTuple`.
- If `latent=true`: A `NamedTuple` with the `:vae` field that contains the
  outputs of the VAE, and the `:mi` field that contains the estimate of the
  variational mutual information. Note that this estimate requires shuffling the
  latent codes between data samples. Therefore, it is only meaningful for batch
  data cases.

# Description
This function first encodes the input `x` . It then samples from this
distribution using the reparametrization trick. The sampled latent vectors are
then decoded, and the MutualInfoChain is used to estimate the mutual
information.

# Note
Ensure the input data `x` matches the expected input dimensionality for the
encoder in the InfoMaxVAE.
"""
function (infomaxvae::InfoMaxVAE)(
    x::AbstractArray;
    latent::Bool=false,
    seed::Union{Nothing,Int}=nothing
)
    # Check if latent variables and mutual information should be returned
    if latent
        # Pass input through VAE and store intermediate results
        vae_output = infomaxvae.vae(x; latent=latent)
        # Extract latent variables
        z = vae_output.z
        # Shuffle latent codes between data samples
        z_shuffle = @ignore_derivatives shuffle_latent(z; seed=seed)

        # Compute mutual information estimate using the MutualInfoChain
        mi = variational_mutual_info(infomaxvae, x, z, z_shuffle)

        # Add mutual_info to the NamedTuple
        return (vae=vae_output, mi=mi,)
    else
        # or return reconstructed data from decoder
        return infomaxvae.vae(x; latent=false)
    end # if
end # function

# ==============================================================================
# Loss InfoMax
# ==============================================================================

@doc raw"""
    infomaxloss(
        vae::VAE,
        mi::MutualInfoChain,
        x::AbstractArray;
        β=1.0f0,
        α=1.0f0,
        n_samples::Int=1,
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        kl_divergence::Function=encoder_kl,
        regularization::Union{Function,Nothing}=nothing,
        reg_strength::Float32=1.0f0,
        seed::Union{Nothing,Int}=nothing
    )

Computes the loss for an InfoMax variational autoencoder (VAE) with mutual
information constraints, by averaging over `n_samples` latent space samples.

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, the variational mutual information between input and latent
representations, and possibly a regularization term, defined as:

loss = -⟨log p(x|z)⟩ + β × Dₖₗ[qᵩ(z|x) || p(z)] - α × I(x;z) + reg_strength ×
reg_term

Where:
- `⟨log p(x|z)⟩` is the expected log likelihood of the probabilistic decoder. -
`Dₖₗ[qᵩ(z|x) || p(z)]` is the KL divergence between the approximated encoder and
the prior over the latent space.
- `I(x;z)` is the variational mutual information between the inputs `x` and the
  latent variables `z`.

# Arguments
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `mi::MutualInfoChain`: A MutualInfoChain instance used to estimate mutual
  information term.
- `x::AbstractArray`: Input data. The last dimension represents each data
  sample.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `α::Float32=1.0f0`: Weighting factor for the mutual information term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  that computes the log likelihood of the decoder's output.
- `kl_divergence::Function=encoder_kl`: A function that computes the KL
  divergence between the encoder's output and the prior.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the VAE outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.
- `seed::Union{Nothing,Int}`: The seed for the random number generator used for
  shuffling the latent codes. If not provided, a random seed will be used.

# Returns
- `Float32`: The computed average loss value for the input `x` and its
  reconstructed counterparts over `n_samples` samples, including possible
  regularization terms and the mutual information constraint.

# Note
- This function takes the `vae` and `mi` instances of an InfoMaxVAE model as
  separate arguments to be able to compute a gradient only with respect to the
  `vae` parameters.
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function infomaxloss(
    vae::VAE,
    mi::MutualInfoChain,
    x::AbstractArray;
    β=1.0f0,
    α=1.0f0,
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0,
    seed::Union{Nothing,Int}=nothing
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_output = vae(x; latent=true)

    # Compute ⟨log p(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = StatsBase.mean(
        reconstruction_loglikelihood(
            x, vae_output.z, vae.decoder, vae_output.decoder
        )
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution p(z)
    kl_div = StatsBase.mean(
        kl_divergence(vae.encoder, vae_output.encoder)
    )

    # Permute latent codes for computation of mutual information
    z_shuffle = @ignore_derivatives shuffle_latent(vae_output.z, seed=seed)

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mi, x, vae_output.z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(vae_output) : 0.0f0

    # Compute loss function
    return -log_likelihood + β * kl_div - α * mutual_info +
           reg_strength * reg_term
end #function

# ------------------------------------------------------------------------------

@doc raw"""
    infomaxloss(
        vae::VAE,
        mi::MutualInfoChain,
        x_in::AbstractArray,
        x_out::AbstractArray;
        β=1.0f0,
        α=1.0f0,
        n_samples::Int=1,
        reconstruction_loglikelihood::Function=decoder_loglikelihood,
        kl_divergence::Function=encoder_kl,
        regularization::Union{Function,Nothing}=nothing,
        reg_strength::Float32=1.0f0,
        seed::Union{Nothing,Int}=nothing
    )

Computes the loss for an InfoMax variational autoencoder (VAE) with mutual
information constraints, by averaging over `n_samples` latent space samples.

The loss function combines the reconstruction loss with the Kullback-Leibler
(KL) divergence, the variational mutual information between input and latent
representations, and possibly a regularization term, defined as:

loss = -⟨log p(x|z)⟩ + β × Dₖₗ[qᵩ(z|x) || p(z)] - α × I(x;z) + reg_strength ×
reg_term

Where:
- `⟨log p(x|z)⟩` is the expected log likelihood of the probabilistic decoder. -
`Dₖₗ[qᵩ(z|x) || p(z)]` is the KL divergence between the approximated encoder and
the prior over the latent space.
- `I(x;z)` is the variational mutual information between the inputs `x` and the
  latent variables `z`.

# Arguments
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `mi::MutualInfoChain`: A MutualInfoChain instance used to estimate mutual
  information term.
- `x_in::AbstractArray`: Input matrix. The last dimension represents each data
  sample.
- `x_out::AbstractArray`: Output matrix against wich reconstructions are
  compared. The last dimension represents each data sample.

# Optional Keyword Arguments
- `β::Float32=1.0f0`: Weighting factor for the KL-divergence term, used for
  annealing.
- `α::Float32=1.0f0`: Weighting factor for the mutual information term.
- `n_samples::Int=1`: The number of samples to draw from the latent space when
  computing the loss.
- `reconstruction_loglikelihood::Function=decoder_loglikelihood`: A function
  that computes the log likelihood of the decoder's output.
- `kl_divergence::Function=encoder_kl`: A function that computes the KL
  divergence between the encoder's output and the prior.
- `regularization::Union{Function, Nothing}=nothing`: A function that computes
  the regularization term based on the VAE outputs. Should return a Float32.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.
- `seed::Union{Nothing,Int}`: The seed for the random number generator used for
  shuffling the latent codes. If not provided, a random seed will be used.

# Returns
- `Float32`: The computed average loss value for the input `x` and its
  reconstructed counterparts over `n_samples` samples, including possible
  regularization terms and the mutual information constraint.

# Note
- This function takes the `vae` and `mi` instances of an InfoMaxVAE model as
  separate arguments to be able to compute a gradient only with respect to the
  `vae` parameters.
- Ensure that the input data `x` match the expected input dimensionality for the
  encoder in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function infomaxloss(
    vae::VAE,
    mi::MutualInfoChain,
    x_in::AbstractArray,
    x_out::AbstractArray;
    β=1.0f0,
    α=1.0f0,
    reconstruction_loglikelihood::Function=decoder_loglikelihood,
    kl_divergence::Function=encoder_kl,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0,
    seed::Union{Nothing,Int}=nothing
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_output = vae(x_in; latent=true)

    # Compute ⟨log p(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = StatsBase.mean(
        reconstruction_loglikelihood(
            x_out, vae_output.z, vae.decoder, vae_output.decoder
        )
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution p(z)
    kl_div = StatsBase.mean(
        kl_divergence(vae.encoder, vae_output.encoder)
    )

    # Permute latent codes for computation of mutual information
    z_shuffle = @ignore_derivatives shuffle_latent(vae_output.z, seed=seed)

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mi, x_in, vae_output.z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(vae_output) : 0.0f0

    # Compute loss function
    return -log_likelihood + β * kl_div - α * mutual_info +
           reg_strength * reg_term
end #function

# ==============================================================================

@doc raw"""
    miloss(
        vae::VAE,
        mi::MutualInfoChain,
        x::AbstractArray;
        regularization::Union{Function,Nothing}=nothing,
        reg_strength::Float32=1.0f0,
        seed::Union{Nothing,Int}=nothing
    )

Calculates the loss for training the MutualInfoChain in the InfoMaxVAE algorithm
to estimate mutual information between the input `x` and the latent
representation `z`. The loss function is based on a variational approximation of
mutual information, using the MutualInfoChain's output `g(x, z)`. The
variational mutual information is then calculated as the difference between the
MutualInfoChain's output for the true `x` and latent `z`, and the exponentiated
average of the MLP's output for `x` and the shuffled latent `z_shuffle`,
adjusted for the regularization term if provided.

# Arguments
- `vae::VAE`: The variational autoencoder.
- `mi::MutualInfoChain`: The MutualInfoChain used for estimating mutual
  information.
- `x::AbstractArray`: The input vector for the VAE.

# Optional Keyword Arguments
- `regularization::Union{Function, Nothing}=nothing`: A regularization function
  applied to the MLP's output.
- `reg_strength::Float32=1.0f0`: The strength of the regularization term.
- `seed::Union{Nothing,Int}=nothing`: The seed for the random number generator
  used for shuffling the latent codes. If not provided, a random seed will be
  used.

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
- This function takes the `vae` and `mi` instances of an InfoMaxVAE model as
  separate arguments to be able to compute a gradient only with respect to the
  `mi` parameters.
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
    expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
    information depends on shuffling the latent codes. This method works for
    large enough batches (≥ 64 samples).
"""
function miloss(
    vae::VAE,
    mi::MutualInfoChain,
    x::AbstractArray;
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0,
    seed::Union{Nothing,Int}=nothing
)
    # Forward Pass 
    vae_output = vae(x; latent=true)

    # Permute latent codes for computation of mutual information
    z_shuffle = @ignore_derivatives shuffle_latent(vae_output.z, seed=seed)

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mi, x, vae_output.z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute (negative) variational mutual information as loss function
    return -mutual_info + reg_strength * reg_term
end #function

# ==============================================================================
# InfoMaxVAE training functions
# ==============================================================================

@doc raw"""
        train!(
            infomaxvae, x, opt; 
            infomaxloss_function=infomaxloss,
            infomaxloss_kwargs, 
            miloss_function=miloss, 
            miloss_kwargs,
            loss_return::Bool=false,
            verbose::Bool=false
        )

Customized training function to update parameters of an InfoMax variational
autoencoder (VAE) given a loss function of the specified form.

The InfoMax VAE loss function can be defined as:

    loss_infoMax = argmin -⟨log p(x|z)⟩ + β Dₖₗ(qᵩ(z) || p(z)) -
                                     α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩],

where `⟨log p(x|z)⟩` is the expected log likelihood of the probabilistic
decoder, `Dₖₗ[qᵩ(z) || p(z)]` is the KL divergence between the approximated
encoder distribution and the prior over the latent space, and `g(x, z)` is the
output of the MutualInfoChain estimating the mutual information between the
input data and the latent representation.

This function simultaneously optimizes two neural networks: the VAE itself and a
multi-layer perceptron `MutualInfoChain` used to compute the mutual information
between input and latent variables.

# Arguments
- `infomaxvae::InfoMaxVAE`: Struct containing the elements of an InfoMax VAE.
- `x::AbstractArray`: Matrix containing the data on which to evaluate the loss
  function. Each column represents a single data point.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword arguments
- `infomaxloss_function::Function`: The loss function to be used during training
  for the VAE, defaulting to `infomaxloss`.
- `infomaxloss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to
  be passed to the VAE loss function.
- `miloss_function::Function`: The loss function to be used during training for
  the MLP computing the variational free energy, defaulting to `miloss`.
- `miloss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
  passed to the MutualInfoChain loss function.
- `loss_return::Bool`: If `true`, the function returns the loss values for the
  VAE and MutualInfoChain. Defaults to `false`.
- `verbose::Bool`: If `true`, the function prints the loss values for the VAE
  and MutualInfoChain. Defaults to `false`.

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function to jointly
train the VAE and MutualInfoChain. The VAE parameters are updated to minimize
the InfoMaxVAE loss, while the MutualInfoChain parameters are updated to
maximize the estimated mutual information. The function allows for customization
of loss hyperparameters during training.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works best for
  large enough batches (≥ 64 samples).
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractArray,
    opt::NamedTuple;
    infomaxloss_function::Function=infomaxloss,
    infomaxloss_kwargs::Union{NamedTuple,Dict}=Dict(),
    miloss_function::Function=miloss,
    miloss_kwargs::Union{NamedTuple,Dict}=Dict(),
    loss_return::Bool=false,
    verbose::Bool=false
)
    # == VAE == #
    # Compute gradient
    L_vae, ∇L_vae = Flux.withgradient(infomaxvae.vae) do vae
        infomaxloss_function(vae, infomaxvae.mi, x; infomaxloss_kwargs...)
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt.vae, infomaxvae.vae, ∇L_vae[1])

    # == MLP == #
    # Compute gradient
    L_mi, ∇L_mi = Flux.withgradient(infomaxvae.mi) do mi
        miloss_function(infomaxvae.vae, mi, x; miloss_kwargs...)
    end # do

    # Update the MutualInfoChain network parameters averaging gradient from all
    # datasets
    Flux.Optimisers.update!(opt.mi, infomaxvae.mi, ∇L_mi[1])

    # # Check if loss should be returned
    if loss_return
        return L_vae, L_mi
    end # if

    # Check if loss should be printed
    if verbose
        println("InfoMax Loss: ", L_vae)
        println("MutualInfoChain Loss: ", L_mi)
    end # if
end # function


# ==============================================================================

@doc raw"""
        train!(
            infomaxvae, x, opt; 
            infomaxloss_function=infomaxloss,
            infomaxloss_kwargs, 
            miloss_function=miloss, 
            miloss_kwargs,
            loss_return::Bool=false,
            verbose::Bool=false
        )

Customized training function to update parameters of an InfoMax variational
autoencoder (VAE) given a loss function of the specified form.

The InfoMax VAE loss function can be defined as:

    loss_infoMax = argmin -⟨log p(x|z)⟩ + β Dₖₗ(qᵩ(z) || p(z)) -
                                     α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩],

where `⟨log p(x|z)⟩` is the expected log likelihood of the probabilistic
decoder, `Dₖₗ[qᵩ(z) || p(z)]` is the KL divergence between the approximated
encoder distribution and the prior over the latent space, and `g(x, z)` is the
output of the MutualInfoChain estimating the mutual information between the
input data and the latent representation.

This function simultaneously optimizes two neural networks: the VAE itself and a
multi-layer perceptron `MutualInfoChain` used to compute the mutual information
between input and latent variables.

# Arguments
- `infomaxvae::InfoMaxVAE`: Struct containing the elements of an InfoMax VAE.
- `x::AbstractArray`: Matrix containing the data on which to evaluate the loss
  function. Each column represents a single data point.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword arguments
- `infomaxloss_function::Function`: The loss function to be used during training
  for the VAE, defaulting to `infomaxloss`.
- `infomaxloss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to
  be passed to the VAE loss function.
- `miloss_function::Function`: The loss function to be used during training for
  the MutualInfoChain computing the variational free energy, defaulting to
  `miloss`.
- `miloss_kwargs::Union{NamedTuple,Dict}`: Additional keyword arguments to be
  passed to the MutualInfoChain loss function.
- `loss_return::Bool`: If `true`, the function returns the loss values for the
  VAE and MLP. Defaults to `false`.

# Description
Performs one step of gradient descent on the InfoMaxVAE loss function to jointly
train the VAE and MutualInfoChain. The VAE parameters are updated to minimize
the InfoMaxVAE loss, while the MutualInfoChain parameters are updated to
maximize the estimated mutual information. The function allows for customization
of loss hyperparameters during training.

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works best for
  large enough batches (≥ 64 samples).
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x_in::AbstractArray,
    x_out::AbstractArray,
    opt::NamedTuple;
    infomaxloss_function::Function=infomaxloss,
    infomaxloss_kwargs::Union{NamedTuple,Dict}=Dict(),
    miloss_function::Function=miloss,
    miloss_kwargs::Union{NamedTuple,Dict}=Dict(),
    loss_return::Bool=false,
    verbose::Bool=false
)
    # == VAE == #
    # Compute gradient
    L_vae, ∇L_vae = Flux.withgradient(infomaxvae.vae) do vae
        infomaxloss_function(
            vae, infomaxvae.mi, x_in, x_out; infomaxloss_kwargs...
        )
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt.vae, infomaxvae.vae, ∇L_vae[1])

    # == MLP == #
    # Compute gradient
    L_mi, ∇L_mi = Flux.withgradient(infomaxvae.mi) do mi
        miloss_function(infomaxvae.vae, mi, x_in; miloss_kwargs...)
    end # do

    # Update the MutualInfoChain network parameters averaging gradient from all
    # datasets
    Flux.Optimisers.update!(opt.mi, infomaxvae.mi, ∇L_mi[1])

    # # Check if loss should be returned
    if loss_return
        return L_vae, L_mi
    end # if

    # Check if loss should be printed
    if verbose
        println("InfoMax Loss: ", L_vae)
        println("MutualInfoChain Loss: ", L_mi)
    end # if
end # function