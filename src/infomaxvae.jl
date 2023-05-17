# Import ML libraries
import Flux

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

##

# Import Abstract Types

using ..AutoEncode: AbstractAutoEncoder, AbstractVariationalAutoEncoder

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMax-VAE
# Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
# Information in Variational Autoencoders. in 2020 IEEE International Symposium
# on Information Theory (ISIT) 2729–2734 (IEEE, 2020).
# doi:10.1109/ISIT44484.2020.9174424.
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    `InfoMaxVAE`

Structure containing the components of an InfoMax variational autoencoder
(InfoMaxVAE).

> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
Information in Variational Autoencoders. in 2020 IEEE International Symposium on
Information Theory (ISIT) 2729–2734 (IEEE, 2020).
doi:10.1109/ISIT44484.2020.9174424.

# Fields
- `encoder::Flux.Chain`: neural network that takes the input and passes it
   through hidden layers.
- `µ::Flux.Dense`: Single layers that map from the encoder to the mean (`µ`) of
the latent variables distributions and 
- `logσ::Flux.Dense`: Single layers that map from the encoder to the log of the
   standard deviation (`logσ`) of the latent variables distributions.
- `decoder::Flux.Chain`: Neural network that takes the latent variables and
   tries to reconstruct the original input.
- `mlp::Flux.Chain`: Multi-layer perceptron (mlp) used to compute the mutual
  information between inputs and latent representations.
"""
mutable struct InfoMaxVAE <: AbstractVariationalAutoEncoder
    encoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
    decoder::Flux.Chain
    mlp::Flux.Chain
end

@doc raw"""
    `infomaxvae_init(
        n_input, 
        n_latent, 
        latent_activation, 
        output_activation,
        encoder, 
        encoder_activation,
        decoder, 
        decoder_activation,
        mlp, 
        mlp_activation,
        mlp_output_activation;
        init
    )`

Function to initialize an Info-Max autoencoder with `Flux.jl`. 

# Arguments
- `n_input::Int`: Dimension of input space.
- `n_latent::Int`: Dimension of latent space
- `latent_activation::Function`: Activation function coming in of the latent
  space layer.
- `output_activation::Function`: Activation function on the output layer
- `encoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the encoder network (one layer per entry).
- `encoder_activation::Vector`: Array containing the activation function for the
  encoder hidden layers. If `nothing` is given, no activation function is
  assigned to the layer. NOTE: length(encoder) must match
  length(encoder_activation).
- `decoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the decoder network (one layer per entry).
- `decoder_activation::Vector`: Array containing the activation function for the
  decoder hidden layers. If `nothing` is given, no activation function is
  assigned to the layer. NOTE: length(encoder) must match
  length(encoder_activation).

## Optional arguments
- `init::Function=Flux.glorot_uniform`: Function to initialize network
parameters.

# Returns
- a `struct` of type `InfoMaxVAE`
"""
function infomaxvae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    mlp::Vector{<:Int},
    mlp_activation::Vector{<:Function},
    mlp_output_activation::Function;
    init::Function=Flux.glorot_uniform
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{Flux.Dense}(undef, length(encoder))

    # Loop through layers   
    for i = 1:length(encoder)
        # Check if it is the first layer
        if i == 1
            # Set first layer from input to encoder with activation
            Encoder[i] = Flux.Dense(
                n_input => encoder[i], encoder_activation[i]; init=init
            )
        else
            # Set middle layers from input to encoder with activation
            Encoder[i] = Flux.Dense(
                encoder[i-1] => encoder[i], encoder_activation[i]; init=init
            )
        end # if
    end # for

    # Define layer that maps from encoder to latent space with activation
    Latent_µ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )
    Latent_logσ = Flux.Dense(
        encoder[end] => n_latent, latent_activation; init=init
    )

    # Initialize list with decoder layers
    Decoder = Array{Flux.Dense}(undef, length(decoder) + 1)

    # Add first layer from latent space to decoder
    Decoder[1] = Flux.Dense(
        n_latent => decoder[1], decoder_activation[1]; init=init
    )

    # Add last layer from decoder to output
    Decoder[end] = Flux.Dense(
        decoder[end] => n_input, output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(decoder) > 1
        # Loop through middle layers
        for i = 2:length(decoder)
            # Set middle layers of decoder
            Decoder[i] = Flux.Dense(
                decoder[i-1] => decoder[i], decoder_activation[i]; init=init
            )
        end # for
    end # if

    # Initialize list with mlp layers   
    MLP = Array{Flux.Dense}(undef, length(mlp) + 1)

    # Add initial layer
    MLP[1] = Flux.Dense(
        n_input + n_latent => mlp[1], mlp_activation[1]; init=init
    )

    # Add last layer
    MLP[end] = Flux.Dense(
        mlp[end] => 1, mlp_output_activation; init=init
    )

    # Check if there are multiple middle layers
    if length(mlp) > 1
        # Loop through middle layers
        for i = 2:length(mlp)
            # Set middle layers of decoder
            MLP[i] = Flux.Dense(
                mlp[i-1] => mlp[i], mlp_activation[i]; init=init
            )
        end # for
    end # if

    # Compile InfoMaxVAE
    return InfoMaxVAE(
        Flux.Chain(Encoder...),
        Latent_µ,
        Latent_logσ,
        Flux.Chain(Decoder...),
        Flux.Chain(MLP...)
    )
end # function

@doc raw"""
`recon(infomaxvae, input; latent)`

This function performs three steps:
1. passes an input `x` through the `encoder`, 
2. samples the latent variable by using the reparametrization trick,
3. reconstructs the input from the latent variables using the `decoder`.

# Arguments
- `infomaxvae::InfoMaxVAE`: InfoMax Variational autoencoder struct with all
  components.
- `input::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional Arguments
- `latent::Bool=false`: Boolean indicating if the latent variables should be
returned as part of the output or not.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
the input when mapped to the latent space.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
autoencoder. Note: This last point depends on a random sampling step, thus it
will change every time.
"""
function recon(
    vae::InfoMaxVAE,
    input::AbstractVecOrMat{Float32};
    latent::Bool=false
)
    # 1. Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(input)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(input)

    # 2. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
    ) .* exp.(logσ)

    # 3. Run sampled latent variables through decoder and return values
    if latent
        return z, vae.decoder(z)
    else
        return vae.decoder(z)
    end # if
end # function


# @doc raw"""
#     `infomax_loss(x, vae, mlp; σ, β, α, reconstruct, n_samples)`

# Loss function for the infoMax variational autoencoder. The loss function is
# defined as

# loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
#                α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

# infoMaxVAE simultaneously optimize two neural networks: the traditional
# variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
# the mutual information between input and latent variables.

# # Arguments
# - `x::AbstractVector{Float32}`: Input to the neural network.
# - `x_shuffle::Vector`: Shuffled input to the neural network needed to compute
#   the mutual information term. This term is used to obtain an encoding
#   `z_shuffle` that represents a random sample from the marginal P(z).
# - `vae::VAE`: Struct containing the elements of the variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#   between input and output.

# ## Optional arguments
# - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
# - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
# - `α::Float32=1`: Annealing inverse temperature for the mutual information term.
# - `n_samples::Int=1`: Number of samples to take from the latent space when
#   computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
#   over multiple latent-space samples due to the variability of each sample. In
#   practice, most algorithms keep this at 1.

# # Returns
# - `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
#   when compared with reconstructed output `x̂`. This is used by the training
#   algorithms to improve the reconstruction.
# """
# function infomax_loss(
#     x::AbstractVector{Float32},
#     x_shuffle::AbstractVector{Float32},
#     vae::VAE,
#     mlp::Flux.Chain;
#     σ::Float32=1.0f0,
#     β::Float32=1.0f0,
#     α::Float32=1.0f0,
#     n_samples::Int=1
# )
#     # Initialize arrays to save µ and logσ
#     µ = similar(Flux.params(vae.µ)[2])
#     logσ = similar(µ)

#     # Initialize value to save log probability
#     logP_x_z = 0.0f0

#     # Initialize value to save variational form of mutual information
#     info_x_z = 0.0f0

#     # Loop through latent space samples
#     for i = 1:n_samples
#         # Run input through reconstruct function
#         µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
#         # Run shuffled input through reconstruction function
#         _, _, z_shuffle, _ = vae_reconstruct(
#             x_shuffle, vae; latent=true
#         )

#         # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
#         logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
#                     1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

#         # Run input and latent variables through mutual information MLP
#         I_xz = first(mlp([x; z]))
#         # Run input and PERMUTED latent variables through mutual info MLP
#         I_xz_perm = first(mlp([x; z_shuffle]))
#         # Compute variational mutual information
#         info_x_z += I_xz - exp(I_xz_perm - 1)
#     end # for

#     # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
#     # and latent prior distribution P(x)
#     kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

#     # Compute loss function
#     return -logP_x_z / n_samples + β * kl_qₓ_p - α * info_x_z / n_samples

# end #function

# @doc raw"""
#     `infomax_loss(x, vae, mlp; σ, β, α, reconstruct, n_samples)`

# Loss function for the infoMax variational autoencoder. The loss function is
# defined as

# loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
#                α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

# infoMaxVAE simultaneously optimize two neural networks: the traditional
# variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
# the mutual information between input and latent variables. 

# # Arguments
# - `x::AbstractVector{Float32}`: Input to the neural network.
# - `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
#   needed to compute the mutual information term. This term is used to obtain an
#   encoding `z_shuffle` that represents a random sample from the marginal P(z).
# - `x_true::Vector`: True input against which to compare autoencoder
#   reconstruction.
# - `vae::VAE`: Struct containing the elements of the variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#   between input and output.

# ## Optional arguments
# - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
# - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
# - `α::Float32=1`: Annealing inverse temperature for the mutual information term.
# - `n_samples::Int=1`: Number of samples to take from the latent space when
#   computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
#   over multiple latent-space samples due to the variability of each sample. In
#   practice, most algorithms keep this at 1.

# # Returns
# - `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
#   when compared with reconstructed output `x̂`. This is used by the training
#   algorithms to improve the reconstruction.
# """
# function infomax_loss(
#     x::AbstractVector{Float32},
#     x_shuffle::AbstractVector{Float32},
#     x_true::AbstractVector{Float32},
#     vae::VAE,
#     mlp::Flux.Chain;
#     σ::Float32=1.0f0,
#     β::Float32=1.0f0,
#     α::Float32=1.0f0,
#     n_samples::Int=1
# )
#     # Initialize arrays to save µ and logσ
#     µ = similar(Flux.params(vae.µ)[2])
#     logσ = similar(µ)

#     # Initialize value to save log probability
#     logP_x_z = 0.0f0

#     # Initialize value to save variational form of mutual information
#     info_x_z = 0.0f0

#     # Loop through latent space samples
#     for i = 1:n_samples
#         # Run input through reconstruct function
#         µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
#         # Run shuffled input through reconstruction function
#         _, _, z_shuffle, _ = vae_reconstruct(
#             x_shuffle, vae; latent=true
#         )

#         # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
#         logP_x_z += -length(x) * (log(σ) + log(2π) / 2) -
#                     1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

#         # Run input and latent variables through mutual information MLP
#         I_xz = first(mlp([x; z]))
#         # Run input and PERMUTED latent variables through mutual info MLP
#         I_xz_perm = first(mlp([x; z_shuffle]))
#         # Compute variational mutual information
#         info_x_z += I_xz - exp(I_xz_perm - 1)
#     end # for

#     # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
#     # and latent prior distribution P(x)
#     kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

#     # Compute loss function
#     return -logP_x_z / n_samples + β * kl_qₓ_p - α * info_x_z / n_samples

# end #function

# @doc raw"""
#     `infomlp_loss(x, vae, mlp; n_samples)`

# Function used to train the multi-layered perceptron (mlp) used in the infoMaxVAE
# algorithm to estimate the mutual information between the input x and the latent
# space encoding z. The loss function is of the form

# Ixz_MLP = ⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩

# The mutual information is expressed in a variational form (optimizing over the
# space of all possible functions) where the MLP encodes the unknown optimal
# function g(x, z).

# # Arguments
# - `x::AbstractVector{Float32}`: Input to the neural network.
# - `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
#   needed to compute the mutual information term. This term is used to obtain an
#   encoding `z_shuffle` that represents a random sample from the marginal P(z).
# - `vae::VAE`: Struct containint the elements of the variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#   between input and output.

# ## Optional arguments
# - `n_samples::Int=1`: Number of samples to take from the latent space when
#   computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to average
#   over multiple latent-space samples due to the variability of each sample. In
#   practice, most algorithms keep this at 1.

# # Returns
# - `Ixz_MLP::Float32`: Variational mutual information between input x and latent
#   space encoding z.

# """
# function infomlp_loss(
#     x::AbstractVector{Float32},
#     x_shuffle::AbstractVector{Float32},
#     vae::VAE,
#     mlp::Flux.Chain;
#     n_samples::Int=1
# )
#     # Initialize arrays to save µ and logσ
#     µ = similar(Flux.params(vae.µ)[2])
#     logσ = similar(µ)

#     # Initialize value to save variational form of mutual information
#     info_x_z = 0.0f0

#     # Loop through latent space samples
#     for i = 1:n_samples
#         # Run input through reconstruct function
#         µ, logσ, z, x̂ = vae_reconstruct(x, vae; latent=true)
#         # Run shuffled input through reconstruction function
#         _, _, z_shuffle, _ = vae_reconstruct(
#             x_shuffle, vae; latent=true
#         )

#         # Run input and latent variables through mutual information MLP
#         I_xz = first(mlp([x; z]))
#         # Run input and PERMUTED latent variables through mutual info MLP
#         I_xz_perm = first(mlp([x; z_shuffle]))
#         # Compute variational mutual information
#         info_x_z += I_xz - exp(I_xz_perm - 1)
#     end # for

#     # Compute loss function
#     return -info_x_z / n_samples

# end #function

# @doc raw"""
#     `mutual_info_mlp(vae, mlp, data)`

# Function to compute the mutual information between the input `x` and the latent
# variable `z` for a given inforMaxVAE architecture.

# # Arguments
# - `vae::VAE`: Struct containint the elements of a variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#     between input and output.
# - `data::AbstractMatrix{Float32}`: Matrix containing the data on which to
#     evaluate the loss function. NOTE: Every column should represent a single
#     input.
# """
# function mutual_info_mlp(
#     vae::VAE, mlp::Flux.Chain, data::AbstractMatrix{Float32}
# )
#     # Generate list of random indexes for data shuffling
#     shuffle_idx = Random.shuffle(1:size(data, 2))

#     # Compute mutual information
#     return StatsBase.mean(
#         [-infomlp_loss(data[:, i], data[:, shuffle_idx[i]], vae, mlp)
#          for i = 1:size(data, 2)]
#     )

# end # function

# @doc raw"""
#     `infomaxvae_train!(vae, mlp, data, opt; kwargs...)`

# Customized training function to update parameters of infoMax variational
# autoencoder given a loss function of the form

# loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
#                α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

# infoMaxVAE simultaneously optimize two neural networks: the traditional
# variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
# the mutual information between input and latent variables.

# # Arguments
# - `vae::VAE`: Struct containint the elements of a variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#   between input and output.
# - `data::AbstractMatrix{Float32}`: Matrix containing the data on which to
#   evaluate the loss function. NOTE: Every column should represent a single
#   input.
# - `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
#   update the variational autoencoder parameters. This should be fed already with
#   the corresponding parameters. For example, one could feed: ⋅ Flux.AMSGrad(η)
# - `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
#   update the multi-layered perceptron parameters. This should be fed already
#   with the corresponding parameters. For example, one could feed: ⋅
#   Flux.AMSGrad(η)

# ## Optional arguments
# - `kwargs::NamedTuple`: Tuple containing arguments for the loss function. For
#     `infomax_loss`, for example, we have
#     - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
#     - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
#     - `α::Float32=1`: Annealing inverse temperature for the mutual information
#       term.
#     - `n_samples::Int=1`: Number of samples to take from the latent space when
#       computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to
#       average over multiple latent-space samples due to the variability of each
#       sample. In practice, most algorithms keep this at 1.
# """
# function infomaxvae_train!(
#     vae::VAE,
#     mlp::Flux.Chain,
#     data::AbstractMatrix{Float32},
#     vae_opt::Flux.Optimise.AbstractOptimiser,
#     mlp_opt::Flux.Optimise.AbstractOptimiser;
#     kwargs...
# )
#     # Extract VAE parameters
#     vae_params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)
#     # Extract MLP parameters
#     mlp_params = Flux.params(mlp)

#     # Generate list of random indexes for data shuffling
#     shuffle_idx = Random.shuffle(1:size(data, 2))

#     # Perform computation for first datum.
#     # NOTE: This is to properly initialize the object on which the gradient will
#     # be evaluated. There's probably better ways to do this, but this works.

#     # == VAE == #
#     # Evaluate the loss function and compute the gradient. Zygote.pullback
#     # gives two outputs: the result of the original function and a pullback,
#     # which is the gradient of the function.
#     vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
#         infomax_loss(data[:, 1], data[:, shuffle_idx[1]], vae, mlp; kwargs...)
#     end # do

#     # Having computed the pullback, we compute the loss function gradient
#     ∇vae_loss_ = vae_back_(one(vae_loss_))

#     # == MLP == #
#     # Evaluate the loss function and compute the gradient. Zygote.pullback
#     # gives two outputs: the result of the original function and a pullback,
#     # which is the gradient of the function.
#     mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
#         infomlp_loss(data[:, 1], data[:, shuffle_idx[1]], vae, mlp)
#     end # do

#     # Having computed the pullback, we compute the loss function gradient
#     ∇mlp_loss_ = mlp_back_(one(mlp_loss_))

#     # Loop through the rest of the datasets data
#     for (i, d) in enumerate(eachcol(data[:, 2:end]))
#         # == VAE == #
#         # Evaluate the loss function and compute the gradient. Zygote.pullback
#         # gives two outputs: the result of the original function and a pullback,
#         # which is the gradient of the function.
#         vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
#             infomax_loss(d, data[:, shuffle_idx[i]], vae, mlp; kwargs...)
#         end # do

#         # Having computed the pullback, we compute the loss function gradient
#         ∇vae_loss_ .+= vae_back_(one(vae_loss_))

#         # == MLP == #
#         # Evaluate the loss function and compute the gradient. Zygote.pullback
#         # gives two outputs: the result of the original function and a pullback,
#         # which is the gradient of the function.
#         mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
#             infomlp_loss(d, data[:, shuffle_idx[i]], vae, mlp)
#         end # do

#         # Having computed the pullback, we compute the loss function gradient
#         ∇mlp_loss_ .+= mlp_back_(one(mlp_loss_))
#     end # for

#     # Update the VAE network parameters averaging gradient from all datasets
#     Flux.Optimise.update!(vae_opt, vae_params, ∇vae_loss_ ./ size(data, 2))

#     # Update the MLP parameters averaging gradient from all datasets
#     Flux.Optimise.update!(mlp_opt, mlp_params, ∇mlp_loss_ ./ size(data, 2))

# end # function

# @doc raw"""
#     `infomaxvae_train!(vae, mlp, data, opt; kwargs...)`

# Customized training function to update parameters of infoMax variational
# autoencoder given a loss function of the form

# loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
#                α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

# infoMaxVAE simultaneously optimize two neural networks: the traditional
# variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
# the mutual information between input and latent variables. For this method, the
# data consists of a `Array{Float32, 3}` object, where the third dimension
# contains both the noisy data and the "real" value against which to compare the
# reconstruction.

# # Arguments
# - `vae::VAE`: Struct containint the elements of a variational autoencoder.
# - `mlp::Flux.Chain`: Multi-layered perceptron to compute mutual information
#   between input and output.
# - `data::Array{Float32, 3}`: Matrix containing the data on which to evaluate the
#   loss function. NOTE: Every column should represent a single input.
# - `vae_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
#   update the variational autoencoder parameters. This should be fed already with
#   the corresponding parameters. For example, one could feed: ⋅ Flux.AMSGrad(η)
# - `mlp_opt::Flux.Optimise.AbstractOptimiser`: Optimizing algorithm to be used to
#   update the multi-layered perceptron parameters. This should be fed already
#   with the corresponding parameters. For example, one could feed: ⋅
#   Flux.AMSGrad(η)

# ## Optional arguments
# - `kwargs::NamedTuple`: Tuple containing arguments for the loss function. For
#     `infomax_loss`, for example, we have
#     - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
#     - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
#     - `α::Float32=1`: Annealing inverse temperature for the mutual information
#       term.
#     - `n_samples::Int=1`: Number of samples to take from the latent space when
#       computing ⟨logP(x|z)⟩. NOTE: This should be increase if you want to
#       average over multiple latent-space samples due to the variability of each
#       sample. In practice, most algorithms keep this at 1.
# """
# function infomaxvae_train!(
#     vae::VAE,
#     mlp::Flux.Chain,
#     data::Array{Float32,3},
#     vae_opt::Flux.Optimise.AbstractOptimiser,
#     mlp_opt::Flux.Optimise.AbstractOptimiser;
#     kwargs...
# )
#     # Extract VAE parameters
#     vae_params = Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder)
#     # Extract MLP parameters
#     mlp_params = Flux.params(mlp)

#     # Split data and real value
#     data_noise = data[:, :, 1]
#     data_true = data[:, :, 2]

#     # Generate list of random indexes for data shuffling
#     shuffle_idx = Random.shuffle(1:size(data, 2))

#     # Perform computation for first datum.
#     # NOTE: This is to properly initialize the object on which the gradient will
#     # be evaluated. There's probably better ways to do this, but this works.

#     # == VAE == #
#     # Evaluate the loss function and compute the gradient. Zygote.pullback
#     # gives two outputs: the result of the original function and a pullback,
#     # which is the gradient of the function.
#     vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
#         infomax_loss(
#             data_noise[:, 1],
#             data_noise[:, shuffle_idx[1]],
#             data_true[:, 1],
#             vae,
#             mlp;
#             kwargs...
#         )
#     end # do

#     # Having computed the pullback, we compute the loss function gradient
#     ∇vae_loss_ = vae_back_(one(vae_loss_))

#     # == MLP == #
#     # Evaluate the loss function and compute the gradient. Zygote.pullback
#     # gives two outputs: the result of the original function and a pullback,
#     # which is the gradient of the function.
#     mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
#         infomlp_loss(data_noise[:, 1], data_noise[:, shuffle_idx[1]], vae, mlp)
#     end # do

#     # Having computed the pullback, we compute the loss function gradient
#     ∇mlp_loss_ = mlp_back_(one(mlp_loss_))

#     # Loop through the rest of the datasets data
#     for i = 2:size(data_noise, 2)
#         # == VAE == #
#         # Evaluate the loss function and compute the gradient. Zygote.pullback
#         # gives two outputs: the result of the original function and a pullback,
#         # which is the gradient of the function.
#         vae_loss_, vae_back_ = Zygote.pullback(vae_params) do
#             infomax_loss(
#                 data_noise[:, i],
#                 data_noise[:, shuffle_idx[i]],
#                 data_true[:, i],
#                 vae,
#                 mlp;
#                 kwargs...
#             )
#         end # do

#         # Having computed the pullback, we compute the loss function gradient
#         ∇vae_loss_ .+= vae_back_(one(vae_loss_))

#         # == MLP == #
#         # Evaluate the loss function and compute the gradient. Zygote.pullback
#         # gives two outputs: the result of the original function and a pullback,
#         # which is the gradient of the function.
#         mlp_loss_, mlp_back_ = Zygote.pullback(mlp_params) do
#             infomlp_loss(
#                 data_noise[:, i], data_noise[:, shuffle_idx[i]], vae, mlp
#             )
#         end # do

#         # Having computed the pullback, we compute the loss function gradient
#         ∇mlp_loss_ .+= mlp_back_(one(mlp_loss_))
#     end # for

#     # Update the VAE network parameters averaging gradient from all datasets
#     Flux.Optimise.update!(vae_opt, vae_params, ∇vae_loss_ ./ size(data, 2))

#     # Update the MLP parameters averaging gradient from all datasets
#     Flux.Optimise.update!(mlp_opt, mlp_params, ∇mlp_loss_ ./ size(data, 2))

# end # function