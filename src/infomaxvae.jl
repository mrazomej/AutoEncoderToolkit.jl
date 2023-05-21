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

using ..AutoEncode: AbstractAutoEncoder, AbstractVariationalAutoEncoder

using ..VAEs: VAE

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
    vae::VAE
    # encoder::Flux.Chain
    # µ::Flux.Dense
    # logσ::Flux.Dense
    # decoder::Flux.Chain
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
- `encoder::Vector{<:Int}`: Array containing the dimensions of the hidden layers
  for the encoder network (one layer per entry).
- `encoder_activation::Vector`: Array containing the activation function for the
  encoder hidden layers. NOTE: length(encoder) must match
  length(encoder_activation).
- `decoder::Vector{<:Int}`: Array containing the dimensions of the hidden layers
  for the decoder network (one layer per entry).
- `decoder_activation::Vector`: Array containing the activation function for the
  decoder hidden layers. NOTE: length(decoder) must match
  length(decoder_activation).
- `mlp::Vector{<:Int}`: Array containing the dimensions of the hidden layers for
  the multi-layer perceptron used to compute the mutual information between the
  latent representation and the data.
- `mlp_activation::Vector{<:Function}`: Array containing the activation function
  for the multi-layer perceptron used to compute the mutual information between
  the latent representation and the data.
- `mlp_output_activation::Function`: Activation function on the output layer for
  the multi-layer perceptron.

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
        VAE(
            Flux.Chain(Encoder...),
            Latent_µ,
            Latent_logσ,
            Flux.Chain(Decoder...)
        ),
        Flux.Chain(MLP...)
    )
end # function

@doc raw"""
    `loss(vae, mlp, x, x_shuffle; σ, β, α)`

Loss function for the infoMax variational autoencoder. The loss function is
defined as

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables.

# Arguments

NOTE: The input to the loss function splits the `InfoMaxVAE` into the `vae` and
`mlp` parts. This is not to compute gradients redundantly when training both
networks.

- `vae::VAE`: Struct containing the elements of the variational autoencoder.
- `mlp::Flux.Chain`: `Flux.jl` chain defining the multi-layered perceptron used
  to compute the variational mutual information.
- `x::AbstractVector{Float32}`: Input to the neural network.
- `x_shuffle::Vector`: Shuffled input to the neural network needed to compute
  the mutual information term. This term is used to obtain an encoding
  `z_shuffle` that represents a random sample from the marginal P(z).

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `α::Float32=1`: Annealing inverse temperature for the mutual information term.

# Returns
- `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
  when compared with reconstructed output `x̂`. This is used by the training
  algorithms to improve the reconstruction.
"""
function loss(
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractVecOrMat{Float32},
    x_shuffle::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, z, x̂ = vae(x; latent=true)
    # Run shuffle input through reconstruct function
    _, _, z_shuffle, _ = vae(x_shuffle; latent=true)

    # Initialize value to save variational form of mutual information
    info_x_z = 0.0f0

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Run input and latent variables through mutual information MLP
    I_xz = mlp([x; z])

    # Run input and PERMUTED latent variables through mutual info MLP
    I_xz_perm = mlp([x; z_shuffle])

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))


    # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
    # and latent prior distribution P(x)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z + β * kl_qₓ_p - α * info_x_z

end #function

@doc raw"""
    `loss(vae, mlp, x, x_true, x_shuffle; σ, β, α)`

Loss function for the infoMax variational autoencoder. The loss function is
defined as

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
               α [⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩].

infoMaxVAE simultaneously optimize two neural networks: the traditional
variational autoencoder (vae) and a multi-layered perceptron (mlp) to compute
the mutual information between input and latent variables. 

# Arguments

NOTE: The input to the loss function splits the `InfoMaxVAE` into the `vae` and
`mlp` parts. This is not to compute gradients redundantly when training both
networks.

- `vae::VAE`: Struct containing the elements of the variational autoencoder.
- `mlp::Flux.Chain`: `Flux.jl` chain defining the multi-layered perceptron used
  to compute the variational mutual information.
- `x::AbstractVector{Float32}`: Input to the neural network.
- `x_true::Vector`: True input against which to compare autoencoder
  reconstruction.
- `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
  needed to compute the mutual information term. This term is used to obtain an
  encoding `z_shuffle` that represents a random sample from the marginal P(z).

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
- `α::Float32=1`: Annealing inverse temperature for the mutual information term.

# Returns
- `loss_infoMax::Float32`: Single value defining the loss function for entry `x`
  when compared with reconstructed output `x̂`. This is used by the training
  algorithms to improve the reconstruction.
"""
function loss(
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32},
    x_shuffle::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0,
    α::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, z, x̂ = vae(x; latent=true)
    # Run shuffle input through reconstruct function
    _, _, z_shuffle, _ = vae(x_shuffle; latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

    # Run input and latent variables through mutual information MLP
    I_xz = mlp([x; z])

    # Run input and PERMUTED latent variables through mutual info MLP
    I_xz_perm = mlp([x; z_shuffle])

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute Kullback-Leibler divergence between approximated decoder qₓ(z)
    # and latent prior distribution P(x)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z + β * kl_qₓ_p - α * info_x_z

end #function

@doc raw"""
    `mlp_loss(vae, mlp, x, x_shuffle)`

Function used to train the multi-layered perceptron (mlp) used in the infoMaxVAE
algorithm to estimate the mutual information between the input x and the latent
space encoding z. The loss function is of the form

Ixz_MLP = ⟨g(x, z)⟩ - ⟨exp(g(x, z) - 1)⟩

The mutual information is expressed in a variational form (optimizing over the
space of all possible functions) where the MLP encodes the unknown optimal
function g(x, z).

# Arguments

NOTE: The input to the loss function splits the `InfoMaxVAE` into the `vae` and
`mlp` parts. This is not to compute gradients redundantly when training both
networks.

- `vae::VAE`: Struct containing the elements of the variational autoencoder.
- `mlp::Flux.Chain`: `Flux.jl` chain defining the multi-layered perceptron used
  to compute the variational mutual information.
- `x::AbstractVector{Float32}`: Input to the neural network.
- `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
  needed to compute the mutual information term. This term is used to obtain an
  encoding `z_shuffle` that represents a random sample from the marginal P(z).

# Returns
- `Ixz_MLP::Float32`: Variational mutual information between input x and latent
  space encoding z.

"""
function mlp_loss(
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractVecOrMat{Float32},
    x_shuffle::AbstractVecOrMat{Float32},
)
    # Run input through reconstruct function
    µ, logσ, z, x̂ = vae(x; latent=true)
    # Run shuffle input through reconstruct function
    _, _, z_shuffle, _ = vae(x_shuffle; latent=true)

    # Run input and latent variables through mutual information MLP
    I_xz = mlp([x; z])

    # Run input and PERMUTED latent variables through mutual info MLP
    I_xz_perm = mlp([x; z_shuffle])

    # Compute variational mutual information
    info_x_z = sum(@. I_xz - exp(I_xz_perm - 1))

    # Compute loss function
    return -info_x_z

end #function

@doc raw"""
    `mutual_info_mlp(vae, x)`

Function to compute the mutual information between the input `x` and the latent
variable `z` for a given inforMaxVAE architecture.

# Arguments
- `vae::InfoMaxVAE`: Struct containint the elements of an InfoMax variational
  autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to
    evaluate the loss function. NOTE: Every column should represent a single
    input.
"""
function mutual_info_mlp(infomaxvae::InfoMaxVAE, x::AbstractVecOrMat{Float32})
    # Compute mutual information
    return -mlp_loss(
        infomaxvae.vae, infomaxvae.mlp, x, x[:, Random.shuffle(1:size(x, 2))]
    ) / size(x, 2)
end # function

@doc raw"""
    `train!(vae, x, x_shuffle, vae_opt, mlp_opt; loss_kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
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
- `x_shuffle::AbstractVector{Float32}`: Shuffled input to the neural network
  needed to compute the mutual information term. This term is used to obtain an
  encoding `z_shuffle` that represents a random sample from the marginal P(z).
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
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractVecOrMat{Float32},
    x_shuffle::AbstractVecOrMat{Float32},
    vae_opt::NamedTuple,
    mlp_opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0, :α => 1.0f0)
)

    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss(vae, infomaxvae.mlp, x, x_shuffle; loss_kwargs...)
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

@doc raw"""
    `train!(vae, x, x_true, x_shuffle, vae_opt, mlp_opt; loss_kwargs...)`

Customized training function to update parameters of infoMax variational
autoencoder given a loss function of the form

loss_infoMax = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qₓ(z) || P(z)) - 
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
  encoding `z_shuffle` that represents a random sample from the marginal P(z).
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
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
    - `α::Float32=1`: Annealing inverse temperature for the mutual information
      term.
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32},
    x_shuffle::AbstractVecOrMat{Float32},
    vae_opt::NamedTuple,
    mlp_opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0, :α => 1.0f0)
)

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