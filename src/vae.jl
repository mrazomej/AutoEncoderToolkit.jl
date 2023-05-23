# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

# Import GPU library
import CUDA

##

# Import Abstract Types

using ..AutoEncode: AbstractAutoEncoder, AbstractVariationalAutoEncoder

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
#    http://arxiv.org/abs/1312.6114 (2014).
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `VAE`

Structure containing the components of a variational autoencoder (VAE).

# Fields
- `encoder::Flux.Chain`: neural network that takes the input and passes it
   through hidden layers.
- `µ::Flux.Dense`: Single layers that map from the encoder to the mean (`µ`) of
the latent variables distributions and 
- `logσ::Flux.Dense`: Single layers that map from the encoder to the log of the
   standard deviation (`logσ`) of the latent variables distributions.
- `decoder::Flux.Chain`: Neural network that takes the latent variables and
   tries to reconstruct the original input.
"""
mutable struct VAE <: AbstractVariationalAutoEncoder
    encoder::Flux.Chain
    µ::Flux.Dense
    logσ::Flux.Dense
    decoder::Flux.Chain
end

@doc raw"""
    `vae_init(
        n_input, 
        n_latent, 
        latent_activation,
        output_activation,
        encoder, 
        encoder_activation,
        decoder, 
        decoder_activation;
        init
    )`

Function to initialize a variational autoencoder neural network with `Flux.jl`.

# Arguments
- `n_input::Int`: Dimension of input space.
- `n_latent::Int`: Dimension of latent space
- `latent_activation::Function`: Activation function coming in of the latent
  space layer.
- `output_activation::Function`: Activation function on the output layer
- `encoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the encoder network (one layer per entry).
- `encoder_activation::Vector`: Array containing the activation function for the
  encoder hidden layers. NOTE: length(encoder) must match
  length(encoder_activation).
- `decoder::Vector{Int}`: Array containing the dimensions of the hidden layers
  of the decoder network (one layer per entry).
- `decoder_activation::Vector`: Array containing the activation function for the
  decoder hidden layers. NOTE: length(decoder) must match
  length(decoder_activation).

## Optional arguments
- `init::Function=Flux.glorot_uniform`: Function to initialize network
parameters.

# Returns
- a `struct` of type `VAE`
"""
function vae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function};
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

    # Compile encoder and decoder into single chain
    return VAE(
        Flux.Chain(Encoder...), Latent_µ, Latent_logσ, Flux.Chain(Decoder...)
    )
end # function

@doc raw"""
`VAE(input; latent)`

This function performs three steps:
1. passes an input `x` through the `encoder`, 
2. samples the latent variable by using the reparametrization trick,
3. reconstructs the input from the latent variables using the `decoder`.

# Arguments
- `input::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional Arguments
- `latent::Bool=true`: Boolean indicating if the parameters of the latent
  representation (mean `µ`, log standard deviation `logσ`) should be returned.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
the input when mapped to the latent space.
- `z::Vector{Float32}`: Random sample of latent space code for the input. NOTE:
  This will change every time the function is called on the same input because
  of the random number sampling.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
  autoencoder. NOTE: This will change every time the function is called on the
  same input because of the random number sampling.
"""
function (vae::VAE)(
    input::AbstractVecOrMat{Float32};
    latent::Bool=false
)

    # 1. Map input through encoder layer
    encode_input = vae.encoder(input)

    # 2. map encoder layer output to mean and log standard deviation of latent
    #    variables
    µ = vae.µ(encode_input)
    logσ = vae.logσ(encode_input)

    # 3. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), size(µ)...
    ) .* exp.(logσ)

    # 4. Run sampled latent variables through decoder and return values
    if latent
        return µ, logσ, z, vae.decoder(z)
    else
        return vae.decoder(z)
    end # if
end # function

# Mark function as Flux.Functors.@functor so that Flux.jl allows for training
Flux.@functor VAE

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE loss functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `loss(vae, x; σ, β)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || P(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z) encodes the
function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f̲(x), σI).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    vae::VAE,
    x::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z + β * kl_qₓ_p

end #function

@doc raw"""
    `loss(vae, x, x_true; σ, β)`

Loss function for the variational autoencoder. The loss function is defined as

loss = argmin -⟨log P(x|z)⟩ + β Dₖₗ(qᵩ(z | x) || P(z)),

where the minimization is taken over the functions f̲, g̲, and h̲̲. f̲(z)
encodes the function that defines the mean ⟨x|z⟩ of the decoder P(x|z), i.e.,

    P(x|z) = Normal(f̲(x), σI).

g̲ and h̲̲ define the mean and covariance of the approximate decoder qᵩ(z|x),
respectively, i.e.,

    P(z|x) ≈ qᵩ(z|x) = Normal(g̲(x), h̲̲(x)).

NOTE: This method accepts an extra argument `x_true` as the ground truth against
which to compare the input values that is not necessarily the same as the input
value.

# Arguments
- `x::AbstractVecOrMat{Float32}`: Input to the neural network.
- `x_true::AbstractVecOrMat{Float32}`: True input against which to compare
  autoencoder reconstruction.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return -logP_x_z + β * kl_qₓ_p

end #function

@doc raw"""
    `kl_div(x, vae)`

Function to compute the KL divergence between the approximate encoder qₓ(z) and
the latent variable prior distribution P(z). Since we assume
        P(z) = Normal(0̲, 1̲),
and
        qₓ(z) = Normal(f̲(x̲), σI̲̲),
the KL divergence has a closed form

# Arguments
- `x::AbstractVector{Float32}`: Input to the neural network.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.        

# Returns
Dₖₗ(qₓ(z)||P(z))
"""
function kl_div(x::AbstractVector{Float32}, vae::VAE)
    # Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(x)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(x)

    return sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)
end # function

@doc raw"""
    `loss_terms(vae, x; σ, β)`

Loss function for the variational autoencoder. NOTE: This function performs the
same computations as the `loss` function, but simply returns each term
individually.

# Arguments
- `vae::VAE`: Struct containint the elements of the variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Input to the neural network.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss_terms(
    vae::VAE,
    x::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return [logP_x_z, β * kl_qₓ_p]
end #function

@doc raw"""
    `loss_terms(vae, x, x_true; σ, β)`

Loss function for the variational autoencoder. NOTE: This function performs the
same computations as the `loss` function, but simply returns each term
individually.

# Arguments
- `x::AbstractVecOrMat{Float32}`: Input to the neural network.
- `x_true::AbstractVecOrMat{Float32}`: True input against which to compare
  autoencoder reconstruction.
- `vae::VAE`: Struct containint the elements of the variational autoencoder.

## Optional arguments
- `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
- `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.

# Returns
- `loss::Float32`: Single value defining the loss function for entry `x` when
compared with reconstructed output `x̂`.
"""
function loss_terms(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32};
    σ::Float32=1.0f0,
    β::Float32=1.0f0
)
    # Run input through reconstruct function
    µ, logσ, _, x̂ = vae(x; latent=true)

    # Compute ⟨log P(x|z)⟩ for a Gaussian decoder
    logP_x_z = -length(x) * (log(σ) + log(2π) / 2) -
               1 / (2 * σ^2) * sum((x_true .- x̂) .^ 2)

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution P(z)
    kl_qₓ_p = sum(@. (exp(2 * logσ) + μ^2 - 1.0f0) / 2.0f0 - logσ)

    # Compute loss function
    return [logP_x_z, β * kl_qₓ_p]
end #function

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# VAE training functions
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@doc raw"""
    `train!(vae, x, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Matrix containing the data on which to evaluate
  the loss function. NOTE: Every column should represent a single input.
- `opt::NamedTuple`: State of optimizer that will be used to update parameters.
  NOTE: This is in agreement with `Flux.jl ≥ 0.13` where implicit `Zygote`
  gradients are not allowed. This `opt` object can be initialized using 
  `Flux.Train.setup`. For example, one can run
  ```
  opt_state = Flux.Train.setup(Flux.Optimisers.Adam(1E-1), vae)
  ```

## Optional arguments
- `loss_kwargs::Union{NamedTuple,Dict}`: Tuple containing arguments for the loss
    function. For `loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0,)
)
    # Compute gradient
    ∇loss_ = Flux.gradient(vae) do vae
        loss(vae, x; loss_kwargs...)
    end # do

    # Update the network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        opt,
        vae,
        ∇loss_[1]
    )
end # function

@doc raw"""
    `train!(vae, x, opt; kwargs...)`

Customized training function to update parameters of variational autoencoder
given a loss function.

# Arguments
- `vae::VAE`: Struct containint the elements of a variational autoencoder.
- `x::AbstractVecOrMat{Float32}`: Array containing the data to be ran through
  the VAE. NOTE: Every column should represent a single input.
- `x_true;:AbstractVecOrMat{Float32}`: Array containing the data used to compare
  the reconstruction for the loss function. This can be used to train denoising
  VAE, for exmaple.
- `opt::NamedTuple`: State of optimizer that will be used to update parameters.
  NOTE: This is in agreement with `Flux.jl ≥ 0.13` where implicit `Zygote`
  gradients are not allowed. This `opt` object can be initialized using
  `Flux.Train.setup`. For example, one can run
  ```
  opt_state = Flux.Train.setup(Flux.Optimisers.Adam(1E-1), vae)
  ```

## Optional arguments
- `loss_kwargs::Union{NamedTuple,Dict}`: Tuple containing arguments for the loss
    function. For `loss`, for example, we have
    - `σ::Float32=1`: Standard deviation of the probabilistic decoder P(x|z).
    - `β::Float32=1`: Annealing inverse temperature for the KL-divergence term.
"""
function train!(
    vae::VAE,
    x::AbstractVecOrMat{Float32},
    x_true::AbstractVecOrMat{Float32},
    opt::NamedTuple;
    loss_kwargs::Union{NamedTuple,Dict}=Dict(:σ => 1.0f0, :β => 1.0f0,)
)
    # Compute gradient
    ∇loss_ = Flux.gradient(vae) do vae
        loss(vae, x, x_true; loss_kwargs...)
    end # do

    # Update the network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(
        opt,
        vae,
        ∇loss_[1]
    )
end # function