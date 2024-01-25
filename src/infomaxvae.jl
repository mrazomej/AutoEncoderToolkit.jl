# Import ML libraries
import Flux
import Zygote

# Import basic math
import Random
import StatsBase
import Distributions

##

# Import Abstract Types

using ..AutoEncode: Float32Array, AbstractVariationalAutoEncoder,
    AbstractVariationalEncoder, AbstractVariationalDecoder,
    JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder, VAE

using ..VAEs: reparameterize, reconstruction_decoder,
    kl_encoder

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
- `x::AbstractVecOrMat{Float32}`: The data to be decoded. This can be a vector
  or a matrix where each column represents a separate sample.

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
    x::AbstractVecOrMat{Float32};
    prior::Distributions.Sampleable=Distributions.Normal{Float32}(0.0f0, 1.0f0),
    latent::Bool=false,
    n_samples::Int=1
)
    # Check if latent variables and mutual information should be returned
    if latent
        outputs = infomaxvae.vae(x, prior; latent=latent, n_samples=n_samples)

        # Compute mutual information estimate using MLP
        mutual_info = infomaxvae.mlp([x; outputs.z])

        # Add mutual_info to the NamedTuple
        outputs = merge(outputs, (mutual_info=mutual_info,))

        return outputs
    else
        # or return reconstructed data from decoder
        return infomaxvae.vae(x; prior, latent=false, n_samples=n_samples)
    end # if
end # function

# ==============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE loss function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ==============================================================================

# ==============================================================================
# Shuffle latent codes between data samples
# ==============================================================================

"""
    shuffle_z(data::AbstractMatrix{T}) where T <: Number

Shuffle the elements of the second dimension of a matrix.

# Arguments
- `z::AbstractMatrix{T}`: A matrix representing latent codes. Each column
  corresponds to a single latent code.

# Returns
- `AbstractMatrix{T}`: A new matrix with the second dimension shuffled.

# Examples
```julia
matrix = [1 2 3; 4 5 6; 7 8 9]
shuffled = shuffle_z(matrix)
```
"""
function shuffle_z(z::AbstractMatrix{T}) where {T<:Number}
    # Define shuffle indexes
    shuffled_indices = Random.shuffle(1:size(z, 2))
    # Return shiffled data
    return z[:, shuffled_indices]
end # function

"""
    `shuffle_z(data::AbstractArray{T, 3}) where T <: Number`

Shuffle the elements of the second dimension of a 3D tensor.

# Arguments
- `z::AbstractArray{T,3}`: A 3D representing latent codes. Each column in the
  2nd dimension corresponds to a single latent code. Each slice in the 3rd
  dimension corresponds to a sample from the resulting latent representation.

# Returns
- `AbstractArray{T, 3}`: A new 3D tensor with the second dimension shuffled.

# Examples

```julia
tensor = cat(1:9, 1:9, 1:9, dims = (1, 3, 3))
shuffled = shuffle_z(tensor)
```
"""
function shuffle_z(z::AbstractArray{T,3}) where {T<:Number}
    # Define shuffle indexes
    shuffled_indices = Random.shuffle(1:size(z, 2))
    # Return shiffled data
    return z[:, shuffled_indices, :]
end

# ==============================================================================
# Variational Mutual information
# ==============================================================================

"""
    variational_mutual_info(mlp, x, z, z_shuffle)

Compute a variational approximation of the mutual information between the input
`x` and the latent code `z` using a multilayer perceptron (MLP).

# Arguments
- `mlp::Flux.Chain`: A multilayer pereceptron, which is a neural network used
  to estimate mutual information.
- `x::AbstractMatrix{Float32}`: Matrix of input data where each column
  represents a data sample.
- `z::AbstractMatrix{Float32}`: Matrix of corresponding latent representations
  of the input data.
- `z_shuffle::AbstractMatrix{Float32}`: Matrix of latent representations where
  the second dimension has been shuffled.

# Returns
- `Float32`: An approximation of the mutual information between the input data
  and its corresponding latent representation.

# References
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).
"""
function variational_mutual_info(
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    z::AbstractMatrix{Float32},
    z_shuffle::AbstractMatrix{Float32},
)
    # Run input and real latent code through MPL
    I_xz = StatsBase.mean(mlp([x; z]))
    # Run input and shuffled latent code through MPL
    I_xz_perm = StatsBase.mean(exp.(mlp([x; z_shuffle]) .- 1))

    # Compute variational mutual information
    return I_xz - I_xz_perm
end # function

"""
    variational_mutual_info(mlp, x, z, z_shuffle)

Compute a variational approximation of the mutual information between the input
`x` and the latent code `z` using a multilayer perceptron (MLP).

# Arguments
- `mlp::Flux.Chain`: A multilayer pereceptron, which is a neural network used to
  estimate mutual information.
- `x::AbstractMatrix{Float32}`: Matrix of input data where each column
  represents a data sample.
- `z::AbstractArray{Float32,3}`: 3D tensor of latent representations where each
  "slice" along the third dimension corresponds to a different sample.  
- `z_shuffle::AbstractArray{Float32,3}`: 3D tensor of shuffled latent
  representations along the second dimension.
  

# Returns
- `Float32`: An approximation of the mutual information between the input data
  and its corresponding latent representation.

# References
> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).
"""
function variational_mutual_info(
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32},
    z::AbstractArray{Float32,3},
    z_shuffle::AbstractArray{Float32,3},
)
    # Run input and real latent code through MPL
    I_xz = StatsBase.mean(mlp.([Ref(x); eachslice(z, dims=3)]))
    # Run input and shuffled latent code through MPL
    I_xz_perm = StatsBase.mean(
        exp.(mlp.([Ref(x); eachslice(z_shuffle, dims=2)]) .- 1)
    )

    # Compute variational mutual information
    return I_xz - I_xz_perm
end # function

# ==============================================================================
# Loss InfoMax
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
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `mlp::Flux.Chain`: A multi-layer perceptron used to compute the mutual
  information term.
- `x::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation.

# Optional Keyword Arguments
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
    vae::VAE,
    mlp::Flux.Chain,
    x::AbstractMatrix{Float32};
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = reconstruction_decoder(
        vae.decoder,
        x,
        vae_outputs;
        n_samples=n_samples
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = kl_encoder(
        vae.encoder, x, vae_outputs; n_samples=n_samples
    )

    # Permute latent codes for computation of mutual information
    z_shuffle = Zygote.ignore() do
        shuffle_z(vae_outputs.z)
    end # do block

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mlp, x, vae_outputs.z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(vae_outputs) : 0.0f0

    # Compute loss function
    return -log_likelihood + β * kl_div - α * mutual_info +
           reg_strength * reg_term
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
- `vae::VAE`: A VAE model with encoder and decoder networks.
- `mlp::Flux.Chain`: A multilayer perceptron for approximating the mutual
  information between inputs and latent variables.
- `x_in::AbstractMatrix{Float32}`: Input matatrix. Every column represents an
  observation input to the encoder.
- `x_out::AbstractMatrix{Float32}`: Target output Matrix for the decoder.

# Optional Keyword Arguments
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
    vae::VAE,
    mlp::Flux.Chain,
    x_in::AbstractMatrix{Float32},
    x_out::AbstractMatrix{Float32};
    β::Float32=1.0f0,
    α::Float32=1.0f0,
    n_samples::Int=1,
    regularization::Union{Function,Nothing}=nothing,
    reg_strength::Float32=1.0f0
)
    # Forward Pass (run input through reconstruct function with n_samples)
    vae_outputs = vae(x_in; latent=true, n_samples=n_samples)

    # Compute ⟨log π(x|z)⟩ for a Gaussian decoder averaged over all samples
    log_likelihood = reconstruction_decoder(
        vae.decoder,
        x_out,
        vae_outputs;
        n_samples=n_samples
    )

    # Compute Kullback-Leibler divergence between approximated decoder qᵩ(z|x)
    # and latent prior distribution π(z)
    kl_div = kl_encoder(
        vae.encoder, x_in, vae_outputs; n_samples=n_samples
    )

    # Permute latent codes for computation of mutual information
    z_shuffle = Zygote.ignore() do
        shuffle_z(vae_outputs.z)
    end # do block

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mlp, x_in, vae_outputs.z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(vae_outputs) : 0.0f0

    # Compute loss function
    return -log_likelihood + β * kl_div - α * mutual_info +
           reg_strength * reg_term
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
    # Forward Pass (run input through reconstruct function with n_samples)
    outputs = vae(x; latent=true, n_samples=n_samples)

    # Extract relevant variable
    z = outputs.z

    # Permute latent codes for computation of mutual information
    z_shuffle = Zygote.ignore() do
        shuffle_z(z)
    end # do block

    # Compute variational mutual information
    mutual_info = variational_mutual_info(mlp, x, z, z_shuffle)

    # Compute regularization term if regularization function is provided
    reg_term = (regularization !== nothing) ? regularization(outputs) : 0.0f0

    # Compute (negative) variational mutual information as loss function
    return -mutual_info + reg_strength * reg_term
end #function

# ==============================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# InfoMaxVAE training functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# ==============================================================================

"""
    `train!(infomaxvae, x, opt_vae, opt_mlp; loss_function=loss, loss_kwargs,
            mlp_loss_function, mlp_loss_kwargs)`

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
- `x::AbstractMatrix{Float32}`: Matrix containing the data on which to evaluate
  the loss function. Each column represents a single data point.
- `opt_vae::NamedTuple`: State of the optimizer for updating the VAE parameters.
    Typically initialized using `Flux.Train.setup`.
- `opt_mlp::NamedTuple`: State of the optimizer for updating the MLP parameters.
  Typically initialized using `Flux.Train.setup`.

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

# Notes
- Ensure that the dimensionality of the input data `x` aligns with the encoder's
  expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
"""
function train!(
    infomaxvae::InfoMaxVAE,
    x::AbstractMatrix{Float32},
    opt_vae::NamedTuple,
    opt_mlp::NamedTuple;
    loss_function::Function=loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    mlp_loss_function::Function=mlp_loss,
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss_function(vae, infomaxvae.mlp, x; loss_kwargs...)
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss_function(infomaxvae.vae, mlp, x; mlp_loss_kwargs...)
    end # do

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
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
- `opt_vae::NamedTuple`: State of the optimizer for updating the VAE parameters.
  Typically initialized using `Flux.Train.setup`.
- `opt_mlp::NamedTuple`: State of the optimizer for updating the MLP parameters.
  Typically initialized using `Flux.Train.setup`.

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

# Notes
- Ensure that the dimensionality of the input data `x_in` and `x_out` aligns
  with the encoder's expected input in the VAE.
- InfoMaxVAEs fully depend on batch training as the estimation of mutual
  information depends on shuffling the latent codes. This method works for large
  enough batches (≥ 64 samples).
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
    mlp_loss_kwargs::Union{NamedTuple,Dict}=Dict()
)
    # == VAE == #
    # Compute gradient
    ∇vae_loss_ = Flux.gradient(infomaxvae.vae) do vae
        loss_function(
            vae, infomaxvae.mlp, x_in, x_out; loss_kwargs...
        )
    end # do

    # Update the VAE network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_vae, infomaxvae.vae, ∇vae_loss_[1])

    # == MLP == #
    # Compute gradient
    ∇mlp_loss_ = Flux.gradient(infomaxvae.mlp) do mlp
        mlp_loss_function(
            infomaxvae.vae, mlp, x_in; mlp_loss_kwargs...
        )
    end # do

    # Update the MLP network parameters averaging gradient from all datasets
    Flux.Optimisers.update!(opt_mlp, infomaxvae.mlp, ∇mlp_loss_[1])
end # function