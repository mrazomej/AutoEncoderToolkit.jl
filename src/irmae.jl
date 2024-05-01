# Import ML libraries
import Flux
import SimpleChains

##

# Import Abstract Types

using ..AutoEncoderToolkit: AbstractAutoEncoder, AbstractDeterministicAutoEncoder

##

@doc raw"""
    `IRMAE`

Structure containing the components of an Intrinsically Rank-Minimizing
Autoencoder (IRMAE).

# Fields
- `encoder::Flux.Chain`: neural network that takes the input and passes it
   through hidden layers.
- `linear::Flux.Chain`: Set of linear transformations that characterize the
  IRMAE architecture. These layers usually have no bias term.
- `decoder::Flux.Chain`: Neural network that takes the latent variables and
   tries to reconstruct the original input.
"""
mutable struct IRMAE <: AbstractDeterministicAutoEncoder
    encoder::Flux.Chain
    linear::Flux.Chain
    decoder::Flux.Chain
end

@doc raw"""
    `irmae_init(
        n_input, 
        n_latent, 
        latent_activation,
        output_activation
        encoder, 
        encoder_activation,
        decoder, 
        decoder_activation,
        n_linear;
        bias=false
    )`

Function to initialize an Implicit Rank-Minimizing Autoencoder neural network
with `Flux.jl`.

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
- `n_linear::Int`: Number of linear layers to be used in the IRMAE architecture
  for rank minimization.

## optional arguments
- `bias::Bool=false`: Whether or not to include the bias in the linear layers.
  Default is set to false as the original paper suggests these should be mere
  matrix multiplications.

# Returns
`IRMAE`: Struct containing the elements of an IRMAE.
"""
function irmae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    n_linear::Int;
    bias::Bool=false
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{Flux.Dense}(undef, length(encoder) + 1)

    # Loop through layers   
    for i = 1:length(encoder)
        # Check if it is the first layer
        if i == 1
            # Set first layer from input to encoder with activation
            Encoder[i] = Flux.Dense(
                n_input => encoder[i], encoder_activation[i]
            )
        else
            # Set middle layers from input to encoder with activation
            Encoder[i] = Flux.Dense(
                encoder[i-1] => encoder[i], encoder_activation[i]
            )
        end # if
    end # for
    # Add last layer from encoder to latent space with activation
    Encoder[end] = Flux.Dense(
        encoder[end] => n_latent, latent_activation
    )

    # Initialize list with encoder layers
    Decoder = Array{Flux.Dense}(undef, length(decoder) + 1)

    # Add first layer from latent space to decoder
    Decoder[1] = Flux.Dense(n_latent => decoder[1], decoder_activation[1])

    # Add last layer from decoder to output
    Decoder[end] = Flux.Dense(decoder[end] => n_input, output_activation)

    # Check if there are multiple middle layers
    if length(decoder) > 1
        # Loop through middle layers
        for i = 2:length(decoder)
            # Set middle layers of decoder
            Decoder[i] = Flux.Dense(
                decoder[i-1] => decoder[i], decoder_activation[i]
            )
        end # for
    end # if

    # Build linear layers between encoder and latent space
    IRM = repeat([Flux.Dense(n_latent => n_latent; bias=bias)], n_linear)

    # Compile encoder and decoder into single chain
    return IRMAE(
        Flux.Chain(Encoder...), Flux.Chain(IRM...), Flux.Chain(Decoder...)
    )
end # function

##

@doc raw"""
    `SimpleIRMAE`

`mutable struct` containing the components of an implicitly rank-minimizing
autoencoder defined for the `SimpleChains.jl` package.

# Fields
- `irmae::SimpleChains.SimpleChain`: Chain defining the autoencoder
- `param::DenseArray{Float32}`:
"""
mutable struct SimpleIRMAE <: AbstractDeterministicAutoEncoder
    irmae::SimpleChains.SimpleChain
    param::DenseArray{Float32}
end # struct


##

@doc raw"""
    `simple_irmae_init(
        n_input, 
        n_latent,
        latent_activation,
        output_activation,
        encoder, 
        encoder_activation,
        decoder, 
        decoder_activation,
        n_linear;
        bias=false
    )`

Function to initialize an autoencoder neural network with `SimpleChains.jl`.

# Arguments
- `n_input::Int`: Dimension of input space.
- `n_latent::Int`: Dimension of latent space
- `latent_activation::Function`: Activation function coming into the latent
  space layer.
- `output_activation::Function`: Activation function coming into the output
  layer.
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
- `n_linear::Int`: Number of linear layers to be used in the IRMAE architecture
  for rank minimization.

## optional arguments
- `bias::Bool=false`: Whether or not to include the bias in the linear layers.
  Default is set to false as the original paper suggests these should be mere
  matrix multiplications.

# Returns
`SimpleIRMAE`
"""
function simple_irmae_init(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function},
    n_linear::Int;
    bias::Bool=false
)
    # Check there's enough activation functions for all layers
    if (length(encoder_activation) != length(encoder)) |
       (length(decoder_activation) != length(decoder))
        error("Each layer needs exactly one activation function")
    end # if

    # Initialize list with encoder layers
    Encoder = Array{SimpleChains.TurboDense}(undef, length(encoder) + 1)

    # Loop through layers
    for i = 1:length(encoder)
        # Add layer
        Encoder[i] = SimpleChains.TurboDense(encoder[i], encoder_activation[i])
    end # for
    # Add last layer from encoder to latent space with activation
    Encoder[end] = SimpleChains.TurboDense(n_latent, latent_activation)

    # Initialize list with decoder layers
    Decoder = Array{SimpleChains.TurboDense}(undef, length(decoder))

    # Loop through layers
    for i = 1:(length(decoder))
        # Add layer
        Decoder[i] = SimpleChains.TurboDense(decoder[i], decoder_activation[i])
    end # for

    # Build linear layers between encoder and latent space
    IRM = repeat(
        [SimpleChains.TurboDense{bias}(n_latent, SimpleChains.identity)],
        n_linear
    )

    # Define autoencoder
    irmae = SimpleChains.SimpleChain(
        SimpleChains.static(n_input),
        Encoder...,
        IRM...,
        Decoder...,
        SimpleChains.TurboDense(output_activation, n_input)
    )
    # Initialize parameters
    param = SimpleChains.init_params(irmae)

    return SimpleIRMAE(irmae, param)
end # function

##

@doc raw"""
    `simple_to_flux(SimpleIRMAE, IRMAE)`

Function to transfer the parameters from a `SimpleChains.jl` trained network to
a `Flux.jl` network with the same architecture for downstream manipulation.

NOTE: This function is agnostic to the activation functions in the
`SimpleChains.jl` network from where `param` was extracted. Therefore, for this
transfer to make sense, you must make sure that both networks have the same
architecture!

# Arguments
- `SimpleIRMAE`: `mutable struct` defining an IRMAE with `SimpleChains.jl`.
  network.
- `ae::AE`: Struct defining an autoencoder.

# Returns
- `IRMAE`: IRMAE with same architecture but with modified parameters dictated by
  `param`.
"""
function simple_to_flux(sirmae::SimpleIRMAE, irmae::IRMAE)
    # Extract parameters
    param = sirmae.param

    # Concatenate autoencoder to single chain
    fluxchain = Flux.Chain(irmae.encoder..., irmae.linear..., irmae.decoder...)

    # Extract list of parameters from the Flux autoencoder. NOTE: This
    # extraction is done with a list comprehension over layers because Flux
    # collapses the parameters of all the linear chains with no bias into a
    # single set of parameters (given the matrix multiplication nature of these)
    # layers
    param_flux = [collect(Flux.params(layer)) for layer in fluxchain]
    # Initialize object where to transfer parameters
    param_transfer = deepcopy(param_flux)

    # Initialize parameter index counter to keep track of the already used
    # parameters
    idx = 1

    # Loop through layers
    for (i, layer) in enumerate(param_flux)
        # Loop through list of parameters in i-th layer
        for (j, p) in enumerate(layer)
            # Initialize object to save transferred parameters
            par = similar(p)
            # Extract parameters using the current index and the length of the
            # parameters
            par = param[idx:(idx+length(par)-1)]
            # Save parameter values with the correct shape
            param_transfer[i][j] = reshape(par, size(p))
            # Update index for next iteration
            idx += length(par)
        end # for
    end # for

    # Make parameter transfer a Flux.Params object
    param_transfer = Flux.Params(param_transfer)

    # Initialize list to save Flux.Dense layers that will later be converted
    # into a Flux.Chain
    layers_transfer = Array{Flux.Dense}(
        undef, length(param_transfer)
    )

    # Loop through layers in Flux.jl Chain
    for (i, layer) in enumerate(fluxchain)
        # Check if layer has bias
        if layer.bias == false
            # Generate Flux.Dense layer with weights and biases as the
            # SimpleChains network, and the activation function from the Flux
            # network
            layers_transfer[i] = Flux.Dense(
                param_transfer[i]..., layer.bias, layer.σ
            )
        else
            # Generate Flux.Dense layer with weights and biases as the
            # SimpleChains network, and the activation function from the Flux
            # network
            layers_transfer[i] = Flux.Dense(
                param_transfer[i]..., layer.σ
            )
        end # if
    end # for

    # Return Autoenconder with transferred parameters
    return IRMAE(
        Flux.Chain(layers_transfer[1:length(irmae.encoder)]...),
        Flux.Chain(
            layers_transfer[
                length(irmae.encoder)+1:length(irmae.encocder)+length(irmae.linear)
            ]...),
        Flux.Chain(
            layers_transfer[length(irmae.encocder)+length(irmae.linear)+1:end]...
        )
    )
end # function