@doc raw"""
    SimpleAE

`mutable struct` representing a basic autoencoder (AE) specifically designed for
the `SimpleChains.jl` package.

# Fields
- `AE::SimpleChains.SimpleChain`: The autoencoder model defined as a sequence of
  layers and operations using the `SimpleChain` construct from the
  `SimpleChains.jl` package. This chain encompasses both the encoder and decoder
  parts of the AE.
- `param::DenseArray{Float32}`: An array of model parameters, presumably to aid
  in training or other operations. The exact nature and organization of these
  parameters depend on the AE's design and requirements.

# Description
This struct encapsulates the components of a basic autoencoder model suitable
for use with the `SimpleChains.jl` framework. While the autoencoder's
architecture is defined in the `AE` field, associated parameters, possibly used
for optimization routines or other tasks, are stored in the `param` field.

# Note
Users should ensure compatibility between the `SimpleAE` structure and
functions/methods provided by `SimpleChains.jl` when integrating or extending
this structure.
"""
mutable struct SimpleAE <: AbstractDeterministicAutoEncoder
    AE::SimpleChains.SimpleChain
    param::DenseArray{Float32}
end # struct


##

@doc raw"""
    SimpleAE(
        n_input::Int, 
        n_latent::Int, 
        latent_activation::Function,
        output_activation::Function,
        encoder::Vector{Int}, 
        encoder_activation::Vector{Function},
        decoder::Vector{Int}, 
        decoder_activation::Vector{Function}
    ) -> SimpleAE

Constructs and initializes a `SimpleAE` autoencoder using components from the
`SimpleChains.jl` package.

# Arguments
- `n_input::Int`: Dimensionality of the input space.
- `n_latent::Int`: Dimensionality of the latent space.
- `latent_activation::Function`: Activation function applied at the latent space
  layer.
- `output_activation::Function`: Activation function applied at the output
  layer.
- `encoder::Vector{Int}`: Specifies the dimensions of each hidden layer in the
  encoder network.
- `encoder_activation::Vector{Function}`: Activation functions for each hidden
  layer in the encoder. The number of functions should match the length of
  `encoder`.
- `decoder::Vector{Int}`: Specifies the dimensions of each hidden layer in the
  decoder network.
- `decoder_activation::Vector{Function}`: Activation functions for each hidden
  layer in the decoder. The number of functions should match the length of
  `decoder`.

# Returns
- A `SimpleAE` instance representing the constructed autoencoder.

# Description
The function creates a `SimpleAE` autoencoder by sequentially combining the
provided encoder and decoder layers, while ensuring the correct matching of
activation functions for each layer. The resultant autoencoder is structured to
operate seamlessly within the `SimpleChains.jl` ecosystem.

# Example
```julia
ae_model = SimpleAE(
    784, 32, relu, sigmoid, [512, 256], [relu, relu], [256, 512], [relu, relu]
)
```

# Note
Ensure that the number of dimensions in encoder matches the number of activation
functions in encoder_activation, and similarly for decoder and
decoder_activation.
"""
function SimpleAE(
    n_input::Int,
    n_latent::Int,
    latent_activation::Function,
    output_activation::Function,
    encoder::Vector{<:Int},
    encoder_activation::Vector{<:Function},
    decoder::Vector{<:Int},
    decoder_activation::Vector{<:Function}
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

    # Define autoencoder
    ae = SimpleChains.SimpleChain(
        SimpleChains.static(n_input),
        Encoder...,
        Decoder...,
        SimpleChains.TurboDense(output_activation, n_input)
    )
    # Initialize parameters
    param = SimpleChains.init_params(ae)

    return SimpleAE(ae, param)
end # function

##

@doc raw"""
    `simple_to_flux(simple_ae::SimpleAE, ae::AE)`

Function to transfer the parameters from a `SimpleChains.jl` trained network to
a `Flux.jl` network with the same architecture for downstream manipulation.

NOTE: This function is agnostic to the activation functions in the
`SimpleChains.jl` network from where `param` was extracted. Therefore, for this
transfer to make sense, you must ensure that both networks have the same
architecture!

# Arguments
- `simple_ae::SimpleAE`: A `SimpleAE` instance representing an autoencoder built
  using `SimpleChains.jl`.
- `ae::AE`: An `AE` instance representing an autoencoder built using `Flux.jl`.

# Returns
- `AE`: The `Flux.jl` autoencoder (`AE`) with the modified parameters dictated
  by the `SimpleAE`.
"""
function simple_to_flux(simple_ae::SimpleAE, ae::AE)
    # Extract parameters from the SimpleChains.jl autoencoder
    param = simple_ae.param

    # Concatenate the encoder and decoder chains from the Flux.jl autoencoder
    # into a single chain for easier parameter extraction
    fluxchain = Flux.Chain(ae.encoder..., ae.decoder...)

    # Extract individual layer parameters from the Flux.jl chain. This creates a
    # nested list where each entry is a list of parameters for a specific layer
    # (usually [weights, biases])
    param_flux = [collect(Flux.params(layer)) for layer in fluxchain]

    # Deepcopy the extracted parameters to create a container for transferred 
    # parameters. This avoids altering the original parameters.
    param_transfer = deepcopy(param_flux)

    # Initialize a counter to keep track of the index position in the
    # SimpleChains parameter vector
    idx = 1

    # Loop through each layer of parameters in the Flux.jl model
    for (i, layer) in enumerate(param_flux)
        # Within each layer, loop through the set of parameters (usually weights 
        # and biases)
        for (j, p) in enumerate(layer)
            # Create a container with the same shape as the current parameter
            # set
            par = similar(p)
            # Transfer the parameters from the SimpleChains model into this
            # container
            par .= param[idx:(idx+length(par)-1)]
            # Store the reshaped transferred parameters in the appropriate slot
            # in the parameter transfer container
            param_transfer[i][j] = reshape(par, size(p))
            # Update the index counter to move to the next set of parameters in
            # the SimpleChains parameter vector
            idx += length(par)
        end # for
    end # for

    # Convert the list of transferred parameters into a Flux.Params object for 
    # compatibility with Flux.jl
    param_transfer = Flux.Params(param_transfer)

    # Create a container to hold the Flux.Dense layers that will be constructed
    # with the transferred parameters
    layers_transfer = Array{Flux.Dense}(undef, length(param_transfer))

    # Loop through each layer in the original Flux.jl Chain to construct new
    # layers using the transferred parameters
    for (i, layer) in enumerate(fluxchain)
        if layer.bias == false
            # Construct a new Flux.Dense layer with no bias using the
            # transferred parameters and the same activation function as the
            # original layer
            layers_transfer[i] = Flux.Dense(
                param_transfer[i]..., layer.bias, layer.σ
            )
        else
            # Construct a new Flux.Dense layer with bias using the transferred
            # parameters and the same activation function as the original layer
            layers_transfer[i] = Flux.Dense(param_transfer[i]..., layer.σ)
        end # if
    end # for

    # Return a new AE autoencoder constructed using the Encoder and Decoder
    # structs and the layers with transferred parameters
    return AE(
        Encoder(layers_transfer[1:length(ae.encoder)]...),
        Decoder(layers_transfer[length(ae.encoder)+1:end]...)
    )
end # function
