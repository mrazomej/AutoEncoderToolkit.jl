# AutoEncode.jl

Welcome to the `AutoEncode.jl` documentation. This package provides a simple
interface for training and using [Flux.jl](https://fluxml.ai)-based autoencoders
and variational autoencoders in Julia.

## Installation

You can install `AutoEncode.jl` using the Julia package manager. From the Julia
REPL, type `]` to enter the Pkg REPL mode and run:

```julia-repl
add AutoEncode
```

## Design

The idea behind `AutoEncode.jl` is to take advantage of Julia's multiple
dispatch to provide a simple and flexible interface for training and using
different types of autoencoders. The package is designed to be modular and allow
the user to easily define and test custom encoder and decoder architectures.
Moreover, when it comes to variational autoencoders, `AutoEncode.jl` takes a
probabilistic perspective, where the type of encoders and decoders defines (via
multiple dispatch) the corresponding distribution used within the corresponding
loss function.

For example, assume you want to train a variational autoencoder with
convolutional layers in the encoder and deconvolutional layers in the decoder on
the [`MNIST`](https://en.wikipedia.org/wiki/MNIST_database) dataset. You can
easily do this as follows:

Let's begin by defining the encoder. For this, we will use the `JointLogEncoder`
type, which is a simple encoder that takes a `Flux.Chain` for the shared layers
between the mean and log-variance layers and two `Flux.Dense` (or `Flux.Chain`)
layers for the last layers of the encoder.

```julia
# Define dimensionality of latent space
n_latent = 2

# Define number of initial channels
n_channels_init = 128

# Define convolutional layers
conv_layers = Flux.Chain(
    # First convolutional layer
    Flux.Conv((3, 3), 1 => n_channels_init, Flux.relu; stride=2, pad=1),
    # Second convolutional layer
    Flux.Conv(
        (3, 3), n_channels_init => n_channels_init * 2, Flux.relu;
        stride=2, pad=1
    ),
    # Flatten the output
    AutoEncode.Flatten()
)

# Define layers for µ and log(σ)
µ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)
logσ_layer = Flux.Dense(n_channels_init * 2 * 7 * 7, n_latent, Flux.identity)

# build encoder
encoder = AutoEncode.JointLogEncoder(conv_layers, µ_layer, logσ_layer)
```

!!! note
    The `Flatten` layer is a custom layer defined in `AutoEncode.jl` that
    flattens the output into a 1D vector. This flattening operation is necessary
    because the output of the convolutional layers is a 4D tensor, while the
    input to the `µ` and `log(σ)` layers is a 1D vector. The custom layer is 
    needed to be able to save the model and load it later as `BSON` and `JLD2`
    do not play well with anonymous functions.

For the decoder, given the binary nature of the `MNIST` dataset, we expect the
output to be a `Bernoulli` distribution. We can define the decoder as follows:

```julia
# Define deconvolutional layers
deconv_layers = Flux.Chain(
    # Define linear layer out of latent space
    Flux.Dense(n_latent => n_channels_init * 2 * 7 * 7, Flux.identity),
    # Unflatten input using custom Reshape layer
    AutoEncode.Reshape(7, 7, n_channels_init * 2, :),
    # First transposed convolutional layer
    Flux.ConvTranspose(
        (4, 4), n_channels_init * 2 => n_channels_init, Flux.relu;
        stride=2, pad=1
    ),
    # Second transposed convolutional layer
    Flux.ConvTranspose(
        (4, 4), n_channels_init => 1, Flux.relu;
        stride=2, pad=1
    ),
    # Add normalization layer
    Flux.BatchNorm(1, Flux.sigmoid),
)

# Define decoder
decoder = AutoEncode.BernoulliDecoder(deconv_layers)
```

!!! note
    Again, the custom `Reshape` layer is used to reshape the output of the
    linear layer to the shape expected by the transposed convolutional layers.
    This custom layer is needed to be able to save the model and load it later.

By defining the decoder as a `BernoulliDecoder`, `AutoEncode.jl` already knows
the log-likehood function to use when training the model. We can then simply
define our variational autoencoder by combining the encoder and decoder as

```julia
# Define variational autoencoder
vae = encoder * decoder
```

If for any reason we were curious to explore a different distribution for the
decoder, for example, a `Normal` distribution with constant variance, it would
be as simple as defining the decoder as a `SimpleDecoder`.

```julia
# Define decoder with Normal likelihood function
decoder = AutoEncode.SimpleDecoder(deconv_layers)

# Re-defining the variational autoencoder
vae = encoder * decoder
```

Everything else in our training pipeline would remain the same thanks to
multiple dispatch.

Furthermore, let's say that we would like to use a different flavor for our
variational autoencoder. In particular the `InfoVAE` (also known as `MMD-VAE`)
includes extra terms in the loss function to maximize mutual information between
the latent space and the input data. We can easily take our `vae` model and
convert it into a `MMDVAE`-type object from the `MMDVAEs` submodule as follows:

```julia
mmdvae = AutoEncode.MMDVAEs.MMDVAE(vae)
```

This is the power of `AutoEncode.jl` and Julia's multiple dispatch!

## Implemented Autoencoders

| model                      | module                                  | description                                                    |
| -------------------------- | --------------------------------------- | -------------------------------------------------------------- |
| Autoencoder                | [`AEs`](@ref AEsmodule)                 | Vanilla deterministic autoencoder                              |
| Variational Autoencoder    | [`VAEs`](@ref VAEsmodule)               | Vanilla variational autoencoder                                |
| β-VAE                      | [`VAEs`](@ref VAEsmodule)               | beta-VAE to weigh the reconstruction vs. KL divergence in ELBO |
| MMD-VAEs                   | [`MMDs`](@ref MMDVAEsmodule)            | Maximum-Mean Discrepancy Variational Autoencoders              |
| InfoMax-VAEs               | [`InfoMaxVAEs`](@ref InfoMaxVAEsmodule) | Information Maximization Variational Autoencoders              |
| Hamiltonian VAE            | [`HVAEs`](@ref HVAEsmodule)             | Hamiltonian Variational Autoencoders                           |
| Riemannian Hamiltonian-VAE | [`RHVAEs`](@ref RHVAEsmodule)           | Riemannian-Hamiltonian Variational Autoencoder                 |

!!! tip "Looking for contributors!" 
    If you are interested in contributing to the package to add a new model,
    please check the [GitHub
    repository](https://github.com/mrazomej/AutoEncode.jl). We are always 
    looking to expand the list of available models. And `AutoEncode.jl`'s 
    structure should make it relatively easy.

## GPU support

`AutoEncode.jl` supports GPU training out of the box for `CUDA.jl`-compatible
GPUs. The `CUDA` functionality is provided as an extension. Therefore, to train
a model on the GPU, simply import `CUDA` into the current environment, then move
the model and data to the GPU. The rest of the training pipeline remains the
same.