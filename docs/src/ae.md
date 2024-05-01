# [Deterministic Autoencoder] (@id AEsmodule)

The deterministic autoencoders are a type of neural network that learns to embed
high-dimensional data into a lower-dimensional space in a one-to-one fashion.
The `AEs` module provides the necessary tools to train these networks. The main
type is the `AE` struct, which is a simple feedforward neural network composed
of two parts: an [`Encoder`](@ref "Encoder") and a [`Decoder`](@ref "Decoder").

## Autoencoder struct `AE`

```@docs
AutoEncoderToolkit.AEs.AE
```

## Forward pass

```@docs
AutoEncoderToolkit.AEs.AE(::AbstractArray)
```

## Loss function

### MSE loss

```@docs
AutoEncoderToolkit.AEs.mse_loss
```

## Training

```@docs
AutoEncoderToolkit.AEs.train!
```