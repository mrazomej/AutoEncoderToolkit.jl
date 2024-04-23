# [Deterministic Autoencoder] (@id AEsmodule)

The deterministic autoencoders are a type of neural network that learns to embed
high-dimensional data into a lower-dimensional space in a one-to-one fashion.
The `AEs` module provides the necessary tools to train these networks. The main
type is the `AE` struct, which is a simple feedforward neural network composed
of two parts: an [`Encoder`](@ref "Encoder") and a [`Decoder`](@ref "Decoder").

## Autoencoder struct `AE`

```@docs
AutoEncode.AEs.AE
```

## Forward pass

```@docs
AutoEncode.AEs.AE(::AbstractArray)
```

## Loss function

### MSE loss

```@docs
AutoEncode.AEs.mse_loss
```

## Training

```@docs
AutoEncode.AEs.train!
```