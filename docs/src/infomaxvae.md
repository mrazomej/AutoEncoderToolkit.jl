# [InfoMax VAE] (@id InfoMaxVAEsmodule)

The InfoMax VAE is a variant of the Variational Autoencoder (VAE) that aims to
explicitly account for the maximization of mutual information between the latent
space representation and the input data. The main difference between the InfoMax
VAE and the [MMD-VAE (InfoVAE)](@ref MMDVAEsmodule) is that rather than using
the Maximum-Mean Discrepancy (MMD) as a measure of the "distance" between the
latent space, the InfoMax VAE explicitly models the mutual information between
latent representations and data inputs via a separate neural network. The loss
function for this separate network then takes the form of a variational lower
bound on the mutual information between the latent space and the input data.

Because of the need of this separate network, the [`InfoMaxVAE`](@ref
InfoMaxVAE) struct in `AutoEncoderToolkit.jl` takes two arguments to construct: the
original [`VAE`](@ref VAEstruct) struct and a network to compute the mutual
information. To properly deploy all relevant functions associated with this
second network, we also provide a [`MutualInfoChain`](@ref MutualInfoChain)
struct.

Furthermore, because of the two networks and the way the training algorithm is
set up, the loss function for the InfoMax VAE includes two separate loss
functions: one for the [`MutualInfoChain`](@ref miloss) and one for the
[`InfoMaxVAE`](@ref infomaxloss).

## References

> Rezaabad, A. L. & Vishwanath, S. Learning Representations by Maximizing Mutual
> Information in Variational Autoencoders. Preprint at
> http://arxiv.org/abs/1912.13361 (2020).

## [`MutualInfoChain` struct] (@id MutualInfoChain)

```@docs
AutoEncoderToolkit.InfoMaxVAEs.MutualInfoChain
```

## [`InfoMaxVAE` struct] (@id InfoMaxVAE)

```@docs
AutoEncoderToolkit.InfoMaxVAEs.InfoMaxVAE
```

## Forward pass

### Mutual Information Network

```@docs
AutoEncoderToolkit.InfoMaxVAEs.MutualInfoChain(::AbstractArray, ::AbstractVecOrMat)

```

### InfoMax VAE
```@docs
AutoEncoderToolkit.InfoMaxVAEs.InfoMaxVAE(::AbstractArray)
```

## [Loss functions] 

### [Mutual Information Network] (@id miloss)
```@docs
AutoEncoderToolkit.InfoMaxVAEs.miloss
```

### [InfoMax VAE] (@id infomaxloss)
```@docs
AutoEncoderToolkit.InfoMaxVAEs.infomaxloss
```

## Training

```@docs
AutoEncoderToolkit.InfoMaxVAEs.train!
```

## Other Functions

```@docs
AutoEncoderToolkit.InfoMaxVAEs.shuffle_latent
AutoEncoderToolkit.InfoMaxVAEs.variational_mutual_info
```

## Default initializations

`AutoEncoderToolkit.jl` provides default initializations for the `MutualInfoChain`.
Although it gives the user less flexibility, it can be useful for quick
prototyping.

```@docs
AutoEncoderToolkit.InfoMaxVAEs.MutualInfoChain(
    ::Union{Int,Vector{<:Int}},
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
```