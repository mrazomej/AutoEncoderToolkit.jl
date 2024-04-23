# [MMD-VAE (InfoVAE)] (@id MMDVAEsmodule)

The Maximum-Mean Discrepancy Variational Autoencoder (MMD-VAE) is a variant of
the Variational Autoencoder (VAE) that adds an extra term to the evidence lower
bound (ELBO) that aims to maximize the mutual information between the latent
space representation and the input data. In particular, the MMD-VAE uses the
Maximum-Mean Discrepancy (MMD) as a measure of the "distance" between the latent
space distribution and the input data distribution.

For the implementation of the MMD-VAE in `AutoEncode.jl`, the [`MMDVAE`](@ref
MMDVAEstruct) struct inherits directly from the [`VAE`](@ref VAEstruct) struct
and adds the necessary functions to compute the extra terms in the loss
function. An `MMDVAE` object is created by simply passing a `VAE` object to the
constructor. This way, we can use `Julia`s multiple dispatch to extend the
functionality of the `VAE` object without having to redefine the entire
structure.

## Reference

> Maximum-Mean Discrepancy Variational Autoencoders
> Zhao, S., Song, J. & Ermon, S. InfoVAE: Information Maximizing Variational
> Autoencoders. Preprint at http://arxiv.org/abs/1706.02262 (2018).

## [`MMDVAE` struct] (@id MMDVAEstruct)

```@docs
AutoEncode.MMDVAEs.MMDVAE{AutoEncode.VAEs.VAE}
```

## Forward pass

```@docs
AutoEncode.MMDVAEs.MMDVAE(::AbstractArray)
```

## Loss function

```@docs
AutoEncode.MMDVAEs.loss
```

## Training

```@docs
AutoEncode.MMDVAEs.train!
```

## Other Functions

```@docs
AutoEncode.MMDVAEs.gaussian_kernel
AutoEncode.MMDVAEs.mmd_div
AutoEncode.MMDVAEs.logP_mmd_ratio
```