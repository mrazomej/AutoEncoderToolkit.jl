# β-Variational Autoencoder

Variational Autoencoders, first introduced by Kingma and Welling in 2014, are a
type of generative model that learns to encode high-dimensional data into a
low-dimensional latent space. The main idea behind VAEs is to learn a
probabilistic mapping (via variational inference) from the input data to the
latent space, which allows for the generation of new data points by sampling
from the latent space.

Their counterpart, the β-VAE, introduced by Higgins et al. in 2017, is a variant
of the original VAE that includes a hyperparameter `β` that controls the
relative importance of the reconstruction loss and the KL divergence term in the
loss function. By adjusting `β`, the user can control the trade-off between the
reconstruction quality and the disentanglement of the latent space.

In terms of implementation, the `VAE` struct in `AutoEncode.jl` is a simple
feedforward network composed of variational [encoder](@ref Encoders) and
[decoder](@ref Decoders) parts. This means that the encoder has a log-posterior
function and a KL divergence function associated with it, while the decoder has
a log-likehood function associated with it.

## References

### VAE
> Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
> http://arxiv.org/abs/1312.6114 (2014).

### β-VAE
> Higgins, I. et al. β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED
> VARIATIONAL FRAMEWORK. (2017).



## [`VAE` struct] (@id VAEstruct)

```@docs
AutoEncode.VAEs.VAE
```

## Forward pass

```@docs
AutoEncode.VAEs.VAE(::AbstractArray)
```

## Loss function

```@docs
AutoEncode.VAEs.loss
```

!!! note
    The `loss` function includes the `β` optional argument that can turn a
    vanilla VAE into a β-VAE by changing the default value of `β` from `1.0` to
    any other value.

## Training

```@docs
AutoEncode.VAEs.train!
```