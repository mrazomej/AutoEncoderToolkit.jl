---
title: 'AutoEncoderToolkit.jl: A Julia package for training (Variational) Autoencoders'
tags:
  - Julia
  - Unsupervised Learning
  - Deep Learning
  - Autoencoders
  - Dimensionality Reduction
authors:
  - name: Manuel Razo-Mejia
    orcid: 0000-0002-9510-0527
    affiliation: "1, 2"
affiliations:
 - name: Department of Biology, Stanford University, CA, United States of America
   index: 1
 - name: For correspondence, contact mrazo@stanford.edu
   index: 2
date: "01 May 2024"
bibliography: paper.bib
---

# Summary

With the advent of generative models, the field of unsupervised learning has
exploded in the last decade. One of the most popular generative models is the
variational autoencoder (VAE) [@kingma2014]. VAEs assume the existence of a
joint probability distribution between a high-dimensional data space and a
lower-dimensional latent space. The VAE parametrizes this joint distribution
with two neural networks--an encoder and a decoder--using a variational
inference approach. This approach allows for the model to approximate the
underlying low-dimensional structure that generated the observed data and
generate new data samples by sampling from the learned latent space. Several
variations of the original VAE have been proposed to extend its capabilities and
tackle different problems. Here, we present `AutoEncoderToolkit.jl`, a Julia
package for training VAEs and its extensions. The package is built on top of the
`Flux.jl` deep learning library [@innes2018] and provides a simple and flexible
interface for training different flavors of VAEs. Furthermore, the package
provides a set of utilities for the geometric analysis of the learned latent
space.

# Statement of need

The collection and analysis of large high-dimensional datasets have become
routine in several scientific fields. Therefore, the need to understand and
uncover the underlying low-dimensional structure of these datasets is more
pressing than ever. VAEs, have shown great promise in this regard, with
applications ranging from single-cell transcriptomics [@lopez2018], to protein
design [@lian2022], to the discovery of the governing equations of dynamical
systems [@champion2019]. However, most of these tools have been exclusively
developed for the `Python` ecosystem. `Julia` is a promising high-performance
language for scientific computing, with a growing ecosystem for deep learning,
state-of-the-art automatic differentiation, and a strong GPU-accelerated
computing backend. Furthermore, the programming paradigm of `Julia`, based on
multiple dispatch, allows for a more flexible, modular, and composable codebase.
`AutoEncoderToolkit.jl` aims to provide a simple and flexible interface for
training VAEs and its extensions in `Julia` taking full advantage of the
programming paradigm of the language.

# Software Description

## Encoder \& Decoder definition

`AutoEncoderToolkit.jl` takes a probability-theory-based approach to the design
of the package. Taking advantage of the multiple dispatch paradigm of `Julia`,
the set of encoders and decoders provided by the package can be easily combined
and extended to quickly prototype different VAE architectures. In other words,
independent of the design of the multi-layer perceptron used for either the
encoder or the decoder, what defines their behavior is their associated
probability distribution. For example, let us consider the loss function for the
standard VAE model, given by the evidence lower bound (ELBO):
$$
\text{ELBO} = \left\langle \log p_\theta(x|z) \right\rangle_{q_\phi(z|x)} - 
D_{KL}(q_\phi(z|x) || p(z)),
\tag{1}
$$
where $p_\theta(x|z)$ is the likelihood of the data given the latent variable
defined by the decoder with parameters $\theta$, $q_\phi(z|x)$ is the posterior
distribution of the latent variable given the data defined by the encoder with
parameters $\phi$, and $D_{KL}(q_\phi(z|x) || p(z))$ is the Kullback-Leibler
divergence between the posterior and the prior distribution of the latent space.
For a particular training, by defining the decoder type as either
`BernoulliDecoder` or `SimpleGaussianDecoder`, the user can decide whether the
$p_\theta(x|z)$ is a Bernoulli distribution or a Gaussian distribution with
constant diagonal covariance, respectively. This design choice allows for the
quick prototyping of different architectures without the overhead of defining
new losses or training loops. 

Furthermore, the design allows for the easy extension of the list of available
encoders and decoders that can directly integrate to any of the existing list of
VAE models. For example, let us assume that for a particular problem, the user
wants to use a decoder whose output is a sample from independent Poisson
distributions, each with a different parameter $\lambda_i$. In other words,
on the decoder side, the decoder returns a vector of parameters $\lambda$ for
each of the dimensions of the data. The user can define a new decoder type

```julia
struct PoissonDecoder <: AbstractVariationalDecoder
    decoder::Flux.Chain
end # struct
```

With this `struct` defined, the user only needs to define two methods: one for
for the forward pass of the decoder

```julia
function (decoder::PoissonDecoder)(z::AbstractArray)
    # Run input to decoder network
    return (λ=decoder.decoder(z),)
end # function
```

and another for the likelihood of the data given the latent variable

```julia
function decoder_loglikelihood(
        x::AbstractArray,
        z::AbstractVector,
        decoder::PoissonDecoder,
        decoder_output::NamedTuple;
)
        # Extract the lambda parameter of the Poisson distribution
        λ = decoder_output.λ

        # Compute log-likelihood
        loglikelihood = sum(x .* log.(λ) - λ - loggamma.(x .+ 1))

        return loglikelihood
end # function
```

where we use the log-likelihood function of multiple independent Poisson 
distributions, given by
$$
\ln p(x|\lambda) = \sum_i x_i \log(\lambda_i) - 
\lambda_i - \log(\Gamma(x_i + 1)).
\tag{2}
$$

Wit these methods defined, the `PoissonDecoder` can be directly integrated into
any of the different VAE models provided by the package.

## Implemented models

At the time of this writing, the package has implemented the following models:

: List of implemented (variational) autoencoder models

| Name                                                       | Reference       |
| ---------------------------------------------------------- | --------------- |
| Deterministic Autoencoder                                  |                 |
| Vanilla Variational Autoencoder (VAE)                      | [@kingma2014]   |
| $\beta$-Variational Autoencoder ($\beta$-VAE)              | [@higgins2017a] |
| Maximum-Mean Discrepancy Variational Autoencoder (InfoVAE) | [@zhao2018]     |
| InfoMax Variational Autoencoder (InfoMaxVAE)               | [@rezaabad2020] |
| Hamiltonian Variational Autoencoder (HVAE)                 | [@caterini2018] |
| Riemannian Hamiltonian Variational Autoencoder (RHVAE)     | [@chadebec2020] |

Moreover, extending the list of VAE models is also straightforward as
contributions to the package only need to focus on a general definition of the
loss function associated with the new VAE model, without the need to define
specific terms for each type of encoder or decoder.

## Differential geometry utilities

## GPU support

`AutoEncoderToolkit.jl` offers GPU support for `CUDA.jl`-compatible GPUs out of
the box. To avoid the overhead of loading this dependency, the `CUDA`-specific
functions are loaded as an extension once `CUDA.jl` is loaded. 

# Acknowledgements

I would like to thank Griffin Chure, Madhav Mani, and Dmitri Petrov for their
helpful advice and discussions during the development of this package. I would
also like to thank the Schmidt Science Fellows program for funding part of this
work via a postdoctoral fellowship.

# References