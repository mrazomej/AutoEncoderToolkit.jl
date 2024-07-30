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
    affiliation: 1
    corresponding: true
affiliations:
 - name: Department of Biology, Stanford University, CA, United States of America
   index: 1
date: "30 July 2024"
bibliography: paper.bib
---

# Summary

With the advent of generative models, the field of unsupervised learning has
exploded in the last decade. One of the most popular generative models is the
variational autoencoder (VAE) [@kingma2014]. VAEs assume the existence of a
joint probability distribution between a high-dimensional data space and a
lower-dimensional latent space. Using a variational inference approach, the VAE
parametrizes this joint distribution with two neural networks--an encoder and a
decoder. This approach allows the model to approximate the underlying
low-dimensional structure that generated the observed data and generate new
samples by sampling from the learned latent space. Several variations of the
original VAE have been proposed to extend its capabilities and tackle different
problems. Here, we present `AutoEncoderToolkit.jl`, a `Julia` package for
training VAEs and its extensions. The package is built on top of the `Flux.jl`
deep learning library [@innes2018] and provides a simple and flexible interface
for training different flavors of VAEs. Furthermore, the package offers a set of
utilities for the geometric analysis of the learned latent space.

# Statement of need

Collecting and analyzing large high-dimensional datasets have become routine in
several scientific fields. Therefore, the need to understand and uncover the
underlying low-dimensional structure of these datasets has become more pressing
than ever. VAEs have shown great promise in this regard, with applications
ranging from single-cell transcriptomics [@lopez2018] to protein design
[@lian2022] to the discovery of the governing equations of dynamical systems
[@champion2019]. However, most of these tools have been exclusively developed
for the `Python` ecosystem. `Julia` is a promising high-performance language for
scientific computing, with a growing ecosystem for deep learning,
state-of-the-art automatic differentiation, and a strong GPU-accelerated
computing backend. Furthermore, the programming paradigm of `Julia`, based on
multiple dispatch, allows for a more flexible, modular, and composable codebase.
`AutoEncoderToolkit.jl` aims to provide a simple and flexible interface for
training VAEs and their extensions in `Julia`, taking full advantage of the
language programming paradigm.

# Software Description

## Encoder \& Decoder definition

`AutoEncoderToolkit.jl` takes a probability-theory-based approach to the package
design. Taking advantage of the multiple dispatch paradigm of `Julia`, the
package can easily combine and extend the encoders and decoders to quickly
prototype different VAE architectures. In other words, independent of the design
of the multi-layer perceptron used for either the encoder or the decoder, what
defines their behavior is their associated probability distribution. For
example, let us consider the loss function for the standard VAE model, given by
the evidence lower bound (ELBO): 
$$ 
\text{ELBO} = \left\langle \log p_\theta(x|z) \right\rangle_{q_\phi(z|x)} -
D_{KL}(q_\phi(z|x)
|| p(z)), 
\tag{1} 
$$ 
where $p_\theta(x|z)$--defined by the decoder with parameters $\theta$--is the
likelihood of the data given the latent variable, $q_\phi(z|x)$--defined by the
encoder with parameters $\phi$--is the posterior distribution of the latent
variable given the data, $D_{KL}(q_\phi(z|x) || p(z))$ is the Kullback-Leibler
divergence between the posterior and the prior distribution of the latent space,
and $\langle\cdot\rangle_{q_\phi(z|x)}$ is the expected value over the posterior
distribution. By defining the decoder type as either `BernoulliDecoder` or
`SimpleGaussianDecoder`, the user can decide whether the decoder parametrizes a
likelihood function $p_\theta(x|z)$ as a Bernoulli distribution or a Gaussian
distribution with constant diagonal covariance, respectively. This design choice
allows for the quick prototyping of different architectures without the overhead
of defining new specific losses for each type of decoder.

Furthermore, the design allows for the easy extension of the available encoders
and decoders that can directly integrate into any available VAE model. For
example, let us assume that for a particular problem, the user wants to define a
decoder whose outputs are the parameters for independent Poisson distributions,
each with a different parameter $\lambda_i$. In other words, on the decoder
side, the decoder returns a vector of parameters $\lambda$ for each of the
dimensions of the data. The user can define a new decoder type

```julia
struct PoissonDecoder <: AbstractVariationalDecoder
    decoder::Flux.Chain
end # struct
```

With this `struct` defined, the user only needs to define two methods: one for
the forward pass of the decoder

```julia
function (decoder::PoissonDecoder)(z::AbstractArray)
    # Forward pass through decoder
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

With these methods defined, the `PoissonDecoder` can be directly integrated into
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

Other than the deterministic autoencoder, all the models listed above use the
same underlying `VAE` struct as part of their definition. Some of them, like the
`InfoMaxVAE` and `RHVAE` require additional neural networks for training or
inference. However, other than those additional elements, the training routines
for all models are virtually the same. This design choice allows users to
quickly explore different VAE models for their specific applications without
writing new training routines for each model.

Moreover, extending the list of VAE models is also straightforward as
contributions to the package only need to focus on a general definition of the
loss function associated with the new VAE model without the need to define
specific terms for each type of encoder or decoder.

## Differential geometry utilities

In recent years, there has been a growing interest in understanding the
geometric properties of the latent space learned by VAEs
[@chadebec2022;@arvanitidis2021]. This is because the non-linearities of the
encoder and decoder networks can induce complex geometries in the latent space,
where the Euclidean distance between points in the latent space does not
necessarily reflect the true distance between the corresponding data points.
Thus, tools from differential geometry such as geodesic distance, parallel
transport, and curvature can provide deeper insights into the structure of the
learned latent space. `AutoEncoderToolkit.jl` provides a set of utilities for
the geometric analysis of the latent space. For example, at the time of this
writing, the `NeuralGeodedics` module provides the tools to approximate geodesic
curves between points in latent space for the Riemannian Hamiltonian VAE (RHVAE)
model. This is achieved by utilizing a neural network to approximate the
geodesic equation [@chen2018a] in the latent space using the explicit
representation of the Riemannian metric learned by the RHVAE model
[@chadebec2020].

## GPU support

`AutoEncoderToolkit.jl` offers GPU support for `CUDA.jl` compatible GPUs out of
the box. 

## Documentation

Documentation is available at
(https://mrazomej.github.io/AutoEncoderToolkit.jl), where there are worked-out
examples and tutorials on how to use the package.

# Acknowledgements

I would like to thank Griffin Chure, Madhav Mani, and Dmitri Petrov for their
advice and helpful discussions during the development of this package. I also
want to thank the Schmidt Science Fellows program for funding part of this work
via a postdoctoral fellowship. I would also like to thank the reviewers of this
manuscript for their helpful comments and suggestions. The journal's transparent
peer review process has dramatically improved the quality of this work.

# References