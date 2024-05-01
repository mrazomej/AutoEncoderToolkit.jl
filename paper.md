---
title: 'AutoEncode.jl: A Julia package for training Variational Autoencoders'
tags:
  - Julia
  - Unsupervised Learning
  - Deep Learning
  - Auto Encoders
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
tackle different problems. Here, we present `AutoEncode.jl`, a Julia package for
training VAEs and its extensions. The package is built on top of the `Flux.jl`
deep learning library [@innes2018] and provides a simple and flexible interface
for training different flavors of VAEs. Furthermore, the package provides a set
of utilities for the geometric analysis of the learned latent space.

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
`AutoEncode.jl` aims to provide a simple and flexible interface for training
VAEs and its extensions in `Julia`. At the time of this writing, the package 
has implemented the following architectures:
- Deterministic Autoencoder
- Vanilla Variational Autoencoder (VAE)
- $\beta$-Variational Autoencoder ($\beta$-VAE) [@higgins2017a]
- Maximum-Mean Discrepancy Variational Autoencoder (InfoVAE)
- InfoMax Variational Autoencoder (InfoMaxVAE) [@rezaabad2020]
- Hamiltonian Variational Autoencoder (HVAE)
- Riemannian Hamiltonian Variational Autoencoder (RHVAE) [@chadebec2020]

# Design

`AutoEncode.jl` takes a probability-theory-based approach to the design of the
package. Taking advantage of the multiple dispatch paradigm of `Julia`, the set
of encoders and decoders provided by the package can be easily combined and
extended to quickly prototype different VAE architectures. In other words,
independent of the design of the multi-layer perceptron used for either the
encoder or the decoder, what defines their behavior is their associated
probability distribution. For example, with the same architecture, by defining
the decoder type as either `BernoulliDecoder` or `SimpleDecoder`, the user can
decide whether the output of the decoder is a Bernoulli distribution or a simple
Gaussian distribution, respectively. This design choice allows for the quick
prototyping of different architectures without the overhead of defining new
losses or training loops. Moreover, the design allows for the easy extension of
the list of available flavors of VAEs, as contributions to the package only need
to focus on the definition of the corresponding loss function of the new VAE.

# Acknowledgements

I would like to thank Griffin Chure, Madhav Mani, and Dmitri Petrov for their
helpful advice and discussions during the development of this package. I would
also like to thank the Schmidt Science Fellows program for funding part of this
work via a postdoctoral fellowship.

# References