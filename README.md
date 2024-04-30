# AutoEncode.jl

[![Build Status](https://github.com/mrazomej/AutoEncode.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mrazomej/AutoEncode.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/mrazomej/AutoEncode.jl/graph/badge.svg?token=9DKTMW94G5)](https://codecov.io/gh/mrazomej/AutoEncode.jl)

Welcome to the `AutoEncode.jl` GitHub repository. This package provides a simple
interface for training and using [Flux.jl](https://fluxml.ai)-based autoencoders
and variational autoencoders in Julia.

## Installation

You can install `AutoEncode.jl` using the Julia package manager. From the Julia
REPL, type `]` to enter the Pkg REPL mode and run:

```julia
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

For more information, please refer to the
[documentation](https://mrazomej.github.io/AutoEncode.jl/).

## Implemented Autoencoders

| model                      | module        | description                                                    |
| -------------------------- | ------------- | -------------------------------------------------------------- |
| Autoencoder                | `AEs`         | Vanilla deterministic autoencoder                              |
| Variational Autoencoder    | `VAEs`        | Vanilla variational autoencoder                                |
| β-VAE                      | `VAEs`        | beta-VAE to weigh the reconstruction vs. KL divergence in ELBO |
| MMD-VAEs                   | `MMDs`        | Maximum-Mean Discrepancy Variational Autoencoders              |
| InfoMax-VAEs               | `InfoMaxVAEs` | Information Maximization Variational Autoencoders              |
| Hamiltonian VAE            | `HVAEs`       | Hamiltonian Variational Autoencoders                           |
| Riemannian Hamiltonian-VAE | `RHVAEs`      | Riemannian-Hamiltonian Variational Autoencoder                 |