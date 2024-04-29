# [Hamiltonian Variational Autoencoder] (@id HVAEsmodule)

The Hamiltonian Variational Autoencoder (HVAE) is a variant of the Variational
autoencoder (VAE) that uses Hamiltonian dynamics to improve the sampling of the
latent space representation. HVAE combines ideas from Hamiltonian Monte Carlo,
annealed importance sampling, and variational inference to improve the latent
space representation of the VAE.

For the implementation of the HVAE in `AutoEncode.jl`, the [`HVAE`](@ref
HVAEstruct) struct inherits directly from the [`VAE`](@ref VAEstruct) struct and
adds the necessary functions to compute the Hamiltonian dynamics steps as part
of the training protocol. An `HVAE` object is created by simply passing a `VAE`
object to the constructor. This way, we can use `Julia`s multiple dispatch to
extend the functionality of the `VAE` object without having to redefine the
entire structure.

!!! warning
    HVAEs require the computation of nested gradients. This means that the
    AutoDiff framework must differentiate a function of an already AutoDiff
    differentiated function. This is known to be problematic for `Julia`'s
    AutoDiff backends. See [details below](@ref gradpotenergy) to understand how
    to we circumvent this problem.

## Reference

> Caterini, A. L., Doucet, A. & Sejdinovic, D. Hamiltonian Variational
> Auto-Encoder. 11 (2018).

## [`HVAE` struct] (@id HVAEstruct)

```@docs
AutoEncode.HVAEs.HVAE
```

## Forward pass

```@docs
AutoEncode.HVAEs.HVAE(::AbstractArray)
```

## Loss function

```@docs
AutoEncode.HVAEs.loss
```

## Training

```@docs
AutoEncode.HVAEs.train!
```

## [Computing the gradient of the potential energy] (@id gradpotenergy)

One of the crucial components in the training of the HVAE is the computation of
the gradient of the potential energy $$\nabla U$$ with respect to the latent
space representation. This gradient is used in the leapfrog steps of the
Hamiltonian dynamics. When training the HVAE, we need to backpropagate through
the leapfrog steps to update the parameters of the neural network. This requires
computing a gradient of a function of the gradient of the potential energy,
i.e., nested gradients. `Zygote.jl` the main AutoDiff backend in `Flux.jl`
[famously
struggle](https://discourse.julialang.org/t/is-it-possible-to-do-nested-ad-elegantly-in-julia-pinns/98888)
with these types of computations. Specifically, `Zygote.jl` does not support
`Zygote` over `Zygote` differentiation (meaning differentiating a function of
something previously differentiated with `Zygote` using `Zygote`), or `Zygote`
over `ForwardDiff` (meaning differentiating a function of something
differentiated with `ForwardDiff` using `Zygote`).

With this, we are left with a couple of options to compute the gradient of the
potential energy:
- Use finite differences to approximate the gradient of the potential energy.
- Use the relatively new
  [`TaylorDiff.jl`](https://github.com/JuliaDiff/TaylorDiff.jl/tree/main)
  AutoDiff backend to compute the gradient of the potential energy. This backend
  is composable with `Zygote.jl`, so we can, in principle, do `Zygote` over
  `TaylorDiff` differentiation.

The second option would be preferred, as the gradients computed with
`TaylorDiff` are much more accurate than the ones computed with finite
differences. However, there are two problems with this approach:
1. The `TaylorDiff` nested gradient capability stopped working with `Julia ≥
    1.10`, as discussed in
    [#70](https://github.com/JuliaDiff/TaylorDiff.jl/issues/70).
2. Even for `Julia < 1.10`, we could not get `TaylorDiff` to work on `CUDA`
    devices. (PRs are welcome!)

With these limitations in mind, we have implemented the gradient of the
potential using both finite differences and `TaylorDiff`. The user can choose
which method to use by setting the `adtype` keyword argument in the `∇U_kwargs`
in the `loss` function to either `:finite` or `:TaylorDiff`. This means that
for the `train!` function, the user can pass `loss_kwargs` that looks like this:

```julia
# Define the autodiff backend to use
loss_kwargs = Dict(
    :∇U_kwargs => Dict(
        :adtype => :finite
    )
)
```
!!! note
    Although verbose, the nested dictionaries help to keep everything organized.
    (PRs with better design ideas are welcome!)

The default both for `cpu` and `gpu` devices is `:finite`.

```@docs
AutoEncode.HVAEs.∇potential_energy_finite
AutoEncode.HVAEs.∇potential_energy_TaylorDiff
```

## Other Functions

```@docs
AutoEncode.HVAEs.potential_energy
AutoEncode.HVAEs.∇potential_energy
AutoEncode.HVAEs.leapfrog_step
AutoEncode.HVAEs.quadratic_tempering
AutoEncode.HVAEs.null_tempering
AutoEncode.HVAEs.leapfrog_tempering_step
AutoEncode.HVAEs._log_p̄
AutoEncode.HVAEs._log_q̄
AutoEncode.HVAEs.hamiltonian_elbo
```