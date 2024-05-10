# [Riemannian Hamiltonian Variational Autoencoder] (@id RHVAEsmodule)

The Riemannian Hamiltonian Variational Autoencoder (RHVAE) is a variant of the
Hamiltonian Variational Autoencoder (HVAE) that uses concepts from Riemannian
geometry to improve the sampling of the latent space representation. As the
HVAE, the RHVAE uses Hamiltonian dynamics to improve the sampling of the latent.
However, the RHVAE accounts for the geometry of the latent space by learning a
Riemannian metric tensor that is used to compute the kinetic energy of the
dynamical system. This allows the RHVAE to sample the latent space more evenly
while learning the curvature of the latent space.

For the implementation of the RHVAE in `AutoEncoderToolkit.jl`, the [`RHVAE`](@ref
RHVAEstruct) requires two arguments to construct: the original [`VAE`](@ref
VAEstruct) as well as a separate neural network used to compute the metric
tensor. To facilitate the dispatch of the necessary functions associated with
this second network, we also provide a [`MetricChain`](@ref MetricChain) struct.

!!! warning
    RHVAEs require the computation of nested gradients. This means that the
    AutoDiff framework must differentiate a function of an already AutoDiff
    differentiated function. This is known to be problematic for `Julia`'s
    AutoDiff backends. See [details below](@ref gradhamiltonian) to understand
    how to we circumvent this problem.

## Reference

> Chadebec, C., Mantoux, C. & Allassonnière, S. Geometry-Aware Hamiltonian
> Variational Auto-Encoder. Preprint at http://arxiv.org/abs/2010.11518 (2020).

## [`MetricChain` struct] (@id MetricChainstruct)

```@docs
AutoEncoderToolkit.RHVAEs.MetricChain
```

## [`RHVAE` struct] (@id RHVAEstruct)

```@docs
AutoEncoderToolkit.RHVAEs.RHVAE
```

## Forward pass

### [Metric Network] (@id MetricChain)

```@docs
AutoEncoderToolkit.RHVAEs.MetricChain(::AbstractArray)
```

### RHVAE

```@docs
AutoEncoderToolkit.RHVAEs.RHVAE(::AbstractArray)
```

## Loss function

```@docs
AutoEncoderToolkit.RHVAEs.loss
```

## Training

```@docs
AutoEncoderToolkit.RHVAEs.train!
```

## [Computing the gradient of the potential energy] (@id gradhamiltonian)

One of the crucial components in the training of the RHVAE is the computation of
the gradient of the Hamiltonian $$\nabla H$$ with respect to the latent space
representation. This gradient is used in the leapfrog steps of the generalized
Hamiltonian dynamics. When training the RHVAE, we need to backpropagate through
the leapfrog steps to update the parameters of the neural network. This requires
computing a gradient of a function of the gradient of the Hamiltonian, i.e.,
nested gradients. `Zygote.jl` the main AutoDiff backend in `Flux.jl` [famously
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
which method to use by setting the `adtype` keyword argument in the `∇H_kwargs`
in the `loss` function to either `:finite` or `:TaylorDiff`. This means that
for the `train!` function, the user can pass `loss_kwargs` that looks like this:

```julia
# Define the autodiff backend to use
loss_kwargs = Dict(
    :∇H_kwargs => Dict(
        :adtype => :finite
    )
)
```
!!! note
    Although verbose, the nested dictionaries help to keep everything organized.
    (PRs with better design ideas are welcome!)

The default both for `cpu` and `gpu` devices is `:finite`.

```@docs
AutoEncoderToolkit.RHVAEs.∇hamiltonian_finite
AutoEncoderToolkit.RHVAEs.∇hamiltonian_TaylorDiff
AutoEncoderToolkit.RHVAEs.∇hamiltonian_ForwardDiff
```

## Other Functions
```@docs
AutoEncoderToolkit.RHVAEs.update_metric
AutoEncoderToolkit.RHVAEs.update_metric!
AutoEncoderToolkit.RHVAEs.G_inv
AutoEncoderToolkit.RHVAEs.metric_tensor
AutoEncoderToolkit.RHVAEs.riemannian_logprior
AutoEncoderToolkit.RHVAEs.hamiltonian
AutoEncoderToolkit.RHVAEs.∇hamiltonian
AutoEncoderToolkit.RHVAEs._leapfrog_first_step
AutoEncoderToolkit.RHVAEs._leapfrog_second_step
AutoEncoderToolkit.RHVAEs._leapfrog_third_step
AutoEncoderToolkit.RHVAEs.general_leapfrog_step
AutoEncoderToolkit.RHVAEs.general_leapfrog_tempering_step
AutoEncoderToolkit.RHVAEs._log_p̄
AutoEncoderToolkit.RHVAEs._log_q̄
AutoEncoderToolkit.RHVAEs.riemannian_hamiltonian_elbo
```

## Default initializations

`AutoEncoderToolkit.jl` provides default initializations for both the metric tensor
network and the RHVAE. Although less flexible than defining your own initial
networks, these can serve as a good starting point for your experiments.

```@docs
AutoEncoderToolkit.RHVAEs.MetricChain(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
AutoEncoderToolkit.RHVAEs.RHVAE(
    ::AutoEncoderToolkit.VAEs.VAE,
    ::AutoEncoderToolkit.RHVAEs.MetricChain,
    ::AbstractArray{AbstractFloat},
    T::AbstractFloat,
    λ::AbstractFloat
)
```
