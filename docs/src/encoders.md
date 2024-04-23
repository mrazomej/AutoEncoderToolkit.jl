# Encoders & Decoders

`AutoEncode.jl` provides a set of predefined encoders and decoders that can be
used to define custom (variational) autoencoder architectures.

## Encoders

The tree structure of the encoder types looks like this (🧱 represents concrete
types):

- `AbstractEncoder`
    - `AbstractDeterministicEncoder`
        - [`Encoder` 🧱](@ref "Encoder")
    - `AbstractVariationalEncoder`
        - `AbstractGaussianEncoder`
            - `AbstractGaussianLinearEncoder`
                - [`JointEncoder` 🧱](@ref JointEncoder)
            - `AbstractGaussianLogEncoder`
                - [`JointLogEncoder` 🧱](@ref JointLogEncoder)

### [`Encoder`] (@id Encoder)

```@docs
AutoEncode.Encoder
AutoEncode.Encoder(::AbstractArray)
```

### [`JointEncoder`] (@id JointEncoder)

```@docs
AutoEncode.JointEncoder
AutoEncode.JointEncoder(::AbstractArray)
```

### [`JointLogEncoder`] (@id JointLogEncoder)

```@docs
AutoEncode.JointLogEncoder
AutoEncode.JointLogEncoder(::AbstractArray)
```

## Decoders

The tree structure of the decoder types looks like this (🧱 represents concrete
types):

- `AbstractDecoder`
    - `AbstractDeterministicDecoder`
        - [`Decoder` 🧱](@ref Decoder)
    - `AbstractVariationalDecoder`
        - [`BernoulliDecoder` 🧱](@ref BernoulliDecoder)
        - [`CategoricalDecoder` 🧱](@ref CategoricalDecoder)
        - `AbstractGaussianDecoder`
            - `AbstractGaussianLinearDecoder`
                - [`JointDecoder` 🧱](@ref JointDecoder)
                - [`SplitDecoder` 🧱](@ref SplitDecoder)
            - `AbstractGaussianLogDecoder`
                - [`JointLogDecoder` 🧱](@ref JointLogDecoder)
                - [`SplitLogDecoder` 🧱](@ref SplitLogDecoder)

### [`Decoder`] (@id Decoder)

```@docs
AutoEncode.Decoder
AutoEncode.Decoder(::AbstractArray)
```

### [`BernoulliDecoder`] (@id BernoulliDecoder)

```@docs
AutoEncode.BernoulliDecoder
AutoEncode.BernoulliDecoder(::AbstractArray)
```

### [`CategoricalDecoder`] (@id CategoricalDecoder)

```@docs
AutoEncode.CategoricalDecoder
AutoEncode.CategoricalDecoder(::AbstractArray)
```

### [`JointDecoder`] (@id JointDecoder)

```@docs
AutoEncode.JointDecoder
AutoEncode.JointDecoder(::AbstractArray)
```

### [`JointLogDecoder`] (@id JointLogDecoder)

```@docs
AutoEncode.JointLogDecoder
AutoEncode.JointLogDecoder(::AbstractArray)
```

### [`SplitDecoder`] (@id SplitDecoder)

```@docs
AutoEncode.SplitDecoder
AutoEncode.SplitDecoder(::AbstractArray)
```

### [`SplitLogDecoder`] (@id SplitLogDecoder)

```@docs
AutoEncode.SplitLogDecoder
AutoEncode.SplitLogDecoder(::AbstractArray)
```

## Default initializations

The package provides a set of functions to initialize encoder and decoder
architectures. Although it gives the user less flexibility, it can be useful for
quick prototyping.

### Encoder initializations

```@docs
AutoEncode.Encoder(
    ::Int, ::Int, ::Vector{<:Int}, ::Vector{<:Function}, ::Function
)
```

```@docs
AutoEncode.JointLogEncoder( 
    ::Int, 
    ::Int, 
    ::Vector{<:Int}, 
    ::Vector{<:Function}, 
    ::Function;
)
AutoEncode.JointLogEncoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Function};
)
```

```@docs
AutoEncode.JointEncoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Function};
)
```

### Decoder initializations

```@docs
AutoEncode.Decoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
```

```@docs
AutoEncode.SimpleDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
```

```@docs
AutoEncode.JointLogDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
AutoEncode.JointLogDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Function};
)
```

```@docs
AutoEncode.JointDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
AutoEncode.JointDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Function};
)
```

```@docs
AutoEncode.SplitLogDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Int},
    ::Vector{<:Function};
)
```

```@docs
AutoEncode.SplitDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Vector{<:Int},
    ::Vector{<:Function};
)
```

```@docs
AutoEncode.BernoulliDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
```

```@docs
AutoEncode.CategoricalDecoder(
    ::AbstractVector{<:Int},
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
AutoEncode.CategoricalDecoder(
    ::Int,
    ::Int,
    ::Vector{<:Int},
    ::Vector{<:Function},
    ::Function;
)
```

## Probabilistic functions

Given the probability-centered design of `AutoEncode.jl`, each variational
encoder and decoder has an associated probabilistic function used when computing
the evidence lower bound (ELBO). The following functions are available:

```@docs
AutoEncode.encoder_logposterior
```

```@docs
AutoEncode.encoder_kl
```

```@docs
AutoEncode.spherical_logprior
```

## Defining custom encoder and decoder types

!!! note
    We will omit all docstrings in the following examples for brevity. However,
    every struct and function in `AutoEncode.jl` is well-documented.

Let us imagine your particular task requires a custom encoder or decoder type.
For example, let's imagine that for a particular application, you need a decoder
whose output distribution is Poisson. In other words, the assumption is that
each dimension in the input $x_i$ is a sample from a Poisson distribution with
mean $\lambda_i$. Thus, on the decoder side, what the decoder return is a vector
of these $\lambda$ paraeters. We thus need to define a custom decoder type.

```julia
struct PoissonDecoder <: AbstractVariationalDecoder
    decoder::Flux.Chain
end # struct
```

With this struct defined, we need to define the forward-pass function for our
custom `PoissonDecoder`. All decoders in `AutoEncode.jl` return a `NamedTuple`
with the corresponding parameters of the distribution that defines them. In this
case, the Poisson distribution is defined by a single parameter $\lambda$. Thus,
we have a forward-pass of the form
```julia
function (decoder::PoissonDecoder)(z::AbstractArray)
    # Run input to decoder network
    return (λ=decoder.decoder(z),)
end # function
```

Next, we need to define the probabilistic function associated with this decoder.
We know that the probability of observing $x_i$ given $\lambda_i$ is given by
```math
P(x_i | \lambda_i) = \frac{\lambda_i^{x_i} e^{-\lambda_i}}{x_i!}.
\tag{1}
```

If each $x_i$ is independent, then the probability of observing the entire input
$x$ given the entire output $\lambda$ is given by the product of the individual
probabilities, i.e.
```math
P(x | \lambda) = \prod_i P(x_i | \lambda_i).
\tag{2}
```

The log-likehood of the data given the output of the decoder is then given by
```math
\mathcal{L}(x, \lambda) = \log P(x | \lambda) = \sum_i \log P(x_i | \lambda_i),
\tag{3}
```
which, by using the properties of the logarithm, can be written as
```math
\mathcal{L}(x, \lambda) = \sum_i x_i \log \lambda_i - \lambda_i - \log(x_i!).
\tag{4}
```

We can then define the probabilistic function associated with the
`PoissonDecoder` as

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
where we use the `loggamma` function from `SpecialFunctions.jl` to compute the
log of the factorial of `x_i`.

!!! warning
    We only defined the `decoder_loglikelihood` method for `z::AbstractVector`.
    One should also include a method for `z::AbstractMatrix` used when
    performing batch training.

With these two functions defined, our `PoissonDecoder` is ready to be used with
any of the different VAE flavors included in `AutoEncode.jl`!