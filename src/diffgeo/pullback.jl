# ==============================================================================
# Pullback Metric computation
# ==============================================================================

# Reference
# > Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
# > Curvature of Deep Generative Models. Preprint at
# > http://arxiv.org/abs/1710.11379 (2021).

@doc raw"""
    pullback_metric(manifold::Function, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M = JᵀJ` of a manifold using
numerical differentiation with `Zygote.jl`. The metric is computed at a specific
point `z` on the manifold.

# Arguments
- `manifold::Function`: A function defining the manifold.
- `z::AbstractVector`: A vector specifying the point on the manifold where
  the metric should be evaluated.

# Returns
- `M::Matrix`: The Riemannian metric matrix evaluated at `val`.

# Example
```julia-repl
manifold(x) = [x[1]^2 + x[2], x[1]*x[2]]
val = [1.0, 2.0]
M = pullback_metric(manifold, val)
```
"""
function pullback_metric(
    manifold::Function, z::AbstractVector
)
    # Compute Jacobian
    jac = first(Zygote.jacobian(manifold, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::AbstractDeterministicDecoder, z::AbstractVector)

Compute the Riemannian metric (pullback metric) `M = JᵀJ` at a specific point on
a manifold. The manifold is defined by the decoder structure of an autoencoder
(AE). This function uses numerical differentiation with `Zygote.jl` to compute
the metric. The metric is the squared Jacobian matrix, which encapsulates the
local geometric properties of the decoder's output space with respect to its
input space.

# Arguments
- `decoder::AbstractDeterministicDecoder`: A deterministic decoder structure
  that defines the manifold. It should be an instance of a subtype of
  `AbstractDeterministicDecoder`.
- `z::AbstractVector`: A vector specifying the point in the latent space of the
  manifold where the metric should be evaluated.

# Returns
- `M::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::AbstractDeterministicDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(decoder.decoder, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        pullback_metric(decoder::SimpleGaussianDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) of a manifold defined by the
decoder structure of a variational autoencoder (VAE) using numerical
differentiation with `Zygote.jl`. The metric is evaluated based on the outputs
of the decoder with respect to its inputs. The metric is defined as

M̲̲ = J̲̲_µᵀ J̲̲_µ

where J̲̲_µ is the Jacobian matrix of the decoder with respect to its input.

The `SimpleGaussianDecoder` is a variational decoder that assumes a constant diagonal
covariance matrix in the output. Therefore, the second term of the metric, which
would account for the variance, is not added in this case.

# Arguments
- `decoder::SimpleGaussianDecoder`: A Variational decoder structure containing a neural
  network model that defines the manifold. It assumes a constant diagonal
  covariance matrix in the output.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::SimpleGaussianDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input
    jac = first(Zygote.jacobian(decoder.decoder, z))
    # Compute the metric
    return jac' * jac
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::JointGaussianLogDecoder, z::AbstractVector) 

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `JointGaussianLogDecoder` using numerical differentiation with `Zygote.jl`.
The metric is evaluated based on the outputs of the neural network with respect
to its inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: 

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input. Given that we compute the Jacobian of `logσ` directly, the
Jacobian of `σ` is obtained using the chain rule.

# Arguments
- `decoder::JointGaussianLogDecoder`: A VAE decoder structure that has separate paths
  for determining both the mean and log standard deviation of the latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::JointGaussianLogDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.µ), z)
    )

    # Compute Jacobian with respect to the input for the log standard deviation
    logσ_val, jac_logσ = Zygote.withjacobian(
        Flux.Chain(decoder.decoder..., decoder.logσ), z
    )

    # Convert jac_logσ to jac_σ using the chain rule:
    # 1. Compute σ by exponentiating logσ
    σ_val = exp.(logσ_val)
    # 2. Use the chain rule
    jac_σ = σ_val .* first(jac_logσ)

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

@doc raw"""
    pullback_metric(decoder::SplitGaussianLogDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `SplitGaussianLogDecoder` using numerical differentiation with `Zygote.jl`.
The metric is evaluated based on the outputs of the individual neural networks
with respect to their inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input. Given that we compute the Jacobian of `logσ` directly, the
Jacobian of `σ` is obtained using the chain rule.

# Arguments
- `decoder::SplitGaussianLogDecoder`: A VAE decoder structure that has separate neural
  networks for determining both the mean and log standard deviation of the
  latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::SplitGaussianLogDecoder, z::AbstractVector)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(Zygote.jacobian(decoder.µ, z))

    # Compute Jacobian with respect to the input for the log standard deviation
    logσ_val, jac_logσ = Zygote.withjacobian(decoder.logσ, z)

    # Convert jac_logσ to jac_σ using the chain rule
    σ_val = exp.(logσ_val)
    jac_σ = σ_val .* first(jac_logσ)

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::JointGaussianDecoder, z::AbstractVector) 

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `JointGaussianDecoder` using numerical differentiation with
`Zygote.jl`. The metric is evaluated based on the outputs of the neural network
with respect to its inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by: 

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input.

# Arguments
- `decoder::JointGaussianDecoder`: A VAE decoder structure that has separate paths
  for determining both the mean and standard deviation of the latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::AbstractMatrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(
    decoder::JointGaussianDecoder, z::AbstractVector
)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.µ), z)
    )

    # Compute Jacobian with respect to the input for the standard deviation
    jac_σ = first(
        Zygote.jacobian(Flux.Chain(decoder.decoder..., decoder.σ), z)
    )

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
    pullback_metric(decoder::SplitGaussianDecoder, z::AbstractVector)

Compute the Riemannian metric (pull-back metric) `M̲̲` for a stochastic manifold
defined by a `SplitGaussianDecoder` using numerical differentiation with
`Zygote.jl`. The metric is evaluated based on the outputs of the individual
neural networks with respect to their inputs.

The Riemannian metric of a stochastic manifold with a mean `µ` and standard
deviation `σ` is given by

M̲̲ = J̲̲_µᵀ J̲̲_µ + J̲̲_σᵀ J̲̲_σ 

where J̲̲_µ and J̲̲_σ are the Jacobians of `µ` and `σ` respectively with respect
to the input.

# Arguments
- `decoder::SplitGaussianDecoder`: A VAE decoder structure that has separate
  neural networks for determining both the mean and standard deviation of the
  latent space.
- `z::AbstractVector`: A vector specifying the latent space input to the decoder
  where the metric should be evaluated.

# Returns
- `M̲̲::Matrix`: The Riemannian metric matrix evaluated at `z`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::SplitGaussianDecoder, z::AbstractVector)
    # Compute Jacobian with respect to the input for the mean µ
    jac_µ = first(Zygote.jacobian(decoder.µ, z))

    # Compute Jacobian with respect to the input for the standard deviation
    jac_σ = first(Zygote.jacobian(decoder.σ, z))

    # Compute the metric
    return jac_µ' * jac_µ + jac_σ' * jac_σ
end # function

# ------------------------------------------------------------------------------

@doc raw"""
        pullback_metric(decoder::AbstractDecoder, z::AbstractMatrix)

Compute the Riemannian metric (pull-back metric) for each column of `z` on a
manifold defined by the decoder structure of a variational autoencoder (VAE).
The metric is evaluated based on the outputs of the decoder with respect to its
inputs. 

This function applies the `pullback_metric` function to each column of `z` using
broadcasting.

# Arguments
- `decoder::AbstractDecoder`: A decoder structure containing a neural network
  model that defines the manifold.
- `z::AbstractMatrix`: A matrix where each column represents a point in the
  latent space where the metric should be evaluated.

# Returns
- `M::AbstractArray`: An array of Riemannian metric matrices, one for each
  column of `z`.

# Notes
- The Riemannian metric provides a measure of the "stretching" or "shrinking"
  effect of the decoder at a particular point in the latent space. It is crucial
  for understanding the geometric properties of the learned latent space in
  VAEs.
- The metric is computed using the Jacobian of the decoder, which is obtained
  through automatic differentiation using `Zygote.jl`.

# Citation
> Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
> Curvature of Deep Generative Models. Preprint at
> http://arxiv.org/abs/1710.11379 (2021).
"""
function pullback_metric(decoder::AbstractDecoder, z::AbstractMatrix)
    # Compute pullback metric for each column of z
    return reduce(
        (x, y) -> cat(x, y, dims=3), pullback_metric.(Ref(decoder), eachcol(z))
    )
end # function