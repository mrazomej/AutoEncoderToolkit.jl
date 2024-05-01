using AutoEncoderToolkit.VAEs

@doc raw"""
        reparameterize(µ, σ; log::Bool=true)

Reparameterize the latent space using the given mean (`µ`) and (log)standard
deviation (`σ` or `logσ`), employing the reparameterization trick. This function
helps in sampling from the latent space in variational autoencoders (or similar
models) while keeping the gradient flow intact.

# Arguments
- `µ::CuVecOrMat{Float32}`: The mean of the latent space. If it is a
  vector, it represents the mean for a single data point. If it is a matrix,
  each column corresponds to the mean for a specific data point, and each row
  corresponds to a dimension of the latent space.
- `σ::CuVecOrMat{Float32}`: The (log )standard deviation of the latent
  space. Like `µ`, if it's a vector, it represents the (log) standard deviation
  for a single data point. If a matrix, each column corresponds to the (log)
  standard deviation for a specific data point.

# Optional Keyword Arguments
- `log::Bool=true`: Boolean indicating whether the provided standard deviation
  is in log scale or not. If `true` (default), then `σ = exp(logσ)` is computed.
- `T::Type=Float32`: The type of the output array.

# Returns
An array containing samples from the reparameterized latent space, obtained by
applying the reparameterization trick on the provided mean and log standard
deviation.

# Description
This function employs the reparameterization trick to sample from the latent
space without breaking the gradient flow. The trick involves expressing the
random variable as a deterministic variable transformed by a standard random
variable, allowing for efficient backpropagation through stochastic nodes.

# Example
```julia
µ = Float32[0.5, 0.2]
logσ = Float32[-0.1, -0.2]
sampled_point = reparameterize(µ, logσ)
```
# Notes
Ensure that the dimensions of µ and logσ match, and that the chosen prior
distribution is consistent with the expectations of the latent space.

# Citation
Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. Preprint at
http://arxiv.org/abs/1312.6114 (2014).
"""
function VAEs.reparameterize(
    µ::CUDA.CuVecOrMat,
    σ::CUDA.CuVecOrMat;
    log::Bool=true,
    T::Type=Float32
)
    # Sample random Gaussian number
    r = ChainRulesCore.ignore_derivatives() do
        CUDA.randn(T, size(µ)...)
    end
    # Check if logσ is provided
    if log
        # Sample random latent variable point estimates given the mean and log
        # standard deviation
        return µ .+ r .* exp.(σ)
    else
        # Sample random latent variable point estimates given the mean and
        # standard deviation
        return µ .+ r .* σ
    end # if
end # function