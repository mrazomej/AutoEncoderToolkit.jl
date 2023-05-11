@doc raw"""
    recon(ae, input)

Function to evaluate an input on an `AE` autoencoder
"""
function recon(ae::AE, input::AbstractVector{<:AbstractFloat})
    return Flux.Chain(ae.encoder..., ae.decoder...)(input)
end # function

@doc raw"""
    recon(irmae, input)

Function to evaluate an input on an `IRMAE` autoencoder
"""
function recon(irmae::IRMAE, input::AbstractVector{<:AbstractFloat})
    return Flux.Chain(
        irmae.encoder..., irmae.linear..., irmae.decoder...
    )(input)
end # function

@doc raw"""
`recon(vae, input; latent)`

This function performs three steps:
1. passes an input `x` through the `encoder`, 
2. samples the latent variable by using the reparametrization trick,
3. reconstructs the input from the latent variables using the `decoder`.

# Arguments
- `vae::VAE`: Variational autoencoder struct with all components.
- `input::AbstractVector{Float32}`: Input to the neural network.

## Optional Arguments
- `latent::Bool=false`: Boolean indicating if the latent variables should be
returned as part of the output or not.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
the input when mapped to the latent space.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
autoencoder. Note: This last point depends on a random sampling step, thus it
will change every time.
"""
function recon(
    vae::VAE,
    input::AbstractVector{Float32};
    latent::Bool=false
)
    # 1. Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(input)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(input)

    # 2. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
    ) .* exp.(logσ)

    # 3. Run sampled latent variables through decoder and return values
    if latent
        return z, vae.decoder(z)
    else
        return vae.decoder(z)
    end # if
end # function

@doc raw"""
`recon(infomaxvae, input; latent)`

This function performs three steps:
1. passes an input `x` through the `encoder`, 
2. samples the latent variable by using the reparametrization trick,
3. reconstructs the input from the latent variables using the `decoder`.

# Arguments
- `infomaxvae::InfoMaxVAE`: InfoMax Variational autoencoder struct with all
  components.
- `input::AbstractVector{Float32}`: Input to the neural network.

## Optional Arguments
- `latent::Bool=false`: Boolean indicating if the latent variables should be
returned as part of the output or not.

# Returns
- `µ::Vector{Float32}`: Array containing the mean value of the input when mapped
to the latent space.
- `logσ::Vector{Float32}`: Array containing the log of the standard deviation of
the input when mapped to the latent space.
- `x̂::Vector{Float32}`: The reconstructed input `x` after passing through the
autoencoder. Note: This last point depends on a random sampling step, thus it
will change every time.
"""
function recon(
    vae::InfoMaxVAE,
    input::AbstractVector{Float32};
    latent::Bool=false
)
    # 1. Map input to mean and log standard deviation of latent variables
    µ = Flux.Chain(vae.encoder..., vae.µ)(input)
    logσ = Flux.Chain(vae.encoder..., vae.logσ)(input)

    # 2. Sample random latent variable point estimate given the mean and
    #    standard deviation
    z = µ .+ Random.rand(
        Distributions.Normal{Float32}(0.0f0, 1.0f0), length(µ)
    ) .* exp.(logσ)

    # 3. Run sampled latent variables through decoder and return values
    if latent
        return z, vae.decoder(z)
    else
        return vae.decoder(z)
    end # if
end # function