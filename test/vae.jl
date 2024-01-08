##
println("Testing VAEs module:\n")
##

# Import AutoEncode.jl module to be tested
import AutoEncode.VAEs
# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

@testset "reparameterize function" begin
    # Define some inputs
    µ = Float32[0.5, 0.2]
    logσ = Float32[-0.1, -0.2]
    prior = Distributions.Normal{Float32}(0.0f0, 1.0f0)

    # Test with log scale standard deviation
    @testset "log scale standard deviation" begin
        result = VAEs.reparameterize(
            µ, logσ, prior=prior, n_samples=1, log=true
        )
        @test size(result) == size(µ)
        @test typeof(result) == typeof(µ)
    end

    # Test with standard deviation (not log scale)
    @testset "standard deviation (not log scale)" begin
        σ = exp.(logσ)
        result = VAEs.reparameterize(
            µ, σ, prior=prior, n_samples=1, log=false
        )
        @test size(result) == size(µ)
        @test typeof(result) == typeof(µ)
    end

    # Test with multiple samples
    @testset "multiple samples" begin
        n_samples = 5
        result = VAEs.reparameterize(
            µ, logσ, prior=prior, n_samples=n_samples, log=true
        )
        @test size(result) == (size(µ)..., n_samples)
        @test typeof(result) == Array{Float32,2}
    end
end
## =============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 10
# Define number of hidden layers
n_hidden = 2
# Define number of neurons in non-linear hidden layers
n_neuron = 3
# Define dimensionality of latent space
n_latent = 2

## =============================================================================

println("Generating synthetic data...")

# Define function
f(x₁, x₂) = 10 * exp(-(x₁^2 + x₂^2))

# Defien radius
radius = 3

# Sample random radius
r_rand = radius .* sqrt.(Random.rand(n_data))

# Sample random angles
θ_rand = 2π .* Random.rand(n_data)

# Convert form polar to cartesian coordinates
x_rand = Float32.(r_rand .* cos.(θ_rand))
y_rand = Float32.(r_rand .* sin.(θ_rand))
# Feed numbers to function
z_rand = f.(x_rand, y_rand)

# Compile data into matrix
data = Matrix(hcat(x_rand, y_rand, z_rand)')

## =============================================================================

@testset "VAE{JointLogEncoder, SimpleDecoder}" begin

    # Define latent space activation function
    latent_activation = Flux.identity
    # Define output layer activation function
    output_activation = Flux.identity

    # Define encoder layer and activation functions
    encoder_neurons = repeat([n_neuron], n_hidden)
    encoder_activation = repeat([Flux.swish], n_hidden)

    # Define decoder layer and activation function
    decoder_neurons = repeat([n_neuron], n_hidden)
    decoder_activation = repeat([Flux.swish], n_hidden)

    # Initialize encoder
    encoder = VAEs.JointLogEncoder(
        n_input,
        n_latent,
        encoder_neurons,
        encoder_activation,
        latent_activation
    )

    # Initialize decoder
    decoder = VAEs.SimpleDecoder(
        n_input,
        n_latent,
        decoder_neurons,
        decoder_activation,
        output_activation
    )

    # Define VAE
    vae = encoder * decoder

    @testset "Type checking" begin
        # Test if it returns the right type
        @test isa(encoder, VAEs.JointLogEncoder)
        # Test if it returns the right type
        @test isa(decoder, VAEs.SimpleDecoder)
        # Test if it returns the right type
        @test isa(vae, VAEs.VAE)
    end # Type checking


    @testset "VAE Forward Pass" begin

        # Define some inputs
        prior = Distributions.Normal{Float32}(0.0f0, 1.0f0)

        # Test with latent=false
        @testset "latent=false" begin
            # Test with latent=false and single data point
            result = vae(data[:, 1], prior=prior, latent=false, n_samples=1)
            @test typeof(result) <: Vector{Float32}

            # Test with latent=false and multiple data points
            result = vae(data, prior=prior, latent=false, n_samples=1)
            @test typeof(result) <: Matrix{Float32}
        end

        # Test with latent=true
        @testset "latent=true" begin
            # Test with latent=true and single data point
            result = vae(data[:, 1], prior=prior, latent=true, n_samples=1)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractVector{Float32}))

            # Test with latent=true and multiple data points
            result = vae(data, prior=prior, latent=true, n_samples=1)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractMatrix{Float32}))

        end

        # Test with multiple samples
        @testset "multiple samples" begin
            # Test with latent=true and multiple data points
            n_samples = 5
            result = vae(data[:, 1], prior=prior, latent=true, n_samples=n_samples)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractArray{Float32}))

            # Test with latent=true and multiple data points
            result = vae(data, prior=prior, latent=true, n_samples=n_samples)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractArray{Float32}))
        end
    end


    # Test reconstruction
    @test isa(VAEs.loss(vae, data[:, 1]), AbstractFloat)

    # Explicit setup of optimizer
    opt_state = Flux.Train.setup(
        Flux.Optimisers.Adam(1E-3),
        vae
    )

    # Extract parameters
    params_init = deepcopy(Flux.params(vae))

    # Loop through a couple of epochs
    losses = Float32[]  # Track the loss
    for epoch = 1:10
        Random.seed!(42)
        # Test training function
        VAEs.train!(vae, data, opt_state)
        push!(losses, VAEs.loss(vae, data))
    end

    # Check if loss is decreasing
    @test all(diff(losses) ≠ 0)

    # Extract modified parameters
    params_end = deepcopy(Flux.params(vae))

    # Check that parameters have significantly changed
    threshold = 1e-5
    # Check if any parameter has changed significantly
    @test all([
        all(abs.(x .- y) .> threshold) for (x, y) in zip(params_init, params_end)
    ])
end

## =============================================================================

@testset "Testing VAE = JointLogEncoder + JointLogDecoder" begin

    # Define latent space activation function
    latent_activation = Flux.identity
    # Define output layer activation function
    output_activation = Flux.identity

    # Define encoder layer and activation functions
    encoder_neurons = repeat([n_neuron], n_hidden)
    encoder_activation = repeat([Flux.swish], n_hidden)

    # Define decoder layer and activation function
    decoder_neurons = repeat([n_neuron], n_hidden)
    decoder_activation = repeat([Flux.swish], n_hidden)

    # Initialize encoder
    encoder = VAEs.JointLogEncoder(
        n_input,
        n_latent,
        encoder_neurons,
        encoder_activation,
        latent_activation
    )

    # Test if it returns the right type
    @test isa(encoder, VAEs.JointLogEncoder)

    # Initialize decoder
    decoder = VAEs.JointLogDecoder(
        n_input,
        n_latent,
        decoder_neurons,
        decoder_activation,
        output_activation
    )

    # Test if it returns the right type
    @test isa(decoder, VAEs.JointLogDecoder)

    # Define VAE
    vae = encoder * decoder

    # Test if it returns the right type
    @test isa(vae, VAEs.VAE)

    # Test if reconstruction returns the right type of data
    @test all(isa.(vae(data), Ref(AbstractVecOrMat)))

    # Test reconstruction
    @test isa(VAEs.loss(vae, data[:, 1]), AbstractFloat)

    # Explicit setup of optimizer
    opt_state = Flux.Train.setup(
        Flux.Optimisers.Adam(1E-3),
        vae
    )

    # Extract parameters
    params_init = deepcopy(Flux.params(vae))

    # Loop through a couple of epochs
    losses = Float64[]  # Track the loss
    for epoch = 1:10
        Random.seed!(42)
        # Test training function
        VAEs.train!(vae, data, opt_state)
        push!(losses, VAEs.loss(vae, data))
    end

    # Check if loss is decreasing
    @test all(diff(losses) ≠ 0)

    # Extract parameters
    params_end = deepcopy(Flux.params(vae))

    # Check that parameters have significantly changed
    threshold = 1e-5
    # Check if any parameter has changed significantly
    @test all([
        all(abs.(x .- y) .> threshold) for (x, y) in zip(params_init, params_end)
    ])
end

## =============================================================================

@testset "Testing VAE = JointLogEncoder + SplitLogDecoder" begin

    # Define latent space activation function
    latent_activation = Flux.identity
    # Define output layer activation function
    output_activation = Flux.identity

    # Define encoder layer and activation functions
    encoder_neurons = repeat([n_neuron], n_hidden)
    encoder_activation = repeat([Flux.swish], n_hidden)

    # Define decoder layer and activation function
    decoder_neurons = [repeat([n_neuron], n_hidden); n_input]
    decoder_activation = [repeat([Flux.swish], n_hidden); Flux.identity]

    # Initialize encoder
    encoder = VAEs.JointLogEncoder(
        n_input,
        n_latent,
        encoder_neurons,
        encoder_activation,
        latent_activation
    )

    # Test if it returns the right type
    @test isa(encoder, VAEs.JointLogEncoder)

    # Initialize decoder
    decoder = VAEs.SplitLogDecoder(
        n_input,
        n_latent,
        decoder_neurons,
        decoder_activation,
        decoder_neurons,
        decoder_activation,
    )

    # Test if it returns the right type
    @test isa(decoder, VAEs.SplitLogDecoder)

    # Define VAE
    vae = encoder * decoder

    # Test if it returns the right type
    @test isa(vae, VAEs.VAE)

    # Test if reconstruction returns the right type of data
    @test all(isa.(vae(data), Ref(AbstractVecOrMat)))

    # Test reconstruction
    @test isa(VAEs.loss(vae, data[:, 1]), AbstractFloat)

    # Explicit setup of optimizer
    opt_state = Flux.Train.setup(
        Flux.Optimisers.Adam(1E-2),
        vae
    )

    # Extract parameters
    params_init = deepcopy(Flux.params(vae))

    # Loop through a couple of epochs
    losses = Float64[]  # Track the loss
    for epoch = 1:10
        Random.seed!(42)
        # Test training function
        VAEs.train!(vae, data, opt_state)
        push!(losses, VAEs.loss(vae, data))
    end

    # Check if loss is decreasing
    @test all(diff(losses) ≠ 0)

    # Extract parameters
    params_end = deepcopy(Flux.params(vae))

    # Check that parameters have significantly changed
    threshold = 1e-5
    # Check if any parameter has changed significantly
    @test all([
        all(abs.(x .- y) .> threshold) for (x, y) in zip(params_init, params_end)
    ])

end
## =============================================================================

println("Passed tests for VAEs module!\n")
##