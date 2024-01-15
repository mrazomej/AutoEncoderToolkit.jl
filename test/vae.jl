##
println("Testing VAEs module:\n")
##

# Import AutoEncode.jl module to be tested
import AutoEncode.VAEs
import AutoEncode.regularization
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
n_data = 5
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

println("Defining layers structure...")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity
# Define output layer activation function
output_activations = [Flux.identity, Flux.softplus]

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder split layers and activation functions
µ_neurons = repeat([n_neuron], n_hidden)
µ_activation = repeat([Flux.swish], n_hidden)

logσ_neurons = repeat([n_neuron], n_hidden)
logσ_activation = repeat([Flux.swish], n_hidden)

σ_neurons = repeat([n_neuron], n_hidden)
σ_activation = [repeat([Flux.swish], n_hidden - 1); Flux.softplus]


## =============================================================================


println("Defining encoders...")
# Initialize JointLogEncoder
joint_log_encoder = VAEs.JointLogEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation,
)

# -----------------------------------------------------------------------------

println("Defining decoders...")

# Initialize SimpleDecoder
simple_decoder = VAEs.SimpleDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize JointLogDecoder
joint_log_decoder = VAEs.JointLogDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize SplitLogDecoder
split_log_decoder = VAEs.SplitLogDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    logσ_neurons,
    logσ_activation,
)

# Initialize JointDecoder
joint_decoder = VAEs.JointDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activations
)

# Initialize SplitDecoder
split_decoder = VAEs.SplitDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    σ_neurons,
    σ_activation,
)

@testset "Type checking" begin
    @test typeof(joint_log_encoder) == VAEs.JointLogEncoder
    @test typeof(simple_decoder) == VAEs.SimpleDecoder
    @test typeof(joint_log_decoder) == VAEs.JointLogDecoder
    @test typeof(split_log_decoder) == VAEs.SplitLogDecoder
    @test typeof(joint_decoder) == VAEs.JointDecoder
    @test typeof(split_decoder) == VAEs.SplitDecoder
    @test typeof(joint_log_encoder * simple_decoder) <: VAEs.VAE
end # @testset "Type checking"

# Collect all decoders
decoders = [
    joint_log_decoder,
    split_log_decoder,
    joint_decoder,
    split_decoder,
    simple_decoder,
]

## =============================================================================

@testset "VAE Forward Pass" begin
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    # Test with latent=false
    @testset "latent=false" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            vae = joint_log_encoder * decoder

            # Test with single data point
            result = vae(x_vector, latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractVector{Float32}))

            # Test with multiple data points
            result = vae(data, latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractMatrix{Float32}))
        end # for decoder in decoders
    end # @testset "latent=false"

    # Test with latent=false
    @testset "latent=true" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            vae = joint_log_encoder * decoder

            # Test with single data point
            result = vae(x_vector, latent=true)
            @test isa(result, NamedTuple)
            @test all(
                isa.(values(result), AbstractVector{Float32}) .||
                isa.(values(result), NamedTuple)
            )

            # Test with multiple data points
            result = vae(data, latent=true)
            @test isa(result, NamedTuple)
            @test all(
                isa.(values(result), AbstractMatrix{Float32}) .||
                isa.(values(result), NamedTuple)
            )
        end # for decoder in decoders
    end # @testset "latent=false"


    # Test with multiple samples
    @testset "multiple samples" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            vae = joint_log_encoder * decoder

            # Test with single data point
            result = vae(x_vector, latent=false, n_samples=2)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractArray{Float32}))

            # Test with multiple data points
            result = vae(data, latent=false, n_samples=2)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractArray{Float32}))
        end # for decoder in decoders
    end # @testset "multiple samples"

end # @testset "VAE Forward Pass"

## =============================================================================

@testset "loss function" begin
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    # Define VAE with any decoder
    vae = joint_log_encoder * joint_log_decoder

    @testset "kl_gaussian_encoder function" begin
        # Test with vector input
        @testset "vector input" begin
            vae_outputs = vae(x_vector; latent=true)
            result = VAEs.kl_gaussian_encoder(
                vae.encoder, x_vector, vae_outputs, n_samples=1
            )
            @test isa(result, Float32)
        end # vector input

        # Test with matrix input
        @testset "matrix input" begin
            vae_outputs = vae(x_matrix; latent=true)
            result = VAEs.kl_gaussian_encoder(
                vae.encoder, x_matrix, vae_outputs, n_samples=1
            )
            @test isa(result, Float32)
        end # matrix input
    end # kl_gaussian_encoder function

    # Loop through decoders
    for decoder in decoders
        # Define VAE
        vae = joint_log_encoder * decoder

        @testset "reconstruction_gaussian_decoder function" begin
            # Test with vector input
            @testset "vector input" begin
                vae_outputs = vae(x_vector; latent=true)
                result = VAEs.reconstruction_gaussian_decoder(
                    decoder, x_vector, vae_outputs, n_samples=1
                )
                @test isa(result, Float32)
            end # vector input

            # Test with matrix input
            @testset "matrix input" begin
                vae_outputs = vae(x_matrix; latent=true)
                result = VAEs.reconstruction_gaussian_decoder(
                    decoder, x_matrix, vae_outputs, n_samples=1
                )
                @test isa(result, Float32)
            end # matrix input
        end # reconstruction_gaussian_decoder function
    end # for decoder in decoders

    # Loop through decoders
    for decoder in decoders
        # Define VAE
        vae = joint_log_encoder * decoder

        @testset "loss function" begin
            # Test with vector input
            @testset "vector input" begin
                result = VAEs.loss(vae, x_vector)
                @test isa(result, Float32)
                result = VAEs.loss(vae, x_vector, x_vector)
                @test isa(result, Float32)

            end # vector input

            # Test with matrix input
            @testset "matrix input" begin
                result = VAEs.loss(vae, x_matrix)
                @test isa(result, Float32)
                result = VAEs.loss(vae, x_matrix, x_matrix)
                @test isa(result, Float32)

            end # matrix input

            # Test with regularization function
            # @testset "with regularization" begin
            #     reg_function = regularization.l2_regularization
            #     reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])
            #     result = VAEs.loss(
            #         vae, x_vector;
            #         reg_function=reg_function, reg_kwargs=reg_kwargs
            #     )
            #     @test isa(result, Float32)

            #     result = VAEs.loss(
            #         vae, x_vector, x_vector;
            #         reg_function=reg_function, reg_kwargs=reg_kwargs
            #     )
            #     @test isa(result, Float32)
            # end # with regularization
        end # loss function
    end # for decoder in decoders
end # @testset "loss function"

## =============================================================================

@testset "VAE training" begin
    # Define number of epochs
    n_epochs = 3
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    @testset "without regularization" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE with any decoder
            vae = deepcopy(joint_log_encoder) * decoder

            # Explicit setup of optimizer
            opt_state = Flux.Train.setup(
                Flux.Optimisers.Adam(1E-3),
                vae
            )

            # Extract parameters
            params_init = deepcopy(Flux.params(vae))

            # Loop through a couple of epochs
            losses = Float32[]  # Track the loss
            for epoch = 1:n_epochs
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
                all(abs.(x .- y) .> threshold)
                for (x, y) in zip(params_init, params_end)
            ])
        end # for decoder in decoders
    end # @testset "without regularization"

    # @testset "with regularization" begin
    #     reg_function = regularization.l2_regularization
    #     reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])
    #     # Loop through decoders
    #     for decoder in decoders
    #         # Define VAE with any decoder
    #         vae = deepcopy(joint_log_encoder) * decoder

    #         # Explicit setup of optimizer
    #         opt_state = Flux.Train.setup(
    #             Flux.Optimisers.Adam(1E-3),
    #             vae
    #         )

    #         # Extract parameters
    #         params_init = deepcopy(Flux.params(vae))

    #         # Loop through a couple of epochs
    #         losses = Float32[]  # Track the loss
    #         for epoch = 1:n_epochs
    #             Random.seed!(42)
    #             # Test training function
    #             VAEs.train!(
    #                 vae, data, opt_state;
    #                 loss_kwargs=Dict(
    #                     :reg_function => reg_function,
    #                     :reg_kwargs => reg_kwargs
    #                 )
    #             )
    #             push!(
    #                 losses,
    #                 VAEs.loss(
    #                     vae, data;
    #                     reg_function=reg_function, reg_kwargs=reg_kwargs
    #                 )
    #             )
    #         end

    #         # Check if loss is decreasing
    #         @test all(diff(losses) ≠ 0)

    #         # Extract modified parameters
    #         params_end = deepcopy(Flux.params(vae))

    #         # Check that parameters have significantly changed
    #         threshold = 1e-5
    #         # Check if any parameter has changed significantly
    #         @test all([
    #             all(abs.(x .- y) .> threshold)
    #             for (x, y) in zip(params_init, params_end)
    #         ])
    #     end # for decoder in decoders
    # end # @testset "with regularization"
end # @testset "VAE training"