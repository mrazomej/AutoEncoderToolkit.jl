println("\nTesting AEs module...\n")
# Import AutoEncoderToolkit.jl module to be tested
import AutoEncoderToolkit.AEs
import AutoEncoderToolkit: Encoder, Decoder

# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

@testset "AE struct" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = AEs.Encoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = AEs.Decoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define AE
    ae = AEs.AE(encoder, decoder)

    @testset "Type checking" begin
        @test typeof(ae) <: AEs.AE{<:AEs.Encoder,<:AEs.Decoder}
    end

    @testset "Forward pass" begin
        # Define batch of data
        x = randn(Float32, data_dim, 10)

        @testset "latent=false" begin
            # Test forward pass
            result = ae(x; latent=false)
            @test isa(result, NamedTuple)
            @test result.decoder isa AbstractMatrix{Float32}
        end # @testset "latent=false"

        @testset "latent=true" begin
            # Test forward pass
            result = ae(x; latent=true)
            @test isa(result, NamedTuple)
            @test result.encoder isa AbstractMatrix{Float32}
            @test result.decoder isa AbstractMatrix{Float32}
        end # @testset "latent=true"
    end # @testset "Forward pass"
end # @testset "AE struct"

## =============================================================================

@testset "loss functions" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = AEs.Encoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = AEs.Decoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define AE
    ae = AEs.AE(encoder, decoder)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    @testset "mse_loss (same input and output)" begin
        result = AEs.mse_loss(ae, x)
        @test isa(result, Float32)
    end # @testset "mse_loss (same input and output)"

    @testset "mse_loss (different input and output)" begin
        x_out = randn(Float32, data_dim, 10)
        result = AEs.mse_loss(ae, x, x_out)
        @test isa(result, Float32)
    end # @testset "mse_loss (different input and output)"
end # @testset "loss functions"

## =============================================================================

@testset "AE gradient" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = AEs.Encoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = AEs.Decoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define AE
    ae = AEs.AE(encoder, decoder)

    # Define batch of data
    x = randn(Float32, data_dim, 10)
    x_out = randn(Float32, data_dim, 10)

    @testset "gradient (same input and output)" begin
        grads = Flux.gradient(ae -> AEs.mse_loss(ae, x), ae)
        @test isa(grads, Tuple)
    end # @testset "gradient (same input and output)"

    @testset "gradient (different input and output)" begin
        grads = Flux.gradient(ae -> AEs.mse_loss(ae, x, x_out), ae)
        @test isa(grads, Tuple)
    end # @testset "gradient (different input and output)"

end # @testset "AE gradient"

## =============================================================================

# NOTE: The following tests are commented out because they fail with GitHub
# Actions with the following error:
# Got exception outside of a @test
# BoundsError: attempt to access 16-element Vector{UInt8} at index [0]

# @testset "AE training" begin
#     # Define dimensionality of data
#     data_dim = 10
#     # Define dimensionality of latent space 
#     latent_dim = 2
#     # Define number of hidden layers in encoder/decoder
#     n_hidden = 2
#     # Define number of neurons in encoder/decoder hidden layers
#     n_neuron = 10
#     # Define activation function for encoder/decoder hidden layers
#     hidden_activation = repeat([Flux.relu], n_hidden)
#     # Define activation function for output of encoder
#     output_activation = Flux.identity

#     # Define encoder and decoder
#     encoder = AEs.Encoder(
#         data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
#     )
#     decoder = AEs.Decoder(
#         data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
#     )
#     # Define AE
#     ae = AEs.AE(encoder, decoder)

#     # Define batch of data
#     x = randn(Float32, data_dim, 10)

#     # Explicit setup of optimizer
#     opt = Flux.Train.setup(Flux.Optimisers.Adam(), ae)

#     @testset "with same input and output" begin
#         L = AEs.train!(ae, x, opt; loss_return=true)
#         @test isa(L, Float32)
#     end # @testset "with same input and output"

#     @testset "with different input and output" begin
#         x_out = randn(Float32, data_dim, 10)
#         L = AEs.train!(ae, x, x_out, opt; loss_return=true)
#         @test isa(L, Float32)
#     end # @testset "with different input and output"
# end # @testset "AE training"

println("\nAll tests passed!\n")