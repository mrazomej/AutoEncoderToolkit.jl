println("\nTesting InfoMaxVAEs.jl...\n")
# Import AutoEncode.jl module to be tested
import AutoEncode.InfoMaxVAEs
import AutoEncode.VAEs
import AutoEncode

# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

@testset "MutualInfoChain struct" begin
    # Define input size
    size_input = 10
    # Define number of latent dimensions
    n_latent = 2
    # Define number of hidden layers
    n_hidden = 2
    # Define number of neurons in non-linear hidden layers
    n_neuron = 10

    # Define latent space activation function
    output_activation = Flux.identity

    # Define encoder layer and activation functions
    mlp_neurons = repeat([n_neuron], n_hidden)
    mlp_activation = repeat([Flux.relu], n_hidden)

    # Initialize MutualInfoChain
    mi_chain = InfoMaxVAEs.MutualInfoChain(
        size_input,
        n_latent,
        mlp_neurons,
        mlp_activation,
        output_activation,
    )

    @testset "Type checking" begin
        @test typeof(mi_chain) == InfoMaxVAEs.MutualInfoChain
    end

    @testset "Forward pass" begin
        # Define batch of data
        x = randn(Float32, size_input, 10)
        # Define batch of latent variables
        z = randn(Float32, n_latent, 10)

        # Test forward pass
        result = mi_chain(x, z)
        # Check output type
        @test isa(result, Vector{Float32})
        # Check otuput length
        @test length(result) == 10
    end # @testset "Forward pass"
end # @testset "MutualInfoChain struct"

## =============================================================================

@testset "InfoMaxVAE struct" begin
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
    encoder = InfoMaxVAEs.JointLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = InfoMaxVAEs.SimpleDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = encoder * decoder

    # Define MutualInfoChain
    mi_chain = InfoMaxVAEs.MutualInfoChain(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )

    # Initialize InfoMaxVAE
    infomaxvae = InfoMaxVAEs.InfoMaxVAE(vae, mi_chain)

    @testset "Type checking" begin
        @test typeof(infomaxvae) <: InfoMaxVAEs.InfoMaxVAE{
            VAEs.VAE{AutoEncode.JointLogEncoder,AutoEncode.SimpleDecoder}
        }
    end

    @testset "Forward pass" begin
        # Define batch of data
        x = randn(Float32, data_dim, 10)

        @testset "latent=false" begin
            # Test forward pass
            result = infomaxvae(x; latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractMatrix{Float32}))
        end # @testset "latent=false"

        @testset "latent=true" begin
            # Test forward pass
            result = infomaxvae(x; latent=true)
            @test isa(result, NamedTuple)
            @test result.vae isa NamedTuple
            @test result.mi isa Float32
        end # @testset "latent=true"
    end # @testset "Forward pass"
end # @testset "InfoMaxVAE struct"

## =============================================================================

@testset "Variational Mutual Information" begin
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
    encoder = VAEs.JointLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Define MutualInfoChain
    mi_chain = InfoMaxVAEs.MutualInfoChain(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )

    # Initialize InfoMaxVAE
    infomaxvae = InfoMaxVAEs.InfoMaxVAE(vae, mi_chain)

    # Define batch of data
    x = randn(Float32, data_dim, 10)
    # Define batch of latent variables
    z = randn(Float32, latent_dim, 10)
    # Define shuffled batch of latent variables
    z_shuffle = InfoMaxVAEs.shuffle_latent(z)

    @testset "variational_mutual_info" begin
        result = InfoMaxVAEs.variational_mutual_info(
            mi_chain, x, z, z_shuffle
        )
        @test isa(result, Float32)

        result = InfoMaxVAEs.variational_mutual_info(
            infomaxvae, x, z, z_shuffle
        )
        @test isa(result, Float32)

        result = InfoMaxVAEs.variational_mutual_info(infomaxvae, x)
        @test isa(result, Float32)
    end # @testset "variational_mutual_info"
end # @testset "Variational Mutual Information"

## =============================================================================

@testset "Loss functions" begin
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
    encoder = VAEs.JointLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Define MutualInfoChain
    mi_chain = InfoMaxVAEs.MutualInfoChain(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )

    # Initialize InfoMaxVAE
    infomaxvae = InfoMaxVAEs.InfoMaxVAE(vae, mi_chain)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    @testset "infomaxloss" begin
        result = InfoMaxVAEs.infomaxloss(vae, mi_chain, x)
        @test isa(result, Float32)

        result = InfoMaxVAEs.infomaxloss(vae, mi_chain, x, x)
        @test isa(result, Float32)
    end # @testset "infomaxloss"

    @testset "miloss" begin
        result = InfoMaxVAEs.miloss(vae, mi_chain, x)
        @test isa(result, Float32)
    end # @testset "miloss"
end # @testset "Loss functions"

## =============================================================================

@testset "InfoMaxVAE training" begin
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
    encoder = VAEs.JointLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Define MutualInfoChain
    mi_chain = InfoMaxVAEs.MutualInfoChain(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )

    # Initialize InfoMaxVAE
    infomaxvae = InfoMaxVAEs.InfoMaxVAE(vae, mi_chain)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    # Explicit setup of optimizers
    opt_infomaxvae = Flux.Train.setup(Flux.Optimisers.Adam(), infomaxvae)

    @testset "with same input and output" begin
        L = InfoMaxVAEs.train!(infomaxvae, x, opt_infomaxvae; loss_return=true)
        @test L isa Tuple{<:Number,<:Number}
    end # @testset "with same input and output"

    @testset "with different input and output" begin
        x_out = randn(Float32, data_dim, 10)
        L = InfoMaxVAEs.train!(
            infomaxvae, x, x_out, opt_infomaxvae; loss_return=true
        )
        @test L isa Tuple{<:Number,<:Number}
    end # @testset "with different input and output"
end # @testset "InfoMaxVAE training"

println("\nAll tests passed!\n")