println("\nTesting decoders...\n")

# Import AutoEncoderToolkit.jl module to be tested
using AutoEncoderToolkit

# Import necessary libraries for testing
using Flux
import LinearAlgebra
import Random
import StatsBase

## =============================================================================

@testset "Decoder" begin
    # For reproducibility
    Random.seed!(42)
    # Create a Decoder instance with sample parameters
    decoder = AutoEncoderToolkit.Decoder(
        4 * 4, 2, [10, 10], [relu, relu], sigmoid
    )

    # Generate random input data
    z = randn(Float32, 2)

    # Pass the input through the decoder
    x_reconstructed = decoder(z)

    # Check if the output has the correct size
    @test size(x_reconstructed) == (4 * 4,)

    # Check if the output values are between 0 and 1 (since sigmoid is used)
    @test all(0 .≤ x_reconstructed .≤ 1)
end

## =============================================================================

@testset "SimpleGaussianDecoder" begin
    # For reproducibility
    Random.seed!(42)

    # Create a SimpleGaussianDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.SimpleGaussianDecoder(
        28 * 28, 64, [256, 128], [relu, relu], tanh
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:μ,)}

    # Check if the output has the correct size
    @test size(decoder_output.μ) == (28 * 28,)
end

## =============================================================================

@testset "JointGaussianLogDecoder" begin
    # Create a JointGaussianLogDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.JointGaussianLogDecoder(
        28 * 28, 64, [256, 128], [relu, relu], [tanh, identity]
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:μ, :logσ)}

    # Check if the output has the correct size
    @test size(decoder_output.μ) == (28 * 28,)
    @test size(decoder_output.logσ) == (28 * 28,)
end

## =============================================================================

@testset "JointGaussianDecoder" begin
    # Create a JointGaussianDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.JointGaussianDecoder(
        28 * 28, 64, [256, 128], [relu, relu], [tanh, softplus]
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:μ, :σ)}

    # Check if the output has the correct size
    @test size(decoder_output.μ) == (28 * 28,)
    @test size(decoder_output.σ) == (28 * 28,)
end

## =============================================================================

@testset "SplitGaussianLogDecoder" begin
    # Create a SplitGaussianLogDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.SplitGaussianLogDecoder(
        28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, relu]
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:μ, :logσ)}

    # Check if the output has the correct size
    @test size(decoder_output.μ) == (28 * 28,)
    @test size(decoder_output.logσ) == (28 * 28,)
end

## =============================================================================

@testset "SplitGaussianDecoder" begin
    # Create a SplitGaussianDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.SplitGaussianDecoder(
        28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, softplus]
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:μ, :σ)}

    # Check if the output has the correct size
    @test size(decoder_output.μ) == (28 * 28,)
    @test size(decoder_output.σ) == (28 * 28,)
end

## =============================================================================

@testset "BernoulliDecoder" begin
    # Create a BernoulliDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.BernoulliDecoder(
        28 * 28, 64, [256, 128], [relu, relu], sigmoid
    )

    # Generate random input data
    z = randn(Float32, 64)

    # Pass the input through the decoder
    decoder_output = decoder(z)

    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:p,)}

    # Check if the output has the correct size
    @test size(decoder_output.p) == (28 * 28,)

    # Check if the output values are between 0 and 1 (since sigmoid is used)
    @test all(0 .<= decoder_output.p .<= 1)
end

## =============================================================================

@testset "CategoricalDecoder" begin
    # Create a CategoricalDecoder instance with sample parameters
    decoder = AutoEncoderToolkit.CategoricalDecoder(
        [4, 4], 2, [256, 128], [relu, relu], softmax
    )
    # Generate random input data
    z = randn(Float32, 2)
    # Pass the input through the decoder
    decoder_output = decoder(z)
    # Check if the output is a NamedTuple with the correct keys
    @test decoder_output isa NamedTuple{(:p,)}
    # Check if the output has the correct size
    @test size(decoder_output.p) == (4, 4)
    # Check if the output values sum to 1 (since softmax is used)
    @test all(isapprox.(sum(decoder_output.p, dims=1), 1, atol=1e-6))
end

## =============================================================================

@testset "decoder_loglikelihood" begin
    @testset "BernoulliDecoder" begin
        # Create a BernoulliDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.BernoulliDecoder(
            28 * 28, 64, [256, 128], [relu, relu], sigmoid
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "CategoricalDecoder" begin
        # Create a CategoricalDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.CategoricalDecoder(
            10, 64, [256, 128], [relu, relu], softmax
        )

        @testset "vector input" begin
            # Generate random input data
            x = Flux.onehotbatch(rand(1:10, 1), 1:10)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = Flux.onehotbatch(rand(1:10, 10), 1:10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SimpleGaussianDecoder" begin
        # Create a SimpleGaussianDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.SimpleGaussianDecoder(
            28 * 28, 64, [256, 128], [relu, relu], tanh
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "JointGaussianLogDecoder" begin
        # Create a JointGaussianLogDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.JointGaussianLogDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [tanh, identity]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SplitGaussianLogDecoder" begin
        # Create a SplitGaussianLogDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.SplitGaussianLogDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, relu]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "JointGaussianDecoder" begin
        # Create a JointGaussianDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.JointGaussianDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [tanh, softplus]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SplitGaussianDecoder" begin
        # Create a SplitGaussianDecoder instance with sample parameters
        decoder = AutoEncoderToolkit.SplitGaussianDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, softplus]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Float32
        end

        @testset "matrix input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28, 10)
            z = randn(Float32, 64, 10)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncoderToolkit.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end
end

println("\nAll tests passed!\n")