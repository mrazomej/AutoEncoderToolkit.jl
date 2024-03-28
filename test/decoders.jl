## ============= decoders module =============
println("Testing decoders module:\n")

# Import AutoEncode.jl module to be tested
using AutoEncode

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
    decoder = AutoEncode.Decoder(
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

@testset "SimpleDecoder" begin
    # For reproducibility
    Random.seed!(42)

    # Create a SimpleDecoder instance with sample parameters
    decoder = AutoEncode.SimpleDecoder(
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

@testset "JointLogDecoder" begin
    # Create a JointLogDecoder instance with sample parameters
    decoder = AutoEncode.JointLogDecoder(
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

@testset "JointDecoder" begin
    # Create a JointDecoder instance with sample parameters
    decoder = AutoEncode.JointDecoder(
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

@testset "SplitLogDecoder" begin
    # Create a SplitLogDecoder instance with sample parameters
    decoder = AutoEncode.SplitLogDecoder(
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

@testset "SplitDecoder" begin
    # Create a SplitDecoder instance with sample parameters
    decoder = AutoEncode.SplitDecoder(
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
    decoder = AutoEncode.BernoulliDecoder(
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
    decoder = AutoEncode.CategoricalDecoder(
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
        decoder = AutoEncode.BernoulliDecoder(
            28 * 28, 64, [256, 128], [relu, relu], sigmoid
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "CategoricalDecoder" begin
        # Create a CategoricalDecoder instance with sample parameters
        decoder = AutoEncode.CategoricalDecoder(
            10, 64, [256, 128], [relu, relu], softmax
        )

        @testset "vector input" begin
            # Generate random input data
            x = Flux.onehotbatch(rand(1:10, 1), 1:10)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SimpleDecoder" begin
        # Create a SimpleDecoder instance with sample parameters
        decoder = AutoEncode.SimpleDecoder(
            28 * 28, 64, [256, 128], [relu, relu], tanh
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "JointLogDecoder" begin
        # Create a JointLogDecoder instance with sample parameters
        decoder = AutoEncode.JointLogDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [tanh, identity]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SplitLogDecoder" begin
        # Create a SplitLogDecoder instance with sample parameters
        decoder = AutoEncode.SplitLogDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, relu]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "JointDecoder" begin
        # Create a JointDecoder instance with sample parameters
        decoder = AutoEncode.JointDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [tanh, softplus]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end

    @testset "SplitDecoder" begin
        # Create a SplitDecoder instance with sample parameters
        decoder = AutoEncode.SplitDecoder(
            28 * 28, 64, [256, 128], [relu, relu], [256, 128], [relu, softplus]
        )

        @testset "vector input" begin
            # Generate random input data
            x = rand(Float32, 28 * 28)
            z = randn(Float32, 64)

            # Pass the input through the decoder
            decoder_output = decoder(z)

            # Compute the log-likelihood
            loglikelihood = AutoEncode.decoder_loglikelihood(
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
            loglikelihood = AutoEncode.decoder_loglikelihood(
                x, z, decoder, decoder_output
            )

            # Check if the log-likelihood is a scalar
            @test loglikelihood isa Vector{Float32}
        end
    end
end