using CUDA
using AutoEncoderToolkit
using Test
using Flux

if CUDA.functional()

    @testset "AutoEncoderToolkitCUDAExt" begin
        @testset "utils.jl" begin
            @testset "vec_to_ltri" begin
                @testset "CUDA.CuVector input" begin
                    diag = CUDA.CuVector{Float32}([1, 2, 3])
                    lower = CUDA.CuVector{Float32}([4, 5, 6])
                    result = AutoEncoderToolkit.utils.vec_to_ltri(diag, lower)
                    expected = CUDA.CuMatrix{Float32}([1 0 0; 4 2 0; 5 6 3])
                    @test result ≈ expected
                end

                @testset "CUDA.CuMatrix input" begin
                    diag = CUDA.CuMatrix{Float32}([1 4; 2 5; 3 6])
                    lower = CUDA.CuMatrix{Float32}([7 10; 8 11; 9 12])
                    result = AutoEncoderToolkit.utils.vec_to_ltri(diag, lower)
                    expected = cat([1 0 0; 7 2 0; 8 9 3], [4 0 0; 10 5 0; 11 12 6], dims=3)
                    @test result ≈ CUDA.CuArray(expected)
                end
            end

            @testset "slogdet" begin
                A = CUDA.randn(Float32, 3, 3, 5)
                A = Flux.batched_mul(A, Flux.batched_transpose(A))  # Make positive definite
                result = AutoEncoderToolkit.utils.slogdet(A)
                @test length(result) == 5
                @test eltype(result) <: Float32
            end

            @testset "sample_MvNormalCanon" begin
                @testset "CUDA.CuMatrix input" begin
                    Σ⁻¹ = CUDA.randn(Float32, 3, 3)
                    Σ⁻¹ = Σ⁻¹ * Σ⁻¹'  # Make positive definite
                    result = AutoEncoderToolkit.utils.sample_MvNormalCanon(Σ⁻¹)
                    @test size(result) == (3,)
                    @test eltype(result) <: Float32
                end

                @testset "CUDA.CuArray{3} input" begin
                    Σ⁻¹ = CUDA.randn(Float32, 3, 3, 5)
                    Σ⁻¹ = Flux.batched_mul(Σ⁻¹, Flux.batched_transpose(Σ⁻¹))  # Make positive definite
                    result = AutoEncoderToolkit.utils.sample_MvNormalCanon(Σ⁻¹)
                    @test size(result) == (3, 5)
                    @test eltype(result) <: Float32
                end
            end

            @testset "unit_vectors" begin
                @testset "CUDA.CuVector input" begin
                    x = CUDA.randn(Float32, 5)
                    result = AutoEncoderToolkit.utils.unit_vectors(x)
                    @test length(result) == 5
                    @test all(v -> sum(v .^ 2) ≈ 1, result)
                end

                @testset "CUDA.CuMatrix input" begin
                    x = CUDA.randn(Float32, 5, 3)
                    result = AutoEncoderToolkit.utils.unit_vectors(x)
                    @test length(result) == 5
                    @test all(m -> all(sum(m .^ 2, dims=1) .≈ 1), result)
                end
            end

            @testset "finite_difference_gradient" begin
                f(x) = sum(x .^ 2)
                x = CUDA.randn(Float32, 5)
                result = AutoEncoderToolkit.utils.finite_difference_gradient(f, x)
                @test size(result) == size(x)
                all(abs.(result - 2 * x) .< 1e-2)
            end
        end

        @testset "vae.jl" begin
            @testset "reparameterize" begin
                μ = CUDA.randn(Float32, 5)
                logσ = CUDA.randn(Float32, 5)
                result = AutoEncoderToolkit.VAEs.reparameterize(μ, logσ)
                @test size(result) == size(μ)
                @test eltype(result) <: Float32

                σ = exp.(logσ)
                result_nolog = AutoEncoderToolkit.VAEs.reparameterize(μ, σ, log=false)
                @test size(result_nolog) == size(μ)
                @test eltype(result_nolog) <: Float32
            end
        end

        @testset "mmdvae.jl" begin
            @testset "Kernel functions" begin
                # Define input data
                x = CUDA.randn(Float32, 10, 10)
                y = CUDA.randn(Float32, 10, 20)

                @testset "gaussian_kernel" begin
                    result = AutoEncoderToolkit.MMDVAEs.gaussian_kernel(x, x)
                    @test isa(result, CUDA.CuMatrix{Float32})
                    @test size(result) == (10, 10)

                    result = AutoEncoderToolkit.MMDVAEs.gaussian_kernel(x, y)
                    @test isa(result, CUDA.CuMatrix{Float32})
                    @test size(result) == (10, 20)
                end # @testset "gaussian_kernel"

                @testset "mmd_div" begin
                    result = AutoEncoderToolkit.MMDVAEs.mmd_div(x, x)
                    @test isa(result, Float32)

                    result = AutoEncoderToolkit.MMDVAEs.mmd_div(x, y)
                    @test isa(result, Float32)
                end # @testset "mmd_div"
            end # @testset "Kernel functions"
        end

        @testset "rhvae.jl" begin
            @testset "G_inv" begin
                z = CUDA.randn(Float32, 3, 5)
                centroids_latent = CUDA.randn(Float32, 3, 10)
                M = CUDA.randn(Float32, 3, 3, 10)
                T = 1.0f0
                λ = 0.1f0
                result = AutoEncoderToolkit.RHVAEs._G_inv(CUDA.CuArray, z, centroids_latent, M, T, λ)
                @test size(result) == (3, 3, 5)
                @test eltype(result) <: Float32
            end

            @testset "metric_tensor" begin
                # This test would require setting up a mock RHVAE or NamedTuple
                # The test should check if the function runs without error
                # and returns the expected type and shape of output
            end

            @testset "train!" begin
                # This test would require setting up a mock RHVAE and data
                # The test should check if the function runs without error
                # and returns the expected type of output
            end
        end
    end
end # if CUDA.functional()