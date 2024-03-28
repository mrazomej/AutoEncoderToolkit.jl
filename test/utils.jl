## ============= Utils module =============
println("Testing utils module:\n")

## =============================================================================

# Import AutoEncode.jl module to be tested
using AutoEncode.utils

# Import NearstNeighbors for locality_sampler tests
import NearestNeighbors

# Import ML library
import Flux

# Import basic math
import LinearAlgebra
import Random
import StatsBase

## =============================================================================

@testset "step_scheduler tests" begin
    # Define epoch_change and learning_rates for test
    epoch_change = [1, 3, 5, 7]
    learning_rates = [0.1, 0.2, 0.3, 0.4]

    # Define epoch for test
    epoch = 5
    @test utils.step_scheduler(epoch, epoch_change, learning_rates) == 0.3

    # Define epoch for test
    epoch = 8
    @test utils.step_scheduler(epoch, epoch_change, learning_rates) == 0.4

    # Define epoch for test
    epoch = 2
    @test utils.step_scheduler(epoch, epoch_change, learning_rates) == 0.2

    # Test case when epochs don't match learning rates
    epoch_change = [1, 3, 5]
    learning_rates = [0.1, 0.2]
    @test_throws ErrorException utils.step_scheduler(
        epoch, epoch_change, learning_rates
    )
end

## =============================================================================

@testset "cycle_anneal tests" begin
    # Define parameters for test
    n_epoch = 10 # Total number of epochs
    n_cycles = 2 # Number of cycles
    frac = 0.5 # Fraction of the cycle where β increases
    βmax = 1.0 # Maximum value of β
    βmin = 0.0 # Minimum value of β
    T = Float32 # Type of β


    epoch = 1 # Current epoch
    @test utils.cycle_anneal(
        epoch, n_epoch, n_cycles, frac=frac, βmax=βmax, βmin=βmin, T=T
    ) == 0.0f0

    epoch = 3
    @test cycle_anneal(epoch, n_epoch, n_cycles, frac=frac, βmax=βmax, βmin=βmin, T=T) > 0.0f0

    epoch = 9
    @test cycle_anneal(epoch, n_epoch, n_cycles, frac=frac, βmax=βmax, βmin=βmin, T=T) == 1.0f0

    frac = 1.5
    @test_throws ArgumentError cycle_anneal(epoch, n_epoch, n_cycles, frac=frac, βmax=βmax, βmin=βmin, T=T)
end

## =============================================================================

@testset "locality_sampler tests" begin
    # for reproducibility
    Random.seed!(42)
    # Generate random data
    data = rand(100, 1000)
    # Build KDTree
    dist_tree = NearestNeighbors.KDTree(data)
    # Define parameters for test
    n_primary = 10 # Number of primary samples
    n_secondary = 5 # Number of secondary samples
    k_neighbors = 50 # Number of neighbors to consider

    @testset "index=true" begin
        indices = utils.locality_sampler(
            data, dist_tree, n_primary, n_secondary, k_neighbors, index=true
        )
        @test length(indices) == n_primary + n_primary * n_secondary
        @test all(i -> 1 <= i <= size(data, 2), indices)
    end

    @testset "index=false" begin
        samples = utils.locality_sampler(
            data, dist_tree, n_primary, n_secondary, k_neighbors, index=false
        )
        @test size(samples) == (size(data, 1), n_primary + n_primary * n_secondary)
    end

    @testset "n_secondary > k_neighbors" begin
        @test_throws ErrorException utils.locality_sampler(
            data, dist_tree, n_primary, n_secondary + k_neighbors, k_neighbors,
            index=true
        )
    end
end

## =============================================================================

@testset "vec_to_ltri tests" begin
    @testset "vector inputs" begin
        diag = [1, 2, 3]
        lower = [4, 5, 6]
        expected = [1 0 0; 4 2 0; 5 6 3]
        @test utils.vec_to_ltri(diag, lower) == expected
    end

    @testset "matrix inputs" begin
        diag = [1 4; 2 5; 3 6]
        lower = [7 10; 8 11; 9 12]
        expected = cat([1 0 0; 7 2 0; 8 9 3], [4 0 0; 10 5 0; 11 12 6], dims=3)
        @test utils.vec_to_ltri(diag, lower) == expected
    end

    @testset "dimension mismatch" begin
        diag = [1 4; 2 5; 3 6]
        lower = [7 10; 8 11]
        @test_throws ErrorException utils.vec_to_ltri(diag, lower)
    end
end

## =============================================================================

@testset "vec_mat_vec_batched tests" begin
    @testset "vector inputs" begin
        v = [1, 2, 3]
        M = [4 5 6; 7 8 9; 10 11 12]
        w = [13, 14, 15]
        expected = LinearAlgebra.dot(v, M, w)
        @test typeof(utils.vec_mat_vec_batched(v, M, w)) <: Number
        @test utils.vec_mat_vec_batched(v, M, w) == expected
    end

    @testset "matrix inputs" begin
        v = [1 4; 2 5; 3 6]
        M = cat([4 7 10; 5 8 11; 6 9 12], [13 16 19; 14 17 20; 15 18 21], dims=3)
        w = [22 25; 23 26; 24 27]
        expected = [
            LinearAlgebra.dot(v[:, i], M[:, :, i], w[:, i]) for i in axes(v, 2)
        ]

        @test typeof(utils.vec_mat_vec_batched(v, M, w)) <: AbstractVector
        @test utils.vec_mat_vec_batched(v, M, w) == expected
    end

    @testset "dimension mismatch" begin
        v = [1, 2, 3]
        M = [4 5 6; 7 8 9]
        w = [10, 11, 12]
        @test_throws DimensionMismatch utils.vec_mat_vec_batched(v, M, w)
    end
end

## =============================================================================

@testset "vec_mat_vec_loop tests" begin
    @testset "vector inputs" begin
        v = [1, 2, 3]
        M = [4 5 6; 7 8 9; 10 11 12]
        w = [13, 14, 15]
        expected = LinearAlgebra.dot(v, M, w)
        @test typeof(utils.vec_mat_vec_loop(v, M, w)) <: Number
        @test utils.vec_mat_vec_loop(v, M, w) == expected
    end

    @testset "matrix inputs" begin
        v = [1 4; 2 5; 3 6]
        M = cat([4 7 10; 5 8 11; 6 9 12], [13 16 19; 14 17 20; 15 18 21], dims=3)
        w = [22 25; 23 26; 24 27]
        expected = [
            LinearAlgebra.dot(v[:, i], M[:, :, i], w[:, i]) for i in axes(v, 2)
        ]

        @test typeof(utils.vec_mat_vec_loop(v, M, w)) <: AbstractVector
        @test utils.vec_mat_vec_loop(v, M, w) == expected
    end

    @testset "dimension mismatch" begin
        v = [1, 2, 3]
        M = [4 5 6; 7 8 9]
        w = [10, 11, 12]
        @test_throws DimensionMismatch utils.vec_mat_vec_loop(v, M, w)
    end
end

## =============================================================================

# Define a test set for the centroids_kmeans function
@testset "centroids_kmeans tests" begin
    # Define a test set for the case without assignments
    @testset "without assignments" begin
        # Generate random data
        data = rand(100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmeans function without assignments
        centroids = utils.centroids_kmeans(data, n_centroids)
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (size(data, 1), n_centroids)
    end

    # Define a test set for the case with assignments
    @testset "with assignments" begin
        # Generate random data
        data = rand(100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmeans function with assignments
        centroids, assignments = utils.centroids_kmeans(
            data, n_centroids, assign=true
        )
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (size(data, 1), n_centroids)
        # Check if the length of the returned assignments is correct
        @test length(assignments) == size(data, 2)
        # Check if all assignments are greater than or equal to 1
        @test all(assignments .>= 1)
        # Check if all assignments are less than or equal to the number of centroids
        @test all(assignments .<= n_centroids)
    end

    # Define a test set for the case without reshaping centroids
    @testset "without reshaping centroids" begin
        # Generate random data
        data = rand(100, 100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmeans function without reshaping centroids
        centroids = utils.centroids_kmeans(
            data, n_centroids, reshape_centroids=false
        )
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (prod(size(data)[1:end-1]), n_centroids)
    end

    # Define a test set for the case with reshaping centroids and assignments
    @testset "with reshaping centroids and assignments" begin
        # Generate random data
        data = rand(100, 100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmeans function with reshaping centroids and assignments
        centroids, assignments = utils.centroids_kmeans(data, n_centroids, reshape_centroids=false, assign=true)
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (prod(size(data)[1:end-1]), n_centroids)
        # Check if the length of the returned assignments is correct
        @test length(assignments) == size(data, 3)
        # Check if all assignments are greater than or equal to 1
        @test all(assignments .>= 1)
        # Check if all assignments are less than or equal to the number of centroids
        @test all(assignments .<= n_centroids)
    end
end

## =============================================================================

# Define a test set for the centroids_kmedoids function
@testset "centroids_kmedoids tests" begin
    # Define a test set for the case without assignments
    @testset "without assignments" begin
        # Generate random data
        data = rand(100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmedoids function without assignments
        centroids = utils.centroids_kmedoids(data, n_centroids)
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (size(data, 1), n_centroids)
    end

    # Define a test set for the case with assignments
    @testset "with assignments" begin
        # Generate random data
        data = rand(100, 10)
        # Define the number of centroids
        n_centroids = 5
        # Call the centroids_kmedoids function with assignments
        centroids, assignments = utils.centroids_kmedoids(data, n_centroids, assign=true)
        # Check if the size of the returned centroids is correct
        @test size(centroids) == (size(data, 1), n_centroids)
        # Check if the length of the returned assignments is correct
        @test length(assignments) == size(data, 2)
        # Check if all assignments are greater than or equal to 1
        @test all(assignments .>= 1)
        # Check if all assignments are less than or equal to the number of centroids
        @test all(assignments .<= n_centroids)
    end
end

## =============================================================================

# Define a test set for the slogdet function
@testset "slogdet tests" begin
    # Define a test set for the case with a 2D matrix
    @testset "with 2D matrix" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random 2D matrix
        A = rand(3, 3)
        # Make the matrix positive-definite
        A = A * A'
        # Call the slogdet function
        logdet = utils.slogdet(A)
        # Check if the returned log determinant is correct
        @test logdet ≈ log(LinearAlgebra.det(A))
    end

    # Define a test set for the case with a 3D array
    @testset "with 3D array" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random 3D array
        A = rand(3, 3, 3)
        # Make each 2D slice of the array positive-definite
        A = Flux.batched_mul(A, Flux.batched_transpose(A))
        # Call the slogdet function
        logdets = utils.slogdet(A)
        # Check if the returned log determinants are correct
        for (logdet, slice) in zip(logdets, eachslice(A, dims=3))
            @test logdet ≈ log(LinearAlgebra.det(slice))
        end
    end
end

## =============================================================================

# Define a test set for the sample_MvNormalCanon functions
@testset "sample_MvNormalCanon tests" begin
    # Define a test set for the case with a 2D precision matrix
    @testset "with 2D precision matrix" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random 2D precision matrix
        Σ⁻¹ = rand(3, 3)
        # Make the matrix positive-definite
        Σ⁻¹ = Σ⁻¹ * Σ⁻¹'
        # Call the sample_MvNormalCanon function
        sample = utils.sample_MvNormalCanon(Σ⁻¹)
        # Check if the size of the returned sample is correct
        @test size(sample) == (size(Σ⁻¹, 1),)
    end

    # Define a test set for the case with a 3D array of precision matrices
    @testset "with 3D array of precision matrices" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random 3D array of precision matrices
        Σ⁻¹ = rand(3, 3, 3)
        # Make each 2D slice of the array positive-definite
        Σ⁻¹ = Σ⁻¹ .* permutedims(Σ⁻¹, (2, 1, 3))
        # Call the sample_MvNormalCanon function
        samples = utils.sample_MvNormalCanon(Σ⁻¹)
        # Check if the size of the returned samples is correct
        @test size(samples) == (size(Σ⁻¹, 1), size(Σ⁻¹, 3))
    end
end

## =============================================================================

# Define a test set for the unit_vector functions
@testset "unit_vector tests" begin
    # Define a test set for the case with a vector
    @testset "with vector" begin
        # Generate a random vector
        x = rand(10)
        # Call the unit_vector function
        e = utils.unit_vector(x, 5)
        # Check if the size of the returned unit vector is correct
        @test size(e) == size(x)
        # Check if the unit vector has a 1 at the correct position
        @test e[5] == 1
        # Check if all other elements are 0
        @test all(e[j] == 0 for j in 1:length(e) if j != 5)
    end

    # Define a test set for the case with a matrix
    @testset "with matrix" begin
        # Generate a random matrix
        x = rand(10, 10)
        # Call the unit_vector function
        e = utils.unit_vector(x, 5)
        # Check if the size of the returned unit vector is correct
        @test size(e) == (size(x, 1),)
        # Check if the unit vector has a 1 at the correct position
        @test e[5] == 1
        # Check if all other elements are 0
        @test all(e[j] == 0 for j in 1:length(e) if j != 5)
    end
end

## =============================================================================

# Define a test set for the unit_vectors functions
@testset "unit_vectors tests" begin
    # Define a test set for the case with a vector
    @testset "with vector" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random vector
        x = rand(10)
        # Call the unit_vectors function
        es = utils.unit_vectors(x)
        # Check if the size of the returned unit vectors is correct
        @test length(es) == length(x)
        for e in es
            @test length(e) == length(x)
        end
        # Check if each unit vector has a 1 at the correct position and 0s elsewhere
        for (i, e) in enumerate(es)
            @test e[i] == 1
            @test all(e[j] == 0 for j in 1:length(e) if j != i)
        end
    end

    # Define a test set for the case with a matrix
    @testset "with matrix" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random matrix
        x = rand(10, 5)
        # Call the unit_vectors function
        es = utils.unit_vectors(x)
        # Check if the size of the returned unit vectors is correct
        @test length(es) == size(x, 1)
        # Check if each unit vector has a 1 at the correct position and 0s elsewhere
        for (i, e) in enumerate(es)
            for j in axes(e, 2)
                @test all(e[i, :] .== 1)
            end
        end
    end
end

## =============================================================================

# Define a test set for the finite_difference_gradient functions
@testset "finite_difference_gradient tests" begin
    # Define a test set for the case with a vector
    @testset "with vector" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random vector
        x = rand(10)
        # Define a simple function for testing
        f = x -> sum(x .^ 2)
        # Call the finite_difference_gradient function
        grad = utils.finite_difference_gradient(f, x)
        # Check if the size of the returned gradient is correct
        @test size(grad) == size(x)
        # Check if the gradient is correct
        @test all(abs.(grad - 2x) .< 1e-5)
    end

    # Define a test set for the case with a matrix
    @testset "with matrix" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random matrix
        x = rand(10, 10)
        # Define a simple function for testing
        f = x -> vec(sum(x .^ 2, dims=1))
        # Call the finite_difference_gradient function
        grad = utils.finite_difference_gradient(f, x)
        # Check if the size of the returned gradient is correct
        @test size(grad) == size(x)
        # Check if the gradient is correct
        @test all(abs.(grad - 2x) .< 1e-5)
    end
end

## =============================================================================

# Define a test set for the finite_difference_gradient functions
@testset "finite_difference_gradient tests" begin
    # Define a test set for the case with a vector
    @testset "with vector" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random vector
        x = rand(10)
        # Define a simple function for testing
        f = x -> sum(x .^ 2)
        # Call the finite_difference_gradient function
        grad = utils.finite_difference_gradient(f, x)
        # Check if the size of the returned gradient is correct
        @test size(grad) == size(x)
        # Check if the gradient is correct
        @test all(abs.(grad - 2x) .< 1e-5)
    end

    # Define a test set for the case with a matrix
    @testset "with matrix" begin
        # For reproducibility
        Random.seed!(42)
        # Generate a random matrix
        x = rand(10, 10)
        # Define a simple function for testing
        f = x -> vec(sum(x .^ 2, dims=1))
        # Call the finite_difference_gradient function
        grad = utils.finite_difference_gradient(f, x)
        # Check if the size of the returned gradient is correct
        @test size(grad) == size(x)
        # Check if the gradient is correct
        @test all(abs.(grad - 2x) .< 1e-5)
    end
end