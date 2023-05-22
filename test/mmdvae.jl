##
println("Testing MMD-VAEs module:\n")
##

import Revise

# Import AutoEncode.jl module to be tested
import AutoEncode.MMDVAEs
import AutoEncode.VAEs: vae_init
# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase

Random.seed!(42)
##

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 1_000
# Define number of epochs
n_epoch = 10_000
# Define how often to compute error
n_error = 100
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 3
# Define number of neurons in non-linear hidden layers
n_neuron = 20
# Define dimensionality of latent space
latent_dim = 1
# Define parameter scheduler
epoch_change = [1, 10^4, 10^5, 5 * 10^5, 10^6]
learning_rates = [10^-4, 10^-5, 10^-6, 10^-5.5, 10^-6];

##

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

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=2)

# Center data to have mean zero and standard deviation one
data = StatsBase.transform(dt, data);

##

# Define VAE architecture

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

##

println("Testing MMD-VAE initialization and training...")

# Initialize autoencoder
vae = vae_init(
    n_input,
    latent_dim,
    latent_activation,
    output_activation,
    encoder,
    encoder_activation,
    decoder,
    decoder_activation
)

##

# Test Gaussian Kernel function
@test isa(MMDVAEs.gaussian_kernel(data, data), AbstractMatrix)

# Test MMD divergence function
@test isa(MMDVAEs.mmd_div(data[:, 1], data[:, 1]), AbstractFloat)

# Test logP_mmd_ratio function
@test isa(MMDVAEs.logP_mmd_ratio(vae, data), AbstractFloat)

##

# Test loss function
@test isa(MMDVAEs.loss(vae, data), AbstractFloat)

##

# Explicit setup of optimizer
opt_state = Flux.Train.setup(
    Flux.Optimisers.Adam(1E-1),
    vae
)

# Extract parameters
params_init = deepcopy(Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder))

# Loop through a couple of epochs
for epoch = 1:10
    # Test training function
    MMDVAEs.train!(vae, data, opt_state)
end # for

# Extract modified parameters
params_end = deepcopy(Flux.params(vae.encoder, vae.µ, vae.logσ, vae.decoder))

@test all(params_init .!= params_end)

println("Passed tests for MMDVAEs module!\n")
##