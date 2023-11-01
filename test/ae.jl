##
println("Testing AEs module:\n")
##

# Import AutoEncode.jl package
import AutoEncode.AEs
# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase

Random.seed!(42)

## =============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 50
# Define number of epochs
n_epoch = 10
# Define how often to compute error
n_error = 100
# Define batch size
n_batch = 32
# Define number of hidden layers
n_hidden = 2
# Define number of neurons in non-linear hidden layers
n_neuron = 10
# Define dimensionality of latent space
n_latent = 2
# Define parameter scheduler
epoch_change = [1, 10^4, 10^5, 5 * 10^5, 10^6]
learning_rates = [10^-4, 10^-5, 10^-6, 10^-5.5, 10^-6];

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

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=2)

# Center data to have mean zero and standard deviation one
data = StatsBase.transform(dt, data);

## =============================================================================

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
encoder = AEs.Encoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Test if it returns the right type
@test isa(encoder, AEs.Encoder)

# Initialize decoder
decoder = AEs.Decoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Test if it returns the right type
@test isa(decoder, AEs.Decoder)

# Define VAE
ae = AEs.AE(encoder, decoder)

# Test if it returns the right type
@test isa(ae, AEs.AE)

# Test that reconstruction works
@test isa(ae(data), AbstractVecOrMat)

# Explicit setup of optimizer
opt_state = Flux.Train.setup(
    Flux.Optimisers.Adam(1E-3),
    ae
)

# Extract parameters
params_init = deepcopy(
    Flux.params(
        ae.encoder.encoder,
        ae.decoder.decoder
    )
)

# Loop through a couple of epochs
losses = Float64[]  # Track the loss
for epoch = 1:10
    Random.seed!(42)
    # Test training function
    AEs.train!(ae, data, opt_state)
    push!(losses, StatsBase.mean(AEs.mse_loss.(Ref(ae), eachcol(data))))
end


# Check if loss is decreasing
@test all(diff(losses) ≠ 0)

# Extract modified parameters
params_end = deepcopy(
    Flux.params(
        ae.encoder.encoder,
        ae.decoder.decoder
    )
)

# Check that parameters have significantly changed
threshold = 1e-5
# Check if any parameter has changed significantly
@test all([
    all(abs.(x .- y) .> threshold) for (x, y) in zip(params_init, params_end)
])

println("Passed tests for AEs module!\n")