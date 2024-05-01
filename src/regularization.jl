# Import Abstract Types
using ..AutoEncoderToolkit: JointGaussianLogEncoder, SimpleGaussianDecoder, JointGaussianLogDecoder, SplitGaussianLogDecoder

# ==============================================================================

@doc raw"""
        l2_regularization(outputs::NamedTuple, reg_terms::Vector{Symbol})

Compute the L2 regularization term (also known as Ridge regularization) based on
the autoencoder outputs and the `reg_terms` vector.

L2 regularization is defined as: L₂(v) = λ ∑ᵢ vᵢ²

where:
- λ is the regularization strength (not computed in this function, but typically
  applied outside).
- vᵢ represents each element of the vector or matrix v.

The primary purpose of L2 regularization is to add a penalty to the magnitude of
the model parameters. By doing so, it discourages the model from fitting the
training data too closely (overfitting) and promotes smoother decision
boundaries or function approximations. This helps in improving generalization to
new, unseen data.

# Arguments
- `outputs::NamedTuple`: NamedTuple containing outputs from the AE or VAE model.

## Optional Keyword Arguments
- `reg_terms::Vector{Symbol}`: Vector that specifies which entries from
  `outputs` to consider for regularization.

# Returns
- `reg_term::Float32`: The computed L2 regularization value.

# Notes
- Ensure that all elements in `reg_terms` are keys in `outputs`.
"""
function l2_regularization(
    outputs::NamedTuple; reg_terms::Vector{Symbol}=[:decoder_µ]
)::Float32
    # Ensyre there is at least one key
    if isempty(reg_terms)
        return 0.0f0
    end # if

    # Ensure all keys in reg_terms are in outputs
    if !all(key ∈ keys(outputs) for key in reg_terms)
        error("All keys in reg_terms must exist in outputs!")
    end # if

    # Compute the regularization term without in-place mutation
    reg_term = sum(sum(outputs[term] .^ 2) for term in reg_terms)

    return reg_term
end # function

# ==============================================================================

@doc raw"""
        min_variance_regularization(outputs::NamedTuple, σ_min::Float32=0.1f0, logσ::Bool=true)

Compute the minimum variance constraint regularization term based on the VAE
outputs and a specified minimum standard deviation (σ_min).

The regularization is defined as: 
- If `logσ` is true: L = λ ∑ᵢ max(0, log(σᵢ²) - log(σ_min²))
- If `logσ` is false: L = λ ∑ᵢ max(0, σᵢ² - σ_min²)

Where:
- λ is the regularization strength (not computed in this function, but typically
  applied outside).
- σᵢ represents the standard deviation for each output in the VAE decoder.

The primary purpose of this regularization is to prevent the variance from going
below a certain threshold, discouraging the model from making overconfident
predictions.

# Arguments
- `outputs::NamedTuple`: NamedTuple containing outputs from the VAE model.

## Optional Keyword Arguments
- `σ_min::Float32`: The minimum allowable standard deviation. Default is 0.1.
- `logσ::Bool`: If true, the regularization is computed in the log-space. If
  false, the regularization is computed in the original space. Default is true.

# Returns
- `reg_term::Float32`: The computed minimum variance constraint regularization
  value.
"""
function min_variance_regularization(
    outputs::NamedTuple; σ_min::Float32=0.1f0, logσ::Bool=true
)::Float32
    # Check if the decoder variance exists in the outputs
    if :decoder_logσ ∈ keys(outputs)
        # Extract decoder log std
        decoder_logσ = outputs[:decoder_logσ]
    elseif :decoder_σ ∈ keys(outputs)
        # Extract decoder log std
        decoder_logσ = log.(outputs[:decoder_σ])
    else
        throw(ArgumentError("The decoder standard deviation does not exist in " *
                            "outputs."))
    end # if

    # Check if logσ is true
    if logσ
        # Compute regularization term to discourage extremeley small variances
        reg_term = sum(max.(0.0f0, decoder_logσ .- log(σ_min^2)))
    else
        # Compute regularization term to discourage extremeley small variances
        reg_term = sum(max.(0.0f0, exp.(decoder_logσ) .- σ_min^2))
    end # if

    return reg_term
end # function

# ==============================================================================

@doc raw"""
        entropy_regularization(outputs::NamedTuple, reg_terms::Vector{Symbol}=[:encoder_logσ])

Compute the entropy regularization term for specified Gaussian-distributed
variables within the `outputs` NamedTuple. The regularization term is based on
the entropy of the Gaussian distributions.

Given a Gaussian with standard deviation σ, its entropy is: H(σ) = 0.5 * n * (1
+ log(2π)) + 0.5 * sum(log(σ²))

Regularizing by this quantity encourages the model to find a balance in the
uncertainty it expresses, preventing it from being either too certain or too
uncertain.

# Arguments
- `outputs::NamedTuple`: A NamedTuple containing VAE outputs.

## Optional Keyword Arguments
- `reg_terms::Vector{Symbol}`: The keys in `outputs` specifying which
  Gaussian-distributed variables' entropy to compute. For a VAE, valid targets
  are `:encoder_logσ`, `:decoder_logσ`, and `:decoder_σ`.

# Returns
- `entropy_reg::Float32`: The computed entropy regularization term for the
  specified targets.

# Notes
- Ensure that all keys in `reg_terms` exist within the `outputs` NamedTuple.
- The entropy is computed in the log-space for `:encoder_logσ` and
  `:decoder_logσ`, and in the original space for `:decoder_σ`.
"""
function entropy_regularization(
    outputs::NamedTuple; reg_terms::Vector{Symbol}=[:encoder_logσ]
)::Float32
    # List possible targets
    targets = [:encoder_logσ, :decoder_logσ, :decoder_σ]

    # Ensyre there is at least one key
    if isempty(reg_terms)
        return 0.0f0
    end # if

    # Check if reg_terms are valid
    if !any(term in targets for term in reg_terms)
        throw(ArgumentError("The specified target is not valid. Valid " *
                            "targets  are $(targets)."))
    end

    # Ensure all keys in reg_terms are in outputs
    if !all(key ∈ keys(outputs) for key in reg_terms)
        throw(ArgumentError("All keys in reg_terms must exist in outputs!"))
    end # if

    # Compute the entropy values directly in the sum
    entropy = sum(
        begin
            if term == :decoder_σ
                # Compute the entropy of the Gaussian distribution
                # H(X) = 0.5 * n * (1 + log(2π)) + 0.5 * sum(log(diagonal_elements))
                0.5f0 * length(outputs[term]) * (1 + log(2π)) +
                0.5f0 * sum(log.(outputs[term]))
            else
                # Compute the entropy of the Gaussian distribution
                # H(X) = 0.5 * n * (1 + log(2π)) + 0.5 * sum(log(diagonal_elements))
                0.5f0 * (length(outputs[term])) * (1 + log(2π)) +
                0.5f0 * sum(outputs[term])
            end # if
        end # begin
        for term in reg_terms
    )

    return entropy
end # function
