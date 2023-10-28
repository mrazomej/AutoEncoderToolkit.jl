# Import Abstract Types
using ..AutoEncode: JointEncoder, SimpleDecoder, JointDecoder, SplitDecoder

# ==============================================================================

@doc raw"""
    l2_regularization(vae_outputs::Dict{Symbol, Any}, 
                    reg_flags::Dict{Symbol, Bool})

Compute the L2 regularization term (also known as Ridge regularization) based on
the VAE outputs and the `reg_flags` dictionary.

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
- `vae_outputs::Dict{Symbol, Any}`: Dictionary containing outputs from the VAE
  model.
- `reg_flags::Dict{Symbol, Bool}`: Dictionary that specifies which entries from
  `vae_outputs` to consider for regularization.

# Returns
- `reg_term::Float32`: The computed L2 regularization value.

# Notes
- Ensure that the keys in `reg_flags` are a subset of the keys in `vae_outputs`.
"""
function l2_regularization(
    vae_outputs::Dict{Symbol,Any}, reg_flags::Dict{Symbol,Bool}
)::Float32
    # Ensure all keys in reg_flags are in vae_outputs
    if !all(key in keys(vae_outputs) for key in keys(reg_flags))
        error("All keys in reg_flags must exist in vae_outputs!")
    end

    # Initialize the regularization term to zero
    reg_term = 0.0f0
    # Loop through dictionary keys
    for (key, flag) in reg_flags
        if flag
            # L2 regularization for the given key's value
            reg_term += sum(vae_outputs[key] .^ 2)
        end # if 
    end # for

    return reg_term
end

# ==============================================================================

@doc raw"""
    min_variance_regularization(vae_outputs::Dict{Symbol, Any}, σ_min::Float32)

Compute the minimum variance constraint regularization term based on the VAE
outputs and a specified minimum standard deviation (σ_min).

The regularization is defined as: L = λ ∑ᵢ max(0, log(σᵢ²) - log(σ_min²))

Where:
- λ is the regularization strength (not computed in this function, but typically
  applied outside).
- σᵢ represents the standard deviation for each output in the VAE decoder.

The primary purpose of this regularization is to prevent the variance from going
below a certain threshold, discouraging the model from making overconfident
predictions.

# Arguments
- `vae_outputs::Dict{Symbol, Any}`: Dictionary containing outputs from the VAE
  model.
- `σ_min::Float32`: The minimum allowable standard deviation.

# Returns
- `reg_term::Float32`: The computed minimum variance constraint regularization
  value.
"""
function min_variance_regularization(
    vae_outputs::Dict{Symbol,Any}, σ_min::Float32
)::Float32
    # Extract decoder log variance
    decoder_logσ = vae_outputs[:decoder_logσ]

    # Compute regularization term to discourage extremeley small variances
    reg_term = sum(max.(0.0f0, decoder_logσ .- log(σ_min^2)))

    return reg_term
end # function

# ==============================================================================

@doc raw"""
    entropy_regularization(vae_output::Dict{Symbol, Any}, target::Symbol)

Compute the entropy regularization term for a specified Gaussian-distributed
variable within the `vae_output` dictionary. The regularization term is based on
the entropy of the Gaussian distribution.

Given a Gaussian with standard deviation σ, its entropy is: H(σ) = 0.5 *
log(2πeσ²)

Regularizing by this quantity encourages the model to find a balance in the
uncertainty it expresses, preventing it from being either too certain or too
uncertain.

# Arguments
- `vae_output::Dict{Symbol, Any}`: A dictionary containing VAE outputs.
- `target::Symbol`: The key in `vae_output` specifying which
  Gaussian-distributed variable's entropy to compute. For a VAE, valid targets
  are `:encoder_logσ` or `:decoder_logσ`.

# Returns
- `entropy_reg::Float32`: The computed entropy regularization term for the
  specified target.

# Notes
- Ensure that the target exists within the `vae_output` dictionary and is a
  valid target.
"""
function entropy_regularization(
    vae_output::Dict{Symbol,Any}, target::Symbol
)::Float32
    # Check if the target is valid
    if !(target in [:encoder_logσ, :decoder_logσ])
        throw(ArgumentError("The specified target is not valid. Valid " *
                            "targets  are :encoder_logσ and :decoder_logσ."))
    end

    # Check if the target exists in the vae_output
    if !haskey(vae_output, target)
        throw(
            ArgumentError("The specified target does not exist in vae_output.")
        )
    end

    # Extract the log variance from the vae_output
    logσ = vae_output[target]

    # Compute the entropy of the Gaussian distribution
    entropy = 0.5f0 * (log(2π) + 1 + 2 * logσ)

    return entropy
end # function
