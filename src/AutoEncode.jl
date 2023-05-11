module AutoEncode

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Abstract Types
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Define main abstract type that captures all autoencoders
abstract type AbstractAutoEncoder end

# Define abstract type that captures deterministic autoencoders
abstract type AbstractDeterministicAutoEncoder <: AbstractAutoEncoder end

# Define abstract type that captures stochastic autoencoders
abstract type AbstractVariationalAutoEncoder <: AbstractAutoEncoder end

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add AEs module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module AEs
include("ae.jl")
end # submodule

# Export AE structure
using .AEs: AE, SimpleAE

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add IRMAEs module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module IRMAEs
include("irmae.jl")
end # submodule

# Export AE structure
using .IRMAEs: IRMAE, SimpleIRMAE


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Evaluation functions
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

include("eval.jl")

end