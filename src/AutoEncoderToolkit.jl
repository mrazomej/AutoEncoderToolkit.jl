module AutoEncoderToolkit

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Import packages
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
import Flux
import Random
import Distributions

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Abstract Types
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

@doc raw"""
    AbstractAutoEncoder

This is an abstract type that serves as a parent for all autoencoder models in
this package.

An autoencoder is a type of artificial neural network used for learning
efficient codings of input data. It consists of an encoder, which compresses the
input data, and a decoder, which reconstructs the original data from the
compressed representation.

Subtypes of this abstract type should define specific types of autoencoders,
such as standard Auto Encoders, Variational AutoEncoders (VAEs), etc. (more to
be added).
"""
abstract type AbstractAutoEncoder end

@doc raw"""
    AbstractDeterministicAutoEncoder <: AbstractAutoEncoder

This is an abstract type that serves as a parent for all deterministic
autoencoder models in this package.

A deterministic autoencoder is a type of autoencoder where the encoding and
decoding processes are deterministic functions. This contrasts with stochastic
or variational autoencoders, where the encoding process involves a random
sampling step.
"""
abstract type AbstractDeterministicAutoEncoder <: AbstractAutoEncoder end

@doc raw"""
    AbstractVariationalAutoEncoder <: AbstractAutoEncoder

This is an abstract type that serves as a parent for all variational autoencoder
models in this package.

A variational autoencoder (VAE) is a type of autoencoder that adds a
probabilistic twist to autoencoding. Instead of learning a deterministic
function for the encoding, a VAE learns the parameters of a probability
distribution representing the data. The encoding process then involves sampling
from this distribution.

Subtypes of this abstract type should define specific types of variational
autoencoders, such as Standard VAEs, InfoMaxVAEs, or Hamiltonian VAEs.
"""
abstract type AbstractVariationalAutoEncoder <: AbstractAutoEncoder end

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# General encoder and decoder types
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Import Encoder types
include("encoders.jl")

# Import Decoder types
include("decoders.jl")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Custom layers
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Import custom layers
include("layers.jl")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Include Utils module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module utils
include("utils.jl")
end # submodule

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Include functions defined in adjoints.jl for custom AutoDiff rules
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

include("adjoints.jl")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add AEs module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module AEs
include("ae.jl")
end # submodule

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add VAEs module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module VAEs
include("vae.jl")
end # submodule

# Export VAE structs
using .VAEs: VAE

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add InfoMaxVAEs module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module InfoMaxVAEs
include("infomaxvae.jl")
end # submodule

# Export InfoMaxVAE structure
using .InfoMaxVAEs: InfoMaxVAE

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Add MMDVAEs (alias InfoVAEs) module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module MMDVAEs
include("mmdvae.jl")
end # submodule

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Include module for Hamiltonian Variational Autoencoders (HVAEs)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module HVAEs
include("hvae.jl")
end # submodule

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Include module for Riemannian Hamiltonian Variational Autoencoders (RHVAEs)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module RHVAEs
include("rhvae.jl")
end # submodule

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Define operators over abstract types
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# Import * operator from Base module
import Base: *

# Define * operator for AbstractEncoder and AbstractDecoder types
*(E::AbstractDeterministicEncoder, D::AbstractDeterministicDecoder) = AEs.AE(E, D)
*(E::AbstractVariationalEncoder, D::AbstractVariationalDecoder) = VAEs.VAE(E, D)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Include Differential Geometry Module
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

module diffgeo
include("diffgeo.jl")
end # submodule

end # module