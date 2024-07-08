module AutoEncoderToolkitCUDAExt

# Import CUDA library
using CUDA
# Import AutoEncode
using AutoEncoderToolkit

# Include utils extension
include("utils.jl")

# Include adjoints extension
include("adjoints.jl")

# Include VAE extension
include("vae.jl")

# Include MMDVAE extension
include("mmdvae.jl")

# Include HVAE extension
include("hvae.jl")

# Include RHVAE extension
include("rhvae.jl")

end # module