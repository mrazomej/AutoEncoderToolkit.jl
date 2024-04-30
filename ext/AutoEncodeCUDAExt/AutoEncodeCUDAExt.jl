module AutoEncodeCUDAExt

# Import CUDA library
import CUDA

# Include utils extension
include("utils.jl")

# Include adjoints extension
include("adjoints.jl")

# Include VAE extension
include("vae.jl")

# Include HVAE extension
include("hvae.jl")

# Include RHVAE extension
include("rhvae.jl")

end # module