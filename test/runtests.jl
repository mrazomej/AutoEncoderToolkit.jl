using AutoEncoderToolkit
using Test

@testset "AutoEncoderToolkit.jl" begin
    # Test utils
    include("utils.jl")

    # Test encoders
    include("encoders.jl")

    # Test decoders
    include("decoders.jl")

    # Test AE module
    include("ae.jl")

    # Test VAE module
    include("vae.jl")

    # Test InfoVAE module
    include("mmdvae.jl")

    # Test InfoMaxVAE module
    include("infomaxvae.jl")

    # Test HVAE module
    include("hvae.jl")

    # Test RHVAE module
    include("rhvae.jl")

    # Test diffgeo module
    include("diffgeo.jl")

    # Test AutoEncoderToolkitCUDAExt module
    include("cuda_ext.jl")
end
