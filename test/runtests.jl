using AutoEncode
using Test

@testset "AutoEncode.jl" begin
    # Test AE module
    include("ae.jl")

    # Test VAE module
    include("vae.jl")

    # Test InfoMaxVAE module
    include("infomaxvae.jl")

end
