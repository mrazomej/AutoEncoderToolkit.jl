using Documenter
using AutoEncode

makedocs(
    sitename="AutoEncode",
    format=Documenter.HTML(),
    # modules=[AutoEncode],
    pages=[
        "Home" => "index.md",
        "Encoders & Decoders" => "encoders.md",
        "Deterministic Autoencoders" => "ae.md",
        "VAE / β-VAE" => "vae.md",
        "MMD-VAE (InfoVAE)" => "mmdvae.md",
        "InfoMax-VAE" => "infomaxvae.md",
        "HVAE" => "hvae.md",
        "RHVAE" => "rhvae.md",
        "Differential Geometry" => "diffgeo.md",
    ],
    remotes=nothing
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/mrazomej/AutoEncode.jl"
)
