@doc raw"""
    eval(ae, input)

Function to evaluate an input on an `AE` autoencoder
"""
function eval(ae::AE, input)
    return Flux.Chain(ae.encoder..., ae.decoder...)(input)
end # function

@doc raw"""
    eval(irmae, input)

Function to evaluate an input on an `IRMAE` autoencoder
"""
function eval(irmae::IRMAE, input)
    return Flux.Chain(
        irmae.encoder..., irmae.linear..., irmae.decoder...
    )(input)
end # function