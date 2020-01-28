abstract type FrameBundleProcess end

mutable struct Increments{S<:Bridge.AbstractPath}
    X::S
end

iterate(dX::Increments, i = 1) = i + 1 > length(dX.X.tt) ? nothing : ((i, dX.X.tt[i], dX.X.tt[i+1]-dX.X.tt[i], dX.X.yy[.., i+1]-dX.X.yy[.., i]), i + 1)

increments(X::Bridge.AbstractPath) = Increments(X)
endpoint(y, P) = y

import Bridge.solve
solve(::StratonovichEuler, u, W::Bridge.SamplePath, P::FrameBundleProcess) = let X = Bridge.samplepath(W.tt, zero(u)); solve!(StratonovichEuler(), X, u, W, P); X end
function solve!(::StratonovichEuler, Y, u::Frame, W::Bridge.SamplePath, P::FrameBundleProcess)
    N = length(W)
    N != length(Y) && error("Y and W differ in length")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y::typeof(u) = u

    for i in 1:N-1
        dt = tt[i+1] - tt[i]
        dw = ww[i+1] - ww[i]
        yy[.., i] = y
        yᴱ = y + Bridge.σ(tt[i], y, P)*dw
        y = y + .5*(Bridge.σ(tt[i+1], yᴱ,P) + Bridge.σ(tt[i], y, P))*dw
    end
    yy[..,N] = endpoint(y, P)
    Y
end

# fallback method
function solve!(::StratonovichEuler, Y, u::Frame, W::Bridge.AbstractPath, P::FrameBundleProcess)
    N = length(W)
    N != length(Y) && error("Y and W differ in length.")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[i] = W.tt

    y::typeof(u) = u

    for (i, t, dt, dw) in increments(W)
        yy[.., i] = y
        yᴱ = y + Bridge.σ(tt[i], y, P)*dw
        y = y + .5*(Bridge.σ(tt[i+1], yᴱ,P) + Bridge.σ(tt[i], y, P))*dw
    end
    yy[.., N] = endpoint(y, P)
    Y
end
