abstract type FrameBundleProcess end

mutable struct Increments{S<:Bridge.AbstractPath}
    X::S
end

iterate(dX::Increments, i = 1) = i + 1 > length(dX.X.tt) ? nothing : ((i, dX.X.tt[i], dX.X.tt[i+1]-dX.X.tt[i], dX.X.yy[.., i+1]-dX.X.yy[.., i]), i + 1)

increments(X::Bridge.AbstractPath) = Increments(X)
endpoint(y, P) = y

import Bridge.solve
solve(::StratonovichEuler, u, W::Bridge.SamplePath, P::FrameBundleProcess) = let X = Bridge.samplepath(W.tt, zero(u)); solve!(StratonovichEuler(), X, u, W, P); X end
function solve!(::StratonovichEuler, Y, u::Frame, W::Bridge.SamplePath, ℙ::FrameBundleProcess)
    N = length(W)
    N != length(Y) && error("Y and W differ in length")

    ww = W.yy
    tt = Y.tt
    yy = Y.yy
    tt[:] = W.tt

    y::typeof(u) = u

    for k in 1:N-1
        dt = tt[k+1] - tt[k]
        dw = ww[k+1] - ww[k]
        yy[.., k] = y
        yᴱ = y + Bridge.H(y, ℙ)*dw
        y = y + .5*(Bridge.H(yᴱ, ℙ) + Bridge.H(y, ℙ))*dw
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

    for (k, t, dt, dw) in increments(W)
        yy[.., k] = y
        yᴱ = y + Bridge.H(y, ℙ)*dw
        y = y + .5*(Bridge.H(yᴱ, ℙ) + Bridge.H(y, ℙ))*dw
    end
    yy[.., N] = endpoint(y, P)
    Y
end
