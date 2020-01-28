"""
    Elements of F(ℳ) consist of a position x and a GL(d, ℝ)-matrix ν that
    represents a basis for Tₓℳ
"""

struct Frame{Tx, Tν}
    x::Tx
    ν::Tν
    function Frame(x::Tx, ν::Tν) where {Tx, Tν <: AbstractArray}
        if rank(ν) != length(x)
            error("A is not of full rank")
        end
        new{Tx, Tν}(x, ν)
    end
end

"""
    Some generic functions for calculations on F(ℳ)
"""

Base.:+(u::Frame{Tx, Tν}, v::Frame{Tx, Tν}) where {Tx, Tν} = Frame(u.x + v.x , u.ν .+ v.ν)
Base.:-(u::Frame{Tx, Tν}, v::Frame{Tx, Tν}) where {Tx, Tν} = Frame(u.x - v.x , u.ν .- v.ν)
Base.:-(u::Frame{Tx, Tν}) where {Tx, Tν} = Frame(-u.x , -u.ν)

Base.:+(u::Frame{Tx, Tν}, y::Tx) where {Tx, Tν} = Frame(u.x + y, u.ν)
Base.zero(u::Frame{Tx, Tν}) where {Tx, Tν} = Frame(zero(u.x), one(u.ν))

# Canonical projection
Π(u::Frame{Tx, Tν}) where {Tx,Tν} = u.x

# The group action of a frame on ℝᵈ
FrameAction(u::Frame{Tx, Tν}, e::T) where {Tx,Tν,T<:AbstractArray} = u.ν*e

# Horizontal lift of the orthogonal projection
Pˣ(u::Frame, ℳ::T) where {T<:EmbeddedManifold} = P(u.x, ℳ)

# Functions for solving SDEs on the frame bundle
include("FrameBundles.jl")

"""
    Now let us create a stochastic process on the frame bundle of the 2-sphere 𝕊²
"""
𝕊 = Sphere(1.0)


struct SphereDiffusion <: FrameBundleProcess
    𝕊::Sphere

    function SphereDiffusion(𝕊::Sphere)
        new(𝕊)
    end
end

Bridge.b(t, u, ℙ::SphereDiffusion) = Frame(zeros(3), zeros(3,3))
Bridge.σ(t, u, ℙ::SphereDiffusion) = Pˣ(u, 𝕊)
Bridge.constdiff(::SphereDiffusion) = false

ℙ = SphereDiffusion(𝕊)

x₀ = [0.,0.,1.]
u₀ = Frame(x₀, [2. 0. 0. ; 0. 1. 0. ; 0. 0. .5])

T = 1.0
dt = 1/1000
τ(T) = (x) -> x*(2-x/T)
tt = τ(T).(0.:dt:T)
W = sample(0:dt:T, Wiener{ℝ{3}}())
U = solve(StratonovichEuler(), u₀, W, ℙ)
X  = SamplePath(tt, Π.(U.yy))

include("Sphereplots.jl")
plotly()
SpherePlot(X, 𝕊)

function SimulatePoints(n, u₀, ℙ::SphereDiffusion)
    out = Frame[]
    it = 0
    while length(out) < n
        W = sample(0.:dt:T, Wiener{ℝ{3}}())
        U = solve(StratonovichEuler(),u₀, W, ℙ)
        push!(out, U.yy[end])
    end
    return out
end

@time ξ = SimulatePoints(1000, u₀, ℙ)

SphereScatterPlot(extractcomp(Π.(ξ),1), extractcomp(Π.(ξ),2), extractcomp(Π.(ξ),3), x₀, 𝕊 )
