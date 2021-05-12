using Distributions: truncated

## Extra distributions ##

@dist shifted_poisson(rate, offset::Int) = poisson(rate) + offset
@dist delta(val) = uniform_discrete(val, val)

struct TruncatedNormal <: Gen.Distribution{Float64} end

"""
    trunc_normal(mu::Real, std::Real, lb::Real, ub::Real)
Samples a `Float64` value from a normal distribution.
"""
const trunc_normal = TruncatedNormal()

(d::TruncatedNormal)(mu, std, lb, ub) = Gen.random(d, mu, std, lb, ub)

Gen.random(::TruncatedNormal, mu::Real, std::Real, lb::Real, ub::Real) =
    rand(truncated(Distributions.Normal(mu, std), lb, ub))

function Gen.logpdf(::TruncatedNormal, x::Real, mu::Real, std::Real, lb::Real, ub::Real)
    d = truncated(Distributions.Normal(mu, std), lb, ub)
    untrunc_lpdf = Distributions.logpdf(d.untruncated, x)
    if d.tp > 0
        untrunc_lpdf - d.logtp
    elseif cdf(d.untruncated, lb) ≈ 1.0
        untrunc_lpdf - Distributions.logccdf(d.untruncated, lb)
    elseif cdf(d.untruncated, ub) ≈ 0.0
        untrunc_lpdf - Distributions.logcdf(d.untruncated, ub)
    end
end

Gen.is_discrete(::TruncatedNormal) = false

## Other utility functions ##

function segment_find(idx, seglens)
    count = 0
    for (i, l) in enumerate(seglens)
        count += l
        if count >= idx return (i, idx-count+l) end
    end
    return nothing
end
