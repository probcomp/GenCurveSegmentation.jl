module GenCurveSegmentation

using Distributions, Gen, GenParticleFilters
using Plots

include("utils.jl")
include("render.jl")
include("model.jl")
include("inference.jl")

export render_trace, render_trace!, render_obs!, render_traces
export curve_model, particle_filter

end
