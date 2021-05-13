module GenCurveSegmentation

using Random, Distributions
using Gen, GenParticleFilters
using Plots

include("utils.jl")
include("data.jl")
include("render.jl")
include("model.jl")
include("inference.jl")

export render_trace, render_trace!, render_obs, render_obs!, render_traces
export parse_strokes, load_strokes, load_dataset
export curve_model, particle_filter

end
