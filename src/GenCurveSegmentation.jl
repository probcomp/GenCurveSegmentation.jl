module GenCurveSegmentation

using Random, Distributions
using Gen, GenParticleFilters
using Plots, StringDistances

include("utils.jl")
include("data.jl")
include("analysis.jl")
include("render.jl")
include("model.jl")
include("inference.jl")

export render_trace, render_trace!, render_obs, render_obs!, render_traces, render_cb!
export stroke_rmse, stroke_edit_dist, log_ml
export parse_strokes, load_strokes, load_dataset
export curve_model, particle_filter

end
