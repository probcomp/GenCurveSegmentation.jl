## Rendering and plotting functions ##

function render_trace(trace::Trace; kwargs...)
    render_trace!(trace, plot(size=(500, 500)); kwargs...)
end

function render_trace!(trace::Trace, plt=nothing; show_obs=true, kwargs...)
    plt = isnothing(plt) ? plot!() : plt
    points = Array{Float64}(undef, 2, 0)
    for (i, stroke) in enumerate(trace[:points_per_stroke])
        if trace[:strokes => i => :disconnect]
            plot!(points[1, :], points[2,:]; color=:black, xlims=[0, 5], ylims=[0, 5],
                  legend=false, ratio=:equal, kwargs...)
            points = Array{Float64}(undef, 2, 0)
        end
        points = hcat(points, reduce(hcat, stroke))
    end
    plot!(points[1, :], points[2,:]; color=:black, xlims=[0, 5], ylims=[0, 5],
          legend=false, ratio=:equal, kwargs...)
    if show_obs render_obs!(trace[:obs], plt) end
    return plt
end

function render_obs(obs; kwargs...)
    plt = plot(size=(500, 500), ratio=:equal, legend=false, xlims=[0, 5], ylims=[0, 5])
    render_obs!(obs, plt; kwargs...)
end

function render_obs!(obs, plt=nothing; kwargs...)
    plt = isnothing(plt) ? plot!() : plt
    if obs isa AbstractArray && obs[1] isa AbstractVector obs = reduce(hcat, obs) end
    scatter!(obs[1, :], obs[2,:], color=:black, figsize=(500, 500),
             marker=:dot, msize=3, markerstrokewidth=0, kwargs...)
    return plt
end

function render_traces(traces::AbstractVector{<:Trace}, weights::AbstractVector{<:Real},
                       observations=nothing, kwargs...)
    plt = plot(size=(500, 500))
    for (tr, w) in zip(traces, weights)
       if w < 0.001 continue end
       render_trace!(tr, plt; alpha=w, kwargs...)
    end
    if !isnothing(observations)
        plt = render_obs!(observations, plt)
    end
    return plt
end

function render_cb!(t::Int, state, observations;
                    show_plot=true, anim=nothing,
                    plt_buffer=nothing, plt_interval=0.2)
    traces, weights = get_traces(state), get_norm_weights(state)
    plt = render_traces(traces, weights, observations[:,1:t])
    title!(plt, "t = $t")
    if show_plot display(plt) end
    if !isnothing(anim) frame(anim) end
    if !isnothing(plt_buffer)
        T = size(observations)[2]
        plt_interval = plt_interval < 1 ?
            floor(Int, plt_interval * T) : round(Int, plt_interval)
        if mod(t, plt_interval) == 0 || t == T
            push!(plt_buffer, plt)
        end
    end
end
