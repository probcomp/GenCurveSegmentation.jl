function trace_to_seq(trace::Trace, incl_disconnect=true)
    sequence = map(enumerate(trace[:strokes])) do (i, (dir, x))
        s = ["D", "U", "L", "R"][dir]
        if incl_disconnect && trace[:strokes => i => :disconnect]
            s = "#" * s
        end
        return s
    end
    return join(sequence)
end

function stroke_edit_dist(trace::Trace, sequence::AbstractString, incl_disconnect=true)
    if !incl_disconnect sequence = filter(c -> c != '#', sequence) end
    return evaluate(Levenshtein(), trace_to_seq(trace, incl_disconnect), sequence)
end

function stroke_edit_dist(state::Gen.ParticleFilterState, sequence::AbstractString,
                          incl_disconnect=true)
    dists = (stroke_edit_dist(tr, sequence, incl_disconnect) for tr in get_traces(state))
    return sum(get_norm_weights(state) .* dists)
end

function stroke_rmse(trace::Trace, targets::AbstractMatrix)
    points = reduce(hcat, last.(trace[:strokes]))
    T = min(size(points)[2], size(targets)[2])
    mse = sum((points[:,1:T] - targets[:,1:T]).^2) ./ T
    return mse ^ 0.5
end

function stroke_rmse(state::Gen.ParticleFilterState, targets::AbstractMatrix)
    rmses = (stroke_rmse(tr, targets) for tr in get_traces(state))
    return sum(get_norm_weights(state) .* rmses)
end

function log_ml(state::Gen.ParticleFilterState)
    log_lls = (project(tr, select(:obs)) for tr in get_traces(state))
    return sum(get_norm_weights(state) .* log_lls)
end
