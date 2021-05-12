## SMC initial proposal ##

@gen function init_proposal(obs)
    n_strokes ~ shifted_poisson(0.25, 1)
    init_x ~ trunc_normal(obs[1], 0.05, 1.25, 3.75)
    init_y ~ trunc_normal(obs[2], 0.05, 1.25, 3.75)
end

## SMC update kernels ##

@gen function fwd_stroke_proposal(trace, t::Int, obs)
    # Resample a larger number of strokes
    n_strokes ~ shifted_poisson(0.25, trace[:n_strokes]+1)
    # Disconnect stroke if new observation is far from previous one
    prev_obs = trace[:obs][t-1]
    p_disconnect = 1 - exp(-sum(((obs .- prev_obs)./1.0).^2))
    disconnect ~ bernoulli(p_disconnect)
    # Sample new stroke location based on observation
    if disconnect
        start_x ~ trunc_normal(obs[1], 0.05, 1.25, 3.75)
        start_y ~ trunc_normal(obs[2], 0.05, 1.25, 3.75)
    end
end

@gen function bwd_stroke_proposal(trace, t::Int, obs)
    n_strokes ~ poisson(t/10 + 1)
end

@transform stroke_transform (p_old, q_fwd) to (p_new, q_bwd) begin
    n_new = @read(q_fwd[:n_strokes], :discrete)
    n_old = @read(p_old[:n_strokes], :discrete)
    @write(p_new[:n_strokes], n_new, :discrete)
    @write(q_bwd[:n_strokes], n_old, :discrete)
    @copy(q_fwd[:disconnect], p_new[:strokes => n_old + 1 => :disconnect])
    if @read(q_fwd[:disconnect], :discrete)
        @copy(q_fwd[:start_x], p_new[:strokes => n_old + 1 => :start_x])
        @copy(q_fwd[:start_y], p_new[:strokes => n_old + 1 => :start_y])
    end
end

@gen function fwd_points_proposal(trace, t::Int, obs)
    n_points = sum(length.(trace[:points_per_stroke]))
    n_stroke_points_prev = trace[:strokes => trace[:n_strokes] => :n_points]
    n_stroke_points ~ shifted_poisson(2, (t - n_points) + n_stroke_points_prev)
end

@gen function bwd_points_proposal(trace, t::Int, obs)
    n_stroke_points ~ poisson(10)
end

@transform points_transform (p_old, q_fwd) to (p_new, q_bwd) begin
    n_strokes = @read(p_old[:n_strokes], :discrete)
    @copy(q_fwd[:n_stroke_points], p_new[:strokes => n_strokes => :n_points])
    @copy(p_old[:strokes => n_strokes => :n_points], q_bwd[:n_stroke_points])
end

@gen function split_proposal(trace, t::Int, obs, i_stroke::Int, offset::Int)
    # Sample new stroke parameters
    prev_stroke = trace[:strokes => i_stroke => :stroke]
    speed ~ gamma(2.0, 0.05)
    # Disconnect stroke if new observation is far from previous one
    prev_obs = trace[:obs][t-1]
    p_disconnect = 1 - exp(-sum(((obs .- prev_obs)./1.0).^2))
    disconnect ~ bernoulli(p_disconnect)
    # Sample new stroke location based on observation
    if disconnect
        stroke ~ categorical(type_prior)
        start_x ~ trunc_normal(obs[1], 0.05, 1.25, 3.75)
        start_y ~ trunc_normal(obs[2], 0.05, 1.25, 3.75)
    else
        stroke ~ categorical(type_transition[prev_stroke,:])
    end
    return (i_stroke, offset)
end

@transform split_transform (p_old, q_fwd) to (p_new, q_bwd) begin
    i_stroke, offset = @read(q_fwd[], :discrete)
    # Increment number of strokes
    n_strokes = @read(p_old[:n_strokes], :discrete)
    @write(p_new[:n_strokes], n_strokes + 1, :discrete)
    # Split points between old and new stroke
    n_points = @read(p_old[:strokes => i_stroke => :n_points], :discrete)
    points_1, points_2 = offset - 1, n_points - offset +1
    @write(p_new[:strokes => i_stroke => :n_points], points_1, :discrete)
    @write(p_new[:strokes => i_stroke + 1 => :n_points], points_2, :discrete)
    # Copy over sampled parameters for new stroke
    @copy(q_fwd[:stroke], p_new[:strokes => i_stroke + 1 => :stroke])
    @copy(q_fwd[:speed], p_new[:strokes => i_stroke + 1 => :speed])
    @copy(q_fwd[:disconnect], p_new[:strokes => i_stroke + 1 => :disconnect])
    if @read(q_fwd[:disconnect], :discrete)
        @copy(q_fwd[:start_x], p_new[:strokes => i_stroke + 1 => :start_x])
        @copy(q_fwd[:start_y], p_new[:strokes => i_stroke + 1 => :start_y])
    end
    # Shift indices forward for all subsequent strokes
    for j_stroke in i_stroke+1:n_strokes
        @copy(p_old[:strokes => j_stroke], p_new[:strokes => j_stroke + 1])
    end
end

@gen function merge_proposal(trace, t::Int, obs, i_stroke::Int)
    # Do nothing (no additional randomness needed)
    return i_stroke
end

@transform merge_transform (p_old, q_fwd) to (p_new, q_bwd) begin
    i_stroke = @read(q_fwd[], :discrete)
    # Decrement number of strokes
    n_strokes = @read(p_old[:n_strokes], :discrete)
    @write(p_new[:n_strokes], n_strokes - 1, :discrete)
    # Merge points from current stroke to previous stroke
    points_1 = @read(p_old[:strokes => i_stroke - 1 => :n_points], :discrete)
    points_2 = @read(p_old[:strokes => i_stroke => :n_points], :discrete)
    @write(p_new[:strokes => i_stroke - 1 => :n_points], points_1 + points_2, :discrete)
    # Copy deleted parameters to backward proposal
    @copy(p_old[:strokes => i_stroke => :stroke], q_bwd[:stroke])
    @copy(p_old[:strokes => i_stroke => :speed], q_bwd[:speed])
    @copy(p_old[:strokes => i_stroke => :disconnect], q_bwd[:disconnect])
    if @read(p_old[:strokes => i_stroke => :disconnect], :discrete)
        @copy(p_old[:strokes => i_stroke => :start_x], q_bwd[:start_x])
        @copy(p_old[:strokes => i_stroke => :start_y], q_bwd[:start_y])
    end
    # Shift indices backward for all subsequent strokes
    for j_stroke in i_stroke+1:n_strokes
        @copy(p_old[:strokes => j_stroke], p_new[:strokes => j_stroke - 1])
    end
end

@gen function fwd_proposal(trace, t::Int, obs)
    # Find total points, current stroke, and offset from stroke start
    stroke_lengths = length.(trace[:points_per_stroke])
    n_points = sum(stroke_lengths)
    i_stroke, offset = t > n_points ? (nothing, nothing) : segment_find(t, stroke_lengths)
    # Guess if new observation is part of the same stroke as previous one
    prev_obs = t == 1 ? obs : trace[:obs][t-1]
    p_connected = exp(-sum(((obs .- prev_obs)./1.0).^2))
    # Decide between sub-proposals
    if t > n_points # Extend with new stroke or points
        p_new_stroke, p_new_points = 1-p_connected, p_connected
        p_split, p_merge, p_default = 0.0, 0.0, 0.0
    elseif offset > 1 # Potentially split current stroke
        p_new_stroke, p_new_points, p_merge = 0.0, 0.0, 0.0
        p_split, p_default = 1-p_connected, p_connected
    elseif i_stroke > 1 # Potentially merge current and previous stroke
        p_new_stroke, p_new_points, p_split = 0.0, 0.0, 0.0
        p_merge, p_default = p_connected/2, 1-p_connected/2
    else # Use default proposal
        p_new_stroke, p_new_points = 0.0, 0.0
        p_split, p_merge, p_default = 0.0, 0.0, 1.0
    end
    # Sample a branch and call sub-proposal
    branch ~ categorical([p_new_stroke, p_new_points, p_split, p_merge, p_default])
    if branch == 1 # Extend with new stroke
        {*} ~ fwd_stroke_proposal(trace, t, obs)
    elseif branch == 2 # Extend with new points
        {*} ~ fwd_points_proposal(trace, t, obs)
    elseif branch == 3 # Split current stroke
        {*} ~ split_proposal(trace, t, obs, i_stroke, offset)
    elseif branch == 4 # Merge current and previous stroke
        {*} ~ merge_proposal(trace, t, obs, i_stroke)
    elseif branch == 5 # Use default proposal
        nothing
    end
end

@gen function bwd_proposal(trace, t::Int, obs)
    # Find current stroke, and offset from stroke start
    stroke_lengths = length.(trace[:points_per_stroke])
    n_points = sum(stroke_lengths)
    i_stroke, offset = t > n_points ? (nothing, nothing) : segment_find(t, stroke_lengths)
    # Sample a branch and call sub-proposal
    branch ~ categorical([0.1, 0.1, 0.1, 0.1, 0.6])
    if branch == 1 # Reverse stroke extension
        {*} ~ bwd_stroke_proposal(trace, t, obs)
    elseif branch == 2 # Reverse points extension
        {*} ~ bwd_points_proposal(trace, t, obs)
    elseif branch == 3 # Reverse split proposal
        {*} ~ merge_proposal(trace, t, obs, i_stroke)
    elseif branch == 4 # Reverse merge proposal
        {*} ~ split_proposal(trace, t, obs, i_stroke, offset)
    elseif branch == 5 # Do nothing for default proposal
        nothing
    end
end

@transform update_transform (p_old, q_fwd) to (p_new, q_bwd) begin
    branch = @read(q_fwd[:branch], :discrete)
    @write(q_bwd[:branch], branch, :discrete)
    if branch == 1
        @tcall(stroke_transform())
    elseif branch == 2
        @tcall(points_transform())
    elseif branch == 3
        @tcall(split_transform())
    elseif branch == 4
        @tcall(merge_transform())
    end
end

## SMC rejuvenation kernels ##

@gen function rejuv_speed(trace, t_prev::Int=0)
    i_stroke = trace[:n_strokes] - t_prev
    if i_stroke <= 0 return end
    prev_speed = trace[:strokes => i_stroke => :speed]
    {:strokes => i_stroke => :speed} ~ normal(prev_speed, 0.1)
end

@gen function rejuv_smart(trace, t_prev::Int=0)
    for t in t_prev:-1:0
        i_stroke = trace[:n_strokes] - t
        if i_stroke <= 0 continue end
        # Compute observation indices for stroke
        n_obs = get_args(trace)[1]
        n_points = trace[:strokes => i_stroke => :n_points]
        t_start = i_stroke == 1 ?
            1 : sum(length.(trace[:points_per_stroke][1:i_stroke-1])) + 1
        t_stop = min(t_start + n_points - 1, n_obs)
        if t_stop <= t_start return end
        # Estimate speed from first and last point
        start = trace[:obs][t_start]
        stop = trace[:obs][t_stop]
        est_speed = maximum(abs.(start - stop)) / (t_start - t_stop)
        {:strokes => i_stroke => :speed} ~ normal(est_speed, 0.1)
        # Estimate direction from first and last point
        est_dir = argmax(directions * (stop - start))
        dir_probs = ones(4) * 0.05
        dir_probs[est_dir] = 0.85
        {:strokes => i_stroke => :stroke} ~ categorical(dir_probs)
    end
end

@gen function rejuv_dir(trace)
    n_strokes = trace[:n_strokes]
    if trace[:strokes => n_strokes => :disconnect]
        {:strokes => n_strokes => :stroke} ~ categorical(type_prior)
    else
        prev_stroke = n_strokes == 1 ? 5 : trace[:strokes => n_strokes-1 => :stroke]
        {:strokes => n_strokes => :stroke} ~ categorical(type_transition[prev_stroke,:])
    end
end

@gen function rejuv_init(trace)
    init_x ~ normal(trace[:obs][1][1], 0.25)
    init_y ~ normal(trace[:obs][1][2], 0.25)
end

@gen function rejuv_connect(trace)
    T = get_args(trace)[1]
    i_stroke = trace[:n_strokes]
    points_before_stroke = i_stroke == 1 ?
        0 : sum(length.(trace[:points_per_stroke][1:i_stroke-1]))
    if i_stroke == 1 || T <= points_before_stroke return end
    new_start = trace[:obs][points_before_stroke + 1]
    prev_stop = trace[:obs][points_before_stroke]
    p_disconnect = 1 - exp(-sum(((new_start .- prev_stop)./1.0).^2))
    disconnect = {:strokes => i_stroke => :disconnect} ~ bernoulli(p_disconnect)
    # Sample new stroke location based on observation
    if disconnect
        {:strokes => i_stroke => :start_x} ~ trunc_normal(obs[1], 0.05, 1.25, 3.75)
        {:strokes => i_stroke => :start_y} ~ trunc_normal(obs[2], 0.05, 1.25, 3.75)
    end
end

@gen function rejuv_start(trace, t_prev::Int=0)
    i_stroke = trace[:n_strokes] - t_prev
    if i_stroke > 0 && trace[:strokes => i_stroke => :disconnect]
        {:strokes => i_stroke => :start_x} ~
            trunc_normal(trace[:strokes => i_stroke => :start_x], 0.1, 1.25, 3.75)
        {:strokes => i_stroke => :start_y} ~
            trunc_normal(trace[:strokes => i_stroke => :start_y], 0.1, 1.25, 3.75)
    end
end

## Top-level particle filter ##

function particle_filter(observations, n_particles, ess_thresh=0.5;
                         show_plot::Bool=true, anim=nothing)
    # Initialize particle filter with first observation
    n_obs = length(observations)
    obs_choices = [choicemap((:obs => t => :pt, observations[t])) for t=1:n_obs]
    state = pf_initialize(curve_model, (0,), choicemap(),
                          init_proposal, (observations[1],), n_particles)
    # Iterate across timesteps
    for t=1:n_obs
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < ess_thresh * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual, priority_fn=x->x/2)
            if 2 <= t <= 10 pf_rejuvenate!(state, mh, (rejuv_init, ())) end
            # pf_rejuvenate!(state, mh, (rejuv_connect, ()))
            for k in 0:2
                pf_rejuvenate!(state, mh, (rejuv_smart, (k,)))
                pf_rejuvenate!(state, mh, (rejuv_start, (k,)))
            end
        end
        pf_rejuvenate!(state, mh, (rejuv_dir, ()), 2)
        pf_rejuvenate!(state, mh, (rejuv_speed, (0,)))
        pf_rejuvenate!(state, mh, (rejuv_speed, (1,)))
        pf_rejuvenate!(state, mh, (rejuv_speed, (2,)))
        # Smart update that proposes new strokes or new points
        obs = observations[t]
        pf_update!(state, (t,), (UnknownChange(),), obs_choices[t],
                   fwd_proposal, (t, obs), bwd_proposal, (t, obs), update_transform)
        # Render trace
        traces, weights = get_traces(state), get_norm_weights(state)
        plt = render_traces(traces, weights, observations[1:t])
        if show_plot display(plt) end
        if !isnothing(anim) frame(anim) end
    end
    return state
end

export particle_filter
