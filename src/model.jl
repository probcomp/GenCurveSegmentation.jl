const type_prior = ones(4) / 4
const type_transition =
    [0.1 * ones(2, 2) 0.4 * ones(2, 2);
     0.4 * ones(2, 2) 0.1 * ones(2, 2);
     type_prior']
const directions = [0 -1; 0 1; -1 0; 1 0]

generate_points(start, speed, dir, n) =
    start .+ speed .* (dir * collect(1:n)')

@gen function stroke_step(t::Int, prev_state::Tuple)
    prev_stroke, prev_points = prev_state
    n_points ~ shifted_poisson(10, 1)
    speed ~ gamma(2.0, 0.05)
    disconnect ~ bernoulli(0.1)
    if disconnect
        stroke ~ categorical(type_prior)
        start_x ~ uniform(1.25, 3.75)
        start_y ~ uniform(1.25, 3.75)
        start = [start_x, start_y]
    else
        stroke ~ categorical(type_transition[prev_stroke,:])
        start = prev_points[:, end]
    end
    points = generate_points(start, speed, directions[stroke, :], n_points)
    return (stroke, points)
end

@gen (static) function get_points(state::Tuple)
    return collect(eachcol(state[2]))
end

@gen function observe_point(point::AbstractVector)
    return pt ~ broadcasted_normal(point, [0.05, 0.05])
end

@gen (static) function curve_model(T::Int)
    n_strokes ~ poisson(10)
    init_x ~ uniform(1.25, 3.75)
    init_y ~ uniform(1.25, 3.75)
    states = {:strokes} ~ Unfold(stroke_step)(n_strokes, (5, [init_x, init_y]))
    points_per_stroke ~ Map(get_points)(states)
    points = reduce(vcat, points_per_stroke)
    points_observed = points[1:min(T,length(points))]
    obs ~ Map(observe_point)(points_observed)
    return obs
end

@load_generated_functions()

export curve_model
