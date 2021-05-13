function parse_strokes(str::AbstractString)
    lines = filter(x -> x[1] != '#', split(str))
    data = map(x -> parse.(Float64, split(x, ",")), lines)
    return reduce(hcat, data)
end

function load_strokes(fn::AbstractString; transform=false, jitter=false)
    data = read(fn, String) |> parse_strokes
    transform && transform_strokes!(data)
    jitter && add_jitter!(data)
    return data
end

function transform_strokes!(data::AbstractMatrix;
                            invert_x::Bool=false, invert_y::Bool=true,
                            in_scale::Real=176.389, out_scale::Real=5.0)
    data .*= (out_scale / in_scale)
    if invert_x data[1,:] .*= -1 end
    if invert_y data[2,:] .*= -1 end
    min_bounds, max_bounds = minimum(data, dims=2), maximum(data, dims=2)
    mid_x, mid_y = 0.5*(min_bounds[1] + max_bounds[1]), 0.5*(min_bounds[2] + max_bounds[2])
    data[1,:] .+= (out_scale / 2 - mid_x)
    data[2,:] .+= (out_scale / 2 - mid_y)
    return data
end

function add_jitter!(data::AbstractMatrix; x_std::Real=0.05, y_std::Real=0.05)
    data[1,:] .+= (randn(size(data)[2]) .* x_std)
    data[2,:] .+= (randn(size(data)[2]) .* y_std)
    return data
end

function load_dataset(dir::AbstractString; transform=true, jitter=false)
    fns = readdir(dir)
    fns = filter(s -> match(r".*\.txt", s) !== nothing, fns)
    dataset = load_strokes.(joinpath.(dir, fns))
    transform && transform_strokes!.(dataset)
    jitter && add_jitter!.(dataset)
    return dataset
end
