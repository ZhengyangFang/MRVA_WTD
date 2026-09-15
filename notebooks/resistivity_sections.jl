# Shared AEM resistivity-section renderer for Figures 3 and S19.

using CairoMakie
using DelimitedFiles
import GLMakie
using Printf
using Statistics
using TiffImages

const DRAW_DIR = @__DIR__
const REPO_ROOT = dirname(DRAW_DIR)
const DEFAULT_OUTPUT_DIR = joinpath(REPO_ROOT, "outputs", "figures", "Fig3")
const TIF_PATH = joinpath(REPO_ROOT, "data", "1 resistivity", "aem_log10res_1km_masked.tif")
const DEPTH_PATH = joinpath(REPO_ROOT, "data", "1 resistivity", "aem_depth_levels_m.json")
const LOOKUP_PATH = joinpath(REPO_ROOT, "data", "2 well WTD", "active_cell_lookup.csv")
const IDOMAIN_PATH = joinpath(REPO_ROOT, "data", "1 resistivity", "idomain_1km_target.dat")
const TOPO_PATH = joinpath(REPO_ROOT, "data", "5 topography", "topo_1km.csv")

const CELL_SIZE_M = 1000.0
const X_ORIGIN_M = 330955.0
const Y_MIN_CENTER_M = 938785.0
const NODATA = 9999.0f0
const RESISTIVITY_COLORMAP = Reverse(:RdYlBu)
const TERRAIN_COLORMAP = [
    RGBAf(0.72, 0.70, 0.61, 1.0),
    RGBAf(0.82, 0.78, 0.66, 1.0),
    RGBAf(0.90, 0.84, 0.69, 1.0),
    RGBAf(0.96, 0.90, 0.75, 1.0),
    RGBAf(0.84, 0.82, 0.76, 1.0),
    RGBAf(0.96, 0.95, 0.90, 1.0),
]
const TERRAIN_HILLSHADE_EXAGGERATION = 34.0
const TERRAIN_HILLSHADE_CONTRAST = 1.32
const TERRAIN_DIFFUSE = 0.88
const TERRAIN_SPECULAR = 0.16
const TERRAIN_SHININESS = 28.0
const FIG_FONT_REGULAR = "Arial"
const FIG_FONT_BOLD = "Arial Bold"
const PNG_PX_PER_UNIT = 9.375
const ORTHO_PNG_PX_PER_UNIT = 9.0
const MAX_PLOT_DEPTH_M = 200.0
const ORTHO_HORIZONTAL_DEPTH_M = 100.0
const ORTHO_HORIZONTAL_DEPTH_M_PANEL3 = 120.0
const ORTHO_Z_LIMITS_M = (-200.0, 50.0)
const ORTHO_COLORBAR_MAX = 3.0
const ORTHO_COLORBAR_TICKS = 0.5:0.5:ORTHO_COLORBAR_MAX
const ORTHO_COLORBAR_TICKLABELSIZE = 20
const SLICE_OUTLINE_LINEWIDTH = 1.2
const RESISTIVITY_LOWCLIP_COLOR = first(Makie.to_colormap(RESISTIVITY_COLORMAP))
const RESISTIVITY_HIGHCLIP_COLOR = last(Makie.to_colormap(RESISTIVITY_COLORMAP))

struct TargetPoint
    grid_id::Int
    lonlat::String
    row::Int
    col::Int
    label::String
    response_class::String
end

TargetPoint(grid_id::Int, lonlat::String, row::Int, col::Int) =
    TargetPoint(grid_id, lonlat, row, col, "", "")

const TARGETS = [
    # Fast, Slow, and Buffered representative sites.
    TargetPoint(45679, "-90.772633, 34.519433", 347, 144),
    TargetPoint(83836, "-89.356028, 36.779472", 608, 256),
    TargetPoint(84953, "-89.552750, 36.888222", 619, 237),
]

struct LocalCube
    target::TargetPoint
    rows::Vector{Int}
    cols::Vector{Int}
    x_offsets_km::Vector{Float64}
    y_offsets_km::Vector{Float64}
    x_coords_km::Vector{Float64}
    y_coords_km::Vector{Float64}
    depths_m::Vector{Float64}
    values::Array{Float32, 3}
    terrain_elevation_m::Matrix{Float64}
    surface_elevation_m::Float64
    target_row_index::Int
    target_col_index::Int
end

function usage()
    return """
    Usage:
      julia --project=. resistivity_sections.jl [--output DIR] [--half-window-km N]
                                                   [--view-azimuth-pi A]
                                                   [--view-elevation-pi E]
                                                   [--targets-csv PATH] [--target-set ID]
                                                   [--output-prefix NAME] [--sections-only]
                                                   [--png-only] [--color-range MIN,MAX]
                                                   [--tif PATH] [--depths PATH]
                                                   [--lookup PATH] [--idomain PATH]
                                                   [--topo PATH]

    Outputs:
      Fig3_resistivity_sections_2D.png/pdf
      Fig3_resistivity_cutaway_3D.png

    The main figure shows local near-surface maps plus E-W and S-N
    resistivity sections around the three configured MRVA grid ids.
    Depths deeper than 200 m are omitted by default.
    View angles are in multiples of pi. Use one value for all panels
    or three comma-separated values for panels 1-3.
    """
end

function parse_panel_view_values(text::String, option_name::String)
    parts = [strip(part) for part in split(text, ",") if !isempty(strip(part))]
    length(parts) in (1, 3) || error("$option_name requires one value or three comma-separated values")
    return [parse(Float64, part) for part in parts]
end

function parse_color_range(text::String)
    parts = [strip(part) for part in split(text, ",") if !isempty(strip(part))]
    length(parts) == 2 || error("--color-range requires two comma-separated values")
    limits = (parse(Float64, parts[1]), parse(Float64, parts[2]))
    limits[1] < limits[2] || error("--color-range minimum must be smaller than maximum")
    return limits
end

function panel_view_angle(values, panel_index::Integer, default_angle::Float64)
    values === nothing && return default_angle
    value = length(values) == 1 ? values[1] : values[panel_index]
    return value * pi
end

function parse_args(args)
    output_dir = DEFAULT_OUTPUT_DIR
    output_prefix = "Fig3"
    half_window_km = 24
    tif_path = TIF_PATH
    depth_path = DEPTH_PATH
    lookup_path = LOOKUP_PATH
    idomain_path = IDOMAIN_PATH
    topo_path = TOPO_PATH
    view_azimuth_pi = nothing
    view_elevation_pi = nothing
    targets_path = nothing
    target_set = nothing
    sections_only = false
    png_only = false
    fixed_colorrange = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--output"
            i += 1
            i <= length(args) || error("--output requires a value")
            output_dir = args[i]
        elseif arg == "--output-prefix"
            i += 1
            i <= length(args) || error("--output-prefix requires a value")
            output_prefix = args[i]
            isempty(strip(output_prefix)) && error("--output-prefix cannot be empty")
        elseif arg == "--targets-csv"
            i += 1
            i <= length(args) || error("--targets-csv requires a value")
            targets_path = args[i]
        elseif arg == "--target-set"
            i += 1
            i <= length(args) || error("--target-set requires a value")
            target_set = args[i]
        elseif arg == "--sections-only"
            sections_only = true
        elseif arg == "--png-only"
            png_only = true
        elseif arg == "--color-range"
            i += 1
            i <= length(args) || error("--color-range requires a value")
            fixed_colorrange = parse_color_range(args[i])
        elseif arg == "--half-window-km"
            i += 1
            i <= length(args) || error("--half-window-km requires a value")
            half_window_km = parse(Int, args[i])
            half_window_km > 0 || error("--half-window-km must be positive")
        elseif arg == "--tif"
            i += 1
            i <= length(args) || error("--tif requires a value")
            tif_path = args[i]
        elseif arg == "--depths"
            i += 1
            i <= length(args) || error("--depths requires a value")
            depth_path = args[i]
        elseif arg == "--lookup"
            i += 1
            i <= length(args) || error("--lookup requires a value")
            lookup_path = args[i]
        elseif arg == "--idomain"
            i += 1
            i <= length(args) || error("--idomain requires a value")
            idomain_path = args[i]
        elseif arg == "--topo"
            i += 1
            i <= length(args) || error("--topo requires a value")
            topo_path = args[i]
        elseif arg == "--view-azimuth-pi"
            i += 1
            i <= length(args) || error("--view-azimuth-pi requires a value")
            view_azimuth_pi = parse_panel_view_values(args[i], "--view-azimuth-pi")
        elseif arg == "--view-elevation-pi"
            i += 1
            i <= length(args) || error("--view-elevation-pi requires a value")
            view_elevation_pi = parse_panel_view_values(args[i], "--view-elevation-pi")
        elseif arg == "--help" || arg == "-h"
            println(usage())
            exit(0)
        else
            error("Unknown argument: $arg\n$(usage())")
        end
        i += 1
    end

    return (;
        output_dir,
        output_prefix,
        half_window_km,
        tif_path,
        depth_path,
        lookup_path,
        idomain_path,
        topo_path,
        view_azimuth_pi,
        view_elevation_pi,
        targets_path,
        target_set,
        sections_only,
        png_only,
        fixed_colorrange,
    )
end

function read_depths(path::String)
    isfile(path) || error("Missing depth file: $path")
    text = read(path, String)
    values = [parse(Float64, m.match) for m in eachmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)]
    isempty(values) && error("No depth levels parsed from $path")
    return values
end

function split_csv_line(line::AbstractString)
    return split(chomp(line), ',')
end

function read_targets(path::String; target_set=nothing)
    isfile(path) || error("Missing targets CSV: $path")
    targets = TargetPoint[]
    open(path, "r") do io
        header = strip.(split_csv_line(readline(io)))
        index = Dict(name => findfirst(==(name), header) for name in header)
        required = ["grid_id", "row", "col", "lon", "lat"]
        missing = [name for name in required if !haskey(index, name) || index[name] === nothing]
        isempty(missing) || error("Targets CSV is missing columns: $(join(missing, ", "))")
        if target_set !== nothing && (!haskey(index, "set_id") || index["set_id"] === nothing)
            error("--target-set requires a set_id column in the targets CSV")
        end

        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = strip.(split_csv_line(line))
            if target_set !== nothing && parts[index["set_id"]] != target_set
                continue
            end
            lon = parse(Float64, parts[index["lon"]])
            lat = parse(Float64, parts[index["lat"]])
            label = haskey(index, "plot_label") && index["plot_label"] !== nothing ? parts[index["plot_label"]] : ""
            response_class = haskey(index, "response_class") && index["response_class"] !== nothing ? parts[index["response_class"]] : ""
            push!(
                targets,
                TargetPoint(
                    parse(Int, parts[index["grid_id"]]),
                    @sprintf("%.6f, %.6f", lon, lat),
                    parse(Int, parts[index["row"]]),
                    parse(Int, parts[index["col"]]),
                    label,
                    response_class,
                ),
            )
        end
    end
    isempty(targets) && error("No targets selected from $path")
    length(targets) == 3 || error("This figure layout requires exactly three targets; selected $(length(targets))")
    return targets
end

function read_lookup(path::String)
    isfile(path) || error("Missing lookup CSV: $path")
    rows = Dict{Int, NamedTuple{(:row, :col, :cell_id, :x, :y), Tuple{Int, Int, Int, Float64, Float64}}}()
    open(path, "r") do io
        header = split_csv_line(readline(io))
        expected = ["node_id", "row", "col", "cell_id", "x_center", "y_center"]
        header == expected || error("Unexpected lookup header: $(join(header, ','))")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split_csv_line(line)
            node_id = parse(Int, parts[1])
            rows[node_id] = (
                row=parse(Int, parts[2]),
                col=parse(Int, parts[3]),
                cell_id=parse(Int, parts[4]),
                x=parse(Float64, parts[5]),
                y=parse(Float64, parts[6]),
            )
        end
    end
    return rows
end

function read_surface_elevations(path::String)
    isfile(path) || error("Missing topo CSV: $path")
    elevations = Dict{Int, Float64}()
    open(path, "r") do io
        header = split_csv_line(readline(io))
        grid_index = findfirst(==("grid_id"), header)
        elev_index = findfirst(==("mean_elevation_m"), header)
        grid_index !== nothing || error("topo CSV is missing grid_id")
        elev_index !== nothing || error("topo CSV is missing mean_elevation_m")
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split_csv_line(line)
            elevations[parse(Int, parts[grid_index])] = parse(Float64, parts[elev_index])
        end
    end
    return elevations
end

function build_terrain_grid(lookup, elevations, nrows::Integer, ncols::Integer)
    terrain = fill(NaN, nrows, ncols)
    for (node_id, row) in lookup
        if haskey(elevations, node_id)
            0 <= row.row < nrows || continue
            0 <= row.col < ncols || continue
            terrain[row.row + 1, row.col + 1] = elevations[node_id]
        end
    end
    return terrain
end

function validate_targets(targets::AbstractVector{TargetPoint}, lookup)
    for target in targets
        haskey(lookup, target.grid_id) || error("grid id $(target.grid_id) is missing from lookup")
        row = lookup[target.grid_id]
        if row.row != target.row || row.col != target.col
            error(
                "Configured row/col for grid id $(target.grid_id) is $(target.row),$(target.col), " *
                "but lookup has $(row.row),$(row.col)"
            )
        end
    end
end

function x_center_m(col_zero_based::Integer)
    return X_ORIGIN_M + (Float64(col_zero_based) + 0.5) * CELL_SIZE_M
end

function y_center_m(row_zero_based::Integer)
    return Y_MIN_CENTER_M + Float64(row_zero_based) * CELL_SIZE_M
end

function tiff_row_index(row_zero_based::Integer, nrows::Integer)
    return nrows - row_zero_based
end

function clean_value(value)
    value == NODATA && return NaN32
    !isfinite(value) && return NaN32
    abs(value) > 1000.0f0 && return NaN32
    return Float32(value)
end

function write_pixel_values!(dest::Array{Float32, 3}, iz_row::Int, iz_col::Int, pixel)
    dest[1, iz_row, iz_col] = clean_value(Float32(getfield(getfield(pixel, :color), :val)))
    extra = getfield(pixel, :extra)
    for i in eachindex(extra)
        dest[i + 1, iz_row, iz_col] = clean_value(Float32(extra[i]))
    end
end

function read_local_cube(
    image,
    idomain::AbstractMatrix,
    depths_m::AbstractVector,
    elevations,
    terrain_grid::AbstractMatrix,
    target::TargetPoint,
    half_window_km::Integer,
)
    nrows, ncols = size(image)
    size(idomain) == (nrows, ncols) || error("idomain size $(size(idomain)) does not match TIF size $(size(image))")
    size(terrain_grid) == (nrows, ncols) || error("terrain size $(size(terrain_grid)) does not match TIF size $(size(image))")

    row_min = max(0, target.row - half_window_km)
    row_max = min(nrows - 1, target.row + half_window_km)
    col_min = max(0, target.col - half_window_km)
    col_max = min(ncols - 1, target.col + half_window_km)
    rows = collect(row_min:row_max)
    cols = collect(col_min:col_max)

    values = Array{Float32, 3}(undef, length(depths_m), length(rows), length(cols))
    fill!(values, NaN32)
    terrain_values = Matrix{Float64}(undef, length(rows), length(cols))
    fill!(terrain_values, NaN)

    for (ir, row0) in pairs(rows), (ic, col0) in pairs(cols)
        terrain_values[ir, ic] = terrain_grid[row0 + 1, col0 + 1]
        idomain[row0 + 1, col0 + 1] == 1 || continue
        pixel = image[tiff_row_index(row0, nrows), col0 + 1]
        write_pixel_values!(values, ir, ic, pixel)
    end

    x0 = x_center_m(target.col)
    y0 = y_center_m(target.row)
    x_offsets_km = [(x_center_m(col) - x0) / 1000.0 for col in cols]
    y_offsets_km = [(y_center_m(row) - y0) / 1000.0 for row in rows]
    x_coords_km = [x_center_m(col) / 1000.0 for col in cols]
    y_coords_km = [y_center_m(row) / 1000.0 for row in rows]
    surface = terrain_grid[target.row + 1, target.col + 1]
    isfinite(surface) || (surface = get(elevations, target.grid_id, NaN))
    isfinite(surface) || error("Missing surface elevation for target grid id $(target.grid_id)")
    target_row_index = findfirst(==(target.row), rows)
    target_col_index = findfirst(==(target.col), cols)
    target_row_index !== nothing || error("Target row was not included in local window")
    target_col_index !== nothing || error("Target col was not included in local window")

    return LocalCube(
        target,
        rows,
        cols,
        x_offsets_km,
        y_offsets_km,
        x_coords_km,
        y_coords_km,
        collect(Float64.(depths_m)),
        values,
        terrain_values,
        surface,
        target_row_index,
        target_col_index,
    )
end

function finite_values(arrays...)
    values = Float64[]
    for array in arrays
        for value in array
            if isfinite(value)
                push!(values, Float64(value))
            end
        end
    end
    return values
end

function quantile_colorrange(cubes::AbstractVector{LocalCube}; lo_q::Float64=0.02, hi_q::Float64=0.98)
    values = finite_values((cube.values for cube in cubes)...)
    isempty(values) && error("No finite resistivity values found in target windows")
    lo = quantile(values, lo_q)
    hi = quantile(values, hi_q)
    if lo == hi
        delta = max(abs(lo), 1.0) * 1e-3
        return (lo - delta, hi + delta)
    end
    return (lo, hi)
end

function crop_depth(cube::LocalCube; max_depth_m::Float64=MAX_PLOT_DEPTH_M)
    depth_indices = findall(<=(max_depth_m), cube.depths_m)
    isempty(depth_indices) && error("No depth levels shallower than or equal to $max_depth_m m")
    return LocalCube(
        cube.target,
        cube.rows,
        cube.cols,
        cube.x_offsets_km,
        cube.y_offsets_km,
        cube.x_coords_km,
        cube.y_coords_km,
        cube.depths_m[depth_indices],
        cube.values[depth_indices, :, :],
        cube.terrain_elevation_m,
        cube.surface_elevation_m,
        cube.target_row_index,
        cube.target_col_index,
    )
end

function finite_mean(values)
    total = 0.0
    count = 0
    for value in values
        if isfinite(value)
            total += Float64(value)
            count += 1
        end
    end
    return count == 0 ? NaN : total / count
end

function near_surface_map(cube::LocalCube; max_depth_m::Float64=50.0)
    depth_indices = findall(<=(max_depth_m), cube.depths_m)
    isempty(depth_indices) && error("No depths shallower than $max_depth_m m")
    ny = length(cube.rows)
    nx = length(cube.cols)
    map_row_col = Matrix{Float64}(undef, ny, nx)
    for iy in 1:ny, ix in 1:nx
        map_row_col[iy, ix] = finite_mean(@view cube.values[depth_indices, iy, ix])
    end
    return permutedims(map_row_col, (2, 1))
end

function ew_section(cube::LocalCube)
    section = @view cube.values[:, cube.target_row_index, :]
    return Float64.(permutedims(section, (2, 1)))
end

function ns_section(cube::LocalCube)
    section = @view cube.values[:, :, cube.target_col_index]
    return Float64.(permutedims(section, (2, 1)))
end

function ew_section_at_row(cube::LocalCube, row_index::Integer)
    section = @view cube.values[:, row_index, :]
    return Float64.(permutedims(section, (2, 1)))
end

function ns_section_at_col(cube::LocalCube, col_index::Integer)
    section = @view cube.values[:, :, col_index]
    return Float64.(permutedims(section, (2, 1)))
end

function depths_with_slice_depth(cube::LocalCube, slice_depth::Real)
    source_depths = Float64.(cube.depths_m)
    slice_depth = Float64(slice_depth)
    plot_depths = copy(source_depths)
    if !any(abs.(plot_depths .- 0.0) .< 1e-8)
        push!(plot_depths, 0.0)
    end
    if !any(abs.(plot_depths .- slice_depth) .< 1e-8)
        push!(plot_depths, slice_depth)
    end
    return sort(plot_depths)
end

function interpolate_section_depths(section::AbstractMatrix, source_depths::AbstractVector, target_depths::AbstractVector)
    source_depths = Float64.(source_depths)
    target_depths = Float64.(target_depths)
    out = fill(NaN, size(section, 1), length(target_depths))

    for (target_index, target_depth) in pairs(target_depths)
        upper_index = searchsortedfirst(source_depths, target_depth)
        if upper_index <= length(source_depths) && abs(source_depths[upper_index] - target_depth) < 1e-8
            out[:, target_index] .= section[:, upper_index]
        elseif upper_index <= 1
            out[:, target_index] .= section[:, 1]
        elseif upper_index > length(source_depths)
            out[:, target_index] .= section[:, end]
        else
            lower_index = upper_index - 1
            lower_depth = source_depths[lower_index]
            upper_depth = source_depths[upper_index]
            weight = (target_depth - lower_depth) / (upper_depth - lower_depth)
            for trace_index in axes(section, 1)
                lower_value = section[trace_index, lower_index]
                upper_value = section[trace_index, upper_index]
                if isfinite(lower_value) && isfinite(upper_value)
                    out[trace_index, target_index] = (1.0 - weight) * lower_value + weight * upper_value
                end
            end
        end
    end

    return out
end

function apply_clean_axis_style!(ax)
    ax.topspinevisible = false
    ax.rightspinevisible = false
    ax.xgridvisible = true
    ax.ygridvisible = true
    ax.xgridcolor = (:gray70, 0.24)
    ax.ygridcolor = (:gray70, 0.24)
    ax.xgridwidth = 0.8
    ax.ygridwidth = 0.8
    ax.leftspinecolor = (:black, 0.70)
    ax.bottomspinecolor = (:black, 0.70)
    ax.spinewidth = 1.0
    return ax
end

function annotate_plan!(ax, cube::LocalCube)
    hlines!(ax, [0.0]; color=(:white, 0.92), linewidth=1.8)
    vlines!(ax, [0.0]; color=(:white, 0.92), linewidth=1.8)
    hlines!(ax, [0.0]; color=(:black, 0.45), linewidth=0.8, linestyle=:dash)
    vlines!(ax, [0.0]; color=(:black, 0.45), linewidth=0.8, linestyle=:dash)
    scatter!(
        ax,
        [0.0],
        [0.0];
        marker=:xcross,
        markersize=18,
        color=:white,
        strokecolor=:black,
        strokewidth=1.4,
    )
end

function annotate_section!(ax)
    vlines!(ax, [0.0]; color=(:white, 0.82), linewidth=1.5)
    vlines!(ax, [0.0]; color=(:black, 0.42), linewidth=0.8, linestyle=:dash)
    for depth in (50.0, 100.0, 150.0)
        hlines!(ax, [depth]; color=(:white, 0.32), linewidth=0.8)
    end
end

function row_title(cube::LocalCube)
    target = cube.target
    lookup_x = x_center_m(target.col)
    lookup_y = y_center_m(target.row)
    elev_text = isfinite(cube.surface_elevation_m) ? @sprintf("%.1f m", cube.surface_elevation_m) : "n/a"
    return @sprintf(
        "grid %d   row/col %d/%d   EPSG:5070 %.0f, %.0f   lon/lat %s   surface %s",
        target.grid_id,
        target.row,
        target.col,
        lookup_x,
        lookup_y,
        target.lonlat,
        elev_text,
    )
end

function column_title(cube::LocalCube)
    target = cube.target
    lookup_x = x_center_m(target.col)
    lookup_y = y_center_m(target.row)
    elev_text = isfinite(cube.surface_elevation_m) ? @sprintf("%.1f m", cube.surface_elevation_m) : "n/a"
    site_line = isempty(target.label) ? "" : "$(target.response_class) | $(target.label)\n"
    return site_line * @sprintf(
        "grid %d | row/col %d/%d\nEPSG:5070 %.0f, %.0f\nlon/lat %s | surface %s",
        target.grid_id,
        target.row,
        target.col,
        lookup_x,
        lookup_y,
        target.lonlat,
        elev_text,
    )
end

function draw_sections(
    cubes::AbstractVector{LocalCube},
    output_dir::String;
    colorrange,
    output_prefix::String="Fig3",
    write_pdf::Bool=true,
)
    CairoMakie.activate!(type="png")
    set_theme!(fonts=(; regular=FIG_FONT_REGULAR, bold=FIG_FONT_BOLD))
    fig = Figure(
        size=(2300, 1780),
        backgroundcolor=:white,
        fontsize=18,
        figure_padding=(28, 34, 34, 60),
    )
    Label(
        fig[1, 1:3],
        "AEM log10 resistivity around target MRVA cells (0-200 m)";
        fontsize=24,
        font=:bold,
        tellwidth=false,
    )

    plotted = nothing
    for (i, cube) in pairs(cubes)
        Label(
            fig[2, i],
            column_title(cube);
            fontsize=15,
            halign=:center,
            tellwidth=false,
            lineheight=1.12,
            padding=(0, 0, 0, 0),
        )

        ax_map = Axis(
            fig[3, i],
            title="Median 0-50 m",
            xlabel="E-W offset (km)",
            ylabel="S-N offset (km)",
            aspect=DataAspect(),
            backgroundcolor=:white,
            titlesize=18,
            xticklabelsize=15,
            yticklabelsize=15,
            xlabelsize=17,
            ylabelsize=17,
        )
        apply_clean_axis_style!(ax_map)
        plotted = heatmap!(
            ax_map,
            cube.x_offsets_km,
            cube.y_offsets_km,
            near_surface_map(cube);
            colormap=RESISTIVITY_COLORMAP,
            colorrange=colorrange,
            nan_color=(:white, 1.0),
            interpolate=false,
        )
        annotate_plan!(ax_map, cube)

        ax_ew = Axis(
            fig[4, i],
            title="E-W section",
            xlabel="E-W offset (km)",
            ylabel="Depth (m)",
            yreversed=true,
            backgroundcolor=:white,
            titlesize=18,
            xticklabelsize=15,
            yticklabelsize=15,
            xlabelsize=17,
            ylabelsize=17,
        )
        apply_clean_axis_style!(ax_ew)
        heatmap!(
            ax_ew,
            cube.x_offsets_km,
            cube.depths_m,
            ew_section(cube);
            colormap=RESISTIVITY_COLORMAP,
            colorrange=colorrange,
            nan_color=(:white, 1.0),
            interpolate=false,
        )
        annotate_section!(ax_ew)
        ylims!(ax_ew, last(cube.depths_m), first(cube.depths_m))

        ax_ns = Axis(
            fig[5, i],
            title="S-N section",
            xlabel="S-N offset (km)",
            ylabel="Depth (m)",
            yreversed=true,
            backgroundcolor=:white,
            titlesize=18,
            xticklabelsize=15,
            yticklabelsize=15,
            xlabelsize=17,
            ylabelsize=17,
        )
        apply_clean_axis_style!(ax_ns)
        heatmap!(
            ax_ns,
            cube.y_offsets_km,
            cube.depths_m,
            ns_section(cube);
            colormap=RESISTIVITY_COLORMAP,
            colorrange=colorrange,
            nan_color=(:white, 1.0),
            interpolate=false,
        )
        annotate_section!(ax_ns)
        ylims!(ax_ns, last(cube.depths_m), first(cube.depths_m))
    end

    Colorbar(
        fig[3:5, 4];
        colormap=RESISTIVITY_COLORMAP,
        limits=colorrange,
        label="log10 resistivity (ohm m)",
        width=24,
        ticklabelsize=16,
        labelsize=18,
        lowclip=RESISTIVITY_LOWCLIP_COLOR,
        highclip=RESISTIVITY_HIGHCLIP_COLOR,
    )
    colgap!(fig.layout, 28)
    rowgap!(fig.layout, 12)
    rowsize!(fig.layout, 1, 56)
    rowsize!(fig.layout, 2, any(!isempty(cube.target.label) for cube in cubes) ? 96 : 76)
    rowsize!(fig.layout, 3, Relative(0.40))
    rowsize!(fig.layout, 4, Relative(0.30))
    rowsize!(fig.layout, 5, Relative(0.30))

    mkpath(output_dir)
    png_path = joinpath(output_dir, "$(output_prefix)_resistivity_sections_2D.png")
    pdf_path = joinpath(output_dir, "$(output_prefix)_resistivity_sections_2D.pdf")
    save(png_path, fig; px_per_unit=PNG_PX_PER_UNIT)
    write_pdf && save(pdf_path, fig)
    println(png_path)
    write_pdf && println(pdf_path)
    return (; png_path, pdf_path=write_pdf ? pdf_path : nothing)
end

function nearest_depth_index(depths_m::AbstractVector, target_depth_m::Real)
    return argmin(abs.(Float64.(depths_m) .- Float64(target_depth_m)))
end

function target_xy_km(cube::LocalCube)
    return (
        cube.x_coords_km[cube.target_col_index],
        cube.y_coords_km[cube.target_row_index],
    )
end

function terrain_relative_surface(cube::LocalCube)
    return Float64.(permutedims(cube.terrain_elevation_m, (2, 1))) .- cube.surface_elevation_m
end

function ew_terrain_profile(cube::LocalCube)
    return Float64.(cube.terrain_elevation_m[cube.target_row_index, :]) .- cube.surface_elevation_m
end

function ns_terrain_profile(cube::LocalCube)
    return Float64.(cube.terrain_elevation_m[:, cube.target_col_index]) .- cube.surface_elevation_m
end

function apply_terrain_side_mask!(values::AbstractMatrix, cube::LocalCube; side::Symbol=:north)
    x0, y0 = target_xy_km(cube)
    nx, ny = size(values)

    if side == :north
        for iy in 1:ny
            if cube.y_coords_km[iy] < y0
                values[:, iy] .= NaN
            end
        end
    elseif side == :south
        for iy in 1:ny
            if cube.y_coords_km[iy] > y0
                values[:, iy] .= NaN
            end
        end
    elseif side == :east
        for ix in 1:nx
            if cube.x_coords_km[ix] < x0
                values[ix, :] .= NaN
            end
        end
    elseif side == :west
        for ix in 1:nx
            if cube.x_coords_km[ix] > x0
                values[ix, :] .= NaN
            end
        end
    elseif side == :all
        return values
    else
        error("Unsupported terrain side: $side")
    end

    return values
end

function mesh_terrain(cube::LocalCube; side::Symbol=:north)
    nx = length(cube.x_coords_km)
    ny = length(cube.y_coords_km)
    x = [Float32(cube.x_coords_km[ix]) for ix in 1:nx, _ in 1:ny]
    y = [Float32(cube.y_coords_km[iy]) for _ in 1:nx, iy in 1:ny]
    z = Float32.(terrain_relative_surface(cube))
    apply_terrain_side_mask!(z, cube; side=side)
    return x, y, z
end

function terrain_texture(cube::LocalCube; side::Symbol=:north)
    terrain = terrain_relative_surface(cube)
    nx, ny = size(terrain)
    texture = fill(NaN32, nx, ny)
    finite_terrain = filter(isfinite, vec(terrain))
    isempty(finite_terrain) && return texture
    lo = quantile(finite_terrain, 0.02)
    hi = quantile(finite_terrain, 0.98)
    hi <= lo && (hi = lo + 1.0)

    azimuth = deg2rad(315.0)
    altitude = deg2rad(45.0)
    lx = cos(altitude) * sin(azimuth)
    ly = cos(altitude) * cos(azimuth)
    lz = sin(altitude)

    for ix in 1:nx, iy in 1:ny
        center = terrain[ix, iy]
        isfinite(center) || continue

        ix_left = max(ix - 1, 1)
        ix_right = min(ix + 1, nx)
        iy_low = max(iy - 1, 1)
        iy_high = min(iy + 1, ny)
        left = isfinite(terrain[ix_left, iy]) ? terrain[ix_left, iy] : center
        right = isfinite(terrain[ix_right, iy]) ? terrain[ix_right, iy] : center
        low = isfinite(terrain[ix, iy_low]) ? terrain[ix, iy_low] : center
        high = isfinite(terrain[ix, iy_high]) ? terrain[ix, iy_high] : center

        dx = max(ix_right - ix_left, 1) * CELL_SIZE_M
        dy = max(iy_high - iy_low, 1) * CELL_SIZE_M
        dzdx = TERRAIN_HILLSHADE_EXAGGERATION * (right - left) / dx
        dzdy = TERRAIN_HILLSHADE_EXAGGERATION * (high - low) / dy
        nxn, nyn, nzn = -dzdx, -dzdy, 1.0
        invnorm = 1.0 / sqrt(nxn * nxn + nyn * nyn + nzn * nzn)
        hillshade = clamp((nxn * lx + nyn * ly + nzn * lz) * invnorm, 0.0, 1.0)
        hillshade = clamp((hillshade - 0.5) * TERRAIN_HILLSHADE_CONTRAST + 0.5, 0.0, 1.0)
        hillshade = 0.30 + 0.70 * hillshade
        elevation_norm = clamp((center - lo) / (hi - lo), 0.0, 1.0)
        texture[ix, iy] = Float32(clamp(0.45 * hillshade + 0.55 * elevation_norm, 0.0, 1.0))
    end

    apply_terrain_side_mask!(texture, cube; side=side)
    return texture
end

function mesh_ew(cube::LocalCube; depths_m=cube.depths_m)
    nx = length(cube.x_coords_km)
    nz = length(depths_m)
    _, y0 = target_xy_km(cube)
    terrain_profile = ew_terrain_profile(cube)
    x = [Float32(cube.x_coords_km[ix]) for ix in 1:nx, _ in 1:nz]
    y = fill(Float32(y0), nx, nz)
    z = [Float32(terrain_profile[ix] - depths_m[iz]) for ix in 1:nx, iz in 1:nz]
    return x, y, z
end

function mesh_ns(cube::LocalCube; depths_m=cube.depths_m)
    ny = length(cube.y_coords_km)
    nz = length(depths_m)
    x0, _ = target_xy_km(cube)
    terrain_profile = ns_terrain_profile(cube)
    x = fill(Float32(x0), ny, nz)
    y = [Float32(cube.y_coords_km[iy]) for iy in 1:ny, _ in 1:nz]
    z = [Float32(terrain_profile[iy] - depths_m[iz]) for iy in 1:ny, iz in 1:nz]
    return x, y, z
end

function mesh_ew_at_row(cube::LocalCube, row_index::Integer; depths_m=cube.depths_m)
    nx = length(cube.x_coords_km)
    nz = length(depths_m)
    y = fill(Float32(cube.y_coords_km[row_index]), nx, nz)
    x = [Float32(cube.x_coords_km[ix]) for ix in 1:nx, _ in 1:nz]
    terrain_profile = Float64.(cube.terrain_elevation_m[row_index, :]) .- cube.surface_elevation_m
    z = [Float32(terrain_profile[ix] - depths_m[iz]) for ix in 1:nx, iz in 1:nz]
    return x, y, z
end

function mesh_ns_at_col(cube::LocalCube, col_index::Integer; depths_m=cube.depths_m)
    ny = length(cube.y_coords_km)
    nz = length(depths_m)
    x = fill(Float32(cube.x_coords_km[col_index]), ny, nz)
    y = [Float32(cube.y_coords_km[iy]) for iy in 1:ny, _ in 1:nz]
    terrain_profile = Float64.(cube.terrain_elevation_m[:, col_index]) .- cube.surface_elevation_m
    z = [Float32(terrain_profile[iy] - depths_m[iz]) for iy in 1:ny, iz in 1:nz]
    return x, y, z
end

function mesh_horizontal(cube::LocalCube, depth_m::Real)
    nx = length(cube.x_coords_km)
    ny = length(cube.y_coords_km)
    x = [Float32(cube.x_coords_km[ix]) for ix in 1:nx, _ in 1:ny]
    y = [Float32(cube.y_coords_km[iy]) for _ in 1:nx, iy in 1:ny]
    z = Float32.(terrain_relative_surface(cube) .- Float64(depth_m))
    return x, y, z
end

function horizontal_slice_at_depth(cube::LocalCube, depth_m::Real)
    target_depth = Float64(depth_m)
    depths = Float64.(cube.depths_m)
    if target_depth <= first(depths)
        slice = @view cube.values[1, :, :]
        return Float64.(permutedims(slice, (2, 1)))
    elseif target_depth >= last(depths)
        slice = @view cube.values[end, :, :]
        return Float64.(permutedims(slice, (2, 1)))
    end

    upper_index = searchsortedfirst(depths, target_depth)
    lower_index = upper_index - 1
    lower_depth = depths[lower_index]
    upper_depth = depths[upper_index]
    weight = (target_depth - lower_depth) / (upper_depth - lower_depth)
    lower_slice = Float64.(permutedims(@view(cube.values[lower_index, :, :]), (2, 1)))
    upper_slice = Float64.(permutedims(@view(cube.values[upper_index, :, :]), (2, 1)))
    return (1.0 - weight) .* lower_slice .+ weight .* upper_slice
end

function masked_horizontal_slice(cube::LocalCube, depth_m::Real; side::Symbol=:all)
    slice = horizontal_slice_at_depth(cube, depth_m)
    apply_terrain_side_mask!(slice, cube; side=side)
    return slice
end

function side_condition(value::Real, split_value::Real, side::Symbol; include_boundary::Bool=true)
    if side == :east || side == :north
        return include_boundary ? value >= split_value : value > split_value
    elseif side == :west || side == :south
        return include_boundary ? value <= split_value : value < split_value
    else
        error("Unsupported cutaway side: $side")
    end
end

function quadrant_mask(cube::LocalCube; x_side::Symbol, y_side::Symbol, include_boundary::Bool=true)
    x0, y0 = target_xy_km(cube)
    nx = length(cube.x_coords_km)
    ny = length(cube.y_coords_km)
    return [
        side_condition(cube.x_coords_km[ix], x0, x_side; include_boundary=include_boundary) &&
        side_condition(cube.y_coords_km[iy], y0, y_side; include_boundary=include_boundary)
        for ix in 1:nx, iy in 1:ny
    ]
end

function mask_top_cutout!(z::AbstractMatrix, color::AbstractMatrix, cube::LocalCube; x_side::Symbol, y_side::Symbol)
    removed = quadrant_mask(cube; x_side=x_side, y_side=y_side, include_boundary=false)
    z[removed] .= NaN
    color[removed] .= NaN
    return z, color
end

function mask_horizontal_cut_floor!(z::AbstractMatrix, color::AbstractMatrix, cube::LocalCube; x_side::Symbol, y_side::Symbol)
    removed = quadrant_mask(cube; x_side=x_side, y_side=y_side, include_boundary=true)
    z[.!removed] .= NaN
    color[.!removed] .= NaN
    return z, color
end

function mask_ew_cutaway!(z::AbstractMatrix, color::AbstractMatrix, cube::LocalCube, slice_depth::Real; x_side::Symbol, depths_m=cube.depths_m)
    x0, _ = target_xy_km(cube)
    for ix in eachindex(cube.x_coords_km), iz in eachindex(depths_m)
        if side_condition(cube.x_coords_km[ix], x0, x_side; include_boundary=false) && depths_m[iz] < slice_depth
            z[ix, iz] = NaN32
            color[ix, iz] = NaN
        end
    end
    return z, color
end

function mask_ns_cutaway!(z::AbstractMatrix, color::AbstractMatrix, cube::LocalCube, slice_depth::Real; y_side::Symbol, depths_m=cube.depths_m)
    _, y0 = target_xy_km(cube)
    for iy in eachindex(cube.y_coords_km), iz in eachindex(depths_m)
        if side_condition(cube.y_coords_km[iy], y0, y_side; include_boundary=false) && depths_m[iz] < slice_depth
            z[iy, iz] = NaN32
            color[iy, iz] = NaN
        end
    end
    return z, color
end

function line3!(ax, xs, ys, zs; color=:black, linewidth=1.6)
    lines!(
        ax,
        Float32.(xs),
        Float32.(ys),
        Float32.(zs);
        color,
        linewidth,
    )
end

function draw_plane_guides!(ax, cube::LocalCube, slice_depth::Real)
    x0, y0 = target_xy_km(cube)
    ztop = 0.0
    zbot = -last(cube.depths_m)
    ew_intersection = ew_terrain_profile(cube) .- Float64(slice_depth)
    ns_intersection = ns_terrain_profile(cube) .- Float64(slice_depth)

    line3!(ax, [x0, x0], [y0, y0], [ztop, zbot]; linewidth=2.4)
    line3!(ax, cube.x_coords_km, fill(y0, length(cube.x_coords_km)), ew_intersection; linewidth=2.2)
    line3!(ax, fill(x0, length(cube.y_coords_km)), cube.y_coords_km, ns_intersection; linewidth=2.2)
end

function draw_horizontal_outline!(ax, cube::LocalCube, slice_depth::Real; side::Symbol=:all)
    x0, y0 = target_xy_km(cube)
    terrain_surface = terrain_relative_surface(cube)
    horizontal_z = terrain_surface .- Float64(slice_depth)
    nx = length(cube.x_coords_km)
    ny = length(cube.y_coords_km)

    if side == :south
        south_indices = findall(<=(y0), cube.y_coords_km)
        isempty(south_indices) && return nothing
        y_start = first(south_indices)
        y_end = last(south_indices)
        line3!(ax, cube.x_coords_km, fill(cube.y_coords_km[y_start], nx), horizontal_z[:, y_start]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(last(cube.x_coords_km), length(south_indices)), cube.y_coords_km[south_indices], horizontal_z[end, south_indices]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, reverse(cube.x_coords_km), fill(cube.y_coords_km[y_end], nx), reverse(horizontal_z[:, y_end]); linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(first(cube.x_coords_km), length(south_indices)), reverse(cube.y_coords_km[south_indices]), reverse(horizontal_z[1, south_indices]); linewidth=SLICE_OUTLINE_LINEWIDTH)
    elseif side == :west
        west_indices = findall(<=(x0), cube.x_coords_km)
        isempty(west_indices) && return nothing
        x_start = first(west_indices)
        x_end = last(west_indices)
        line3!(ax, fill(cube.x_coords_km[x_start], ny), cube.y_coords_km, horizontal_z[x_start, :]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, cube.x_coords_km[west_indices], fill(last(cube.y_coords_km), length(west_indices)), horizontal_z[west_indices, end]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(cube.x_coords_km[x_end], ny), reverse(cube.y_coords_km), reverse(horizontal_z[x_end, :]); linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, reverse(cube.x_coords_km[west_indices]), fill(first(cube.y_coords_km), length(west_indices)), reverse(horizontal_z[west_indices, 1]); linewidth=SLICE_OUTLINE_LINEWIDTH)
    elseif side == :all
        line3!(ax, cube.x_coords_km, fill(first(cube.y_coords_km), nx), horizontal_z[:, 1]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(last(cube.x_coords_km), ny), cube.y_coords_km, horizontal_z[end, :]; linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, reverse(cube.x_coords_km), fill(last(cube.y_coords_km), nx), reverse(horizontal_z[:, end]); linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(first(cube.x_coords_km), ny), reverse(cube.y_coords_km), reverse(horizontal_z[1, :]); linewidth=SLICE_OUTLINE_LINEWIDTH)
    else
        error("Unsupported horizontal outline side: $side")
    end
    return nothing
end

function draw_slice_outlines!(ax, cube::LocalCube, slice_depth::Real; horizontal_side::Symbol=:all)
    x0, y0 = target_xy_km(cube)
    max_depth = last(cube.depths_m)
    ew_top = ew_terrain_profile(cube)
    ns_top = ns_terrain_profile(cube)
    ew_bottom = ew_top .- max_depth
    ns_bottom = ns_top .- max_depth
    nx = length(cube.x_coords_km)
    ny = length(cube.y_coords_km)

    draw_horizontal_outline!(ax, cube, slice_depth; side=horizontal_side)

    line3!(ax, cube.x_coords_km, fill(y0, nx), ew_top; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, cube.x_coords_km, fill(y0, nx), ew_bottom; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, [first(cube.x_coords_km), first(cube.x_coords_km)], [y0, y0], [first(ew_top), first(ew_bottom)]; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, [last(cube.x_coords_km), last(cube.x_coords_km)], [y0, y0], [last(ew_top), last(ew_bottom)]; linewidth=SLICE_OUTLINE_LINEWIDTH)

    line3!(ax, fill(x0, ny), cube.y_coords_km, ns_top; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, fill(x0, ny), cube.y_coords_km, ns_bottom; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, [x0, x0], [first(cube.y_coords_km), first(cube.y_coords_km)], [first(ns_top), first(ns_bottom)]; linewidth=SLICE_OUTLINE_LINEWIDTH)
    line3!(ax, [x0, x0], [last(cube.y_coords_km), last(cube.y_coords_km)], [last(ns_top), last(ns_bottom)]; linewidth=SLICE_OUTLINE_LINEWIDTH)
end

function draw_terrain_outline!(ax, cube::LocalCube; side::Symbol=:north)
    x0, y0 = target_xy_km(cube)
    terrain_surface = terrain_relative_surface(cube)

    if side == :north
        north_indices = findall(>=(y0), cube.y_coords_km)
        isempty(north_indices) && return nothing
        y_start = first(north_indices)
        y_end = last(north_indices)
        nx = length(cube.x_coords_km)
        ny = length(north_indices)

        line3!(ax, cube.x_coords_km, fill(cube.y_coords_km[y_start], nx), terrain_surface[:, y_start]; color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(last(cube.x_coords_km), ny), cube.y_coords_km[north_indices], terrain_surface[end, north_indices]; color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, reverse(cube.x_coords_km), fill(cube.y_coords_km[y_end], nx), reverse(terrain_surface[:, y_end]); color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(first(cube.x_coords_km), ny), reverse(cube.y_coords_km[north_indices]), reverse(terrain_surface[1, north_indices]); color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
    elseif side == :east
        east_indices = findall(>=(x0), cube.x_coords_km)
        isempty(east_indices) && return nothing
        x_start = first(east_indices)
        x_end = last(east_indices)
        nx = length(east_indices)
        ny = length(cube.y_coords_km)

        line3!(ax, fill(cube.x_coords_km[x_start], ny), cube.y_coords_km, terrain_surface[x_start, :]; color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, cube.x_coords_km[east_indices], fill(last(cube.y_coords_km), nx), terrain_surface[east_indices, end]; color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, fill(cube.x_coords_km[x_end], ny), reverse(cube.y_coords_km), reverse(terrain_surface[x_end, :]); color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
        line3!(ax, reverse(cube.x_coords_km[east_indices]), fill(first(cube.y_coords_km), nx), reverse(terrain_surface[east_indices, 1]); color=:black, linewidth=SLICE_OUTLINE_LINEWIDTH)
    else
        error("Unsupported terrain side: $side")
    end

    return nothing
end

function draw_orthoslices(
    cubes::AbstractVector{LocalCube},
    output_dir::String;
    colorrange,
    view_azimuth_pi=nothing,
    view_elevation_pi=nothing,
    output_prefix::String="Fig3",
)
    GLMakie.activate!()
    set_theme!(fonts=(; regular=FIG_FONT_REGULAR, bold=FIG_FONT_BOLD))
    fig = Figure(size=(1800, 680), backgroundcolor=:white, fontsize=17)
    plotted = nothing
    z_label_text = "Depth (m)"
    orthoslice_colorrange = (colorrange[1], ORTHO_COLORBAR_MAX)

    for (i, cube) in pairs(cubes)
        slice_depth = i in (2, 3) ? ORTHO_HORIZONTAL_DEPTH_M_PANEL3 : ORTHO_HORIZONTAL_DEPTH_M
        vertical_depths = depths_with_slice_depth(cube, slice_depth)
        x0, y0 = target_xy_km(cube)
        default_azimuth = -0.30pi
        view_azimuth = panel_view_angle(view_azimuth_pi, i, default_azimuth)
        view_elevation = panel_view_angle(view_elevation_pi, i, 0.16pi)
        cut_x_side = :east
        cut_y_side = :south
        ax = Axis3(
            fig[1, i],
            title="",
            xlabel="Easting (km)",
            ylabel="Northing (km)",
            zlabel=z_label_text,
            aspect=(1.0, 1.0, 0.62),
            azimuth=view_azimuth,
            elevation=view_elevation,
            perspectiveness=0.42,
            zticks=([-200.0, -150.0, -100.0, -50.0, 0.0, 50.0], ["-200", "-150", "-100", "-50", "0", "50"]),
            titlesize=18,
            xlabelsize=18,
            ylabelsize=18,
            zlabelsize=18,
            xticklabelsize=15,
            yticklabelsize=15,
            zticklabelsize=15,
            protrusions=i == 3 ? (42, 70, 42, 4) : (42, 42, 42, 4),
        )

        tx, ty, tz = mesh_terrain(cube; side=:all)
        terrain_color = terrain_texture(cube; side=:all)
        mask_top_cutout!(tz, terrain_color, cube; x_side=cut_x_side, y_side=cut_y_side)
        surface!(
            ax,
            tx,
            ty,
            tz;
            color=terrain_color,
            colormap=TERRAIN_COLORMAP,
            colorrange=(0.0, 1.0),
            nan_color=(:white, 0.0),
            shading=true,
            diffuse=TERRAIN_DIFFUSE,
            specular=TERRAIN_SPECULAR,
            shininess=TERRAIN_SHININESS,
            transparency=false,
        )

        hx, hy, hz = mesh_horizontal(cube, slice_depth)
        hcolor = horizontal_slice_at_depth(cube, slice_depth)
        plotted = surface!(
            ax,
            hx,
            hy,
            hz;
            color=hcolor,
            colormap=RESISTIVITY_COLORMAP,
            colorrange=orthoslice_colorrange,
            nan_color=(:white, 0.0),
            shading=false,
            transparency=false,
        )

        boundary_row_index = cut_y_side == :south ? 1 : length(cube.rows)
        bx, by, bz = mesh_ew_at_row(cube, boundary_row_index; depths_m=vertical_depths)
        bcolor = interpolate_section_depths(ew_section_at_row(cube, boundary_row_index), cube.depths_m, vertical_depths)
        mask_ew_cutaway!(bz, bcolor, cube, slice_depth; x_side=cut_x_side, depths_m=vertical_depths)
        surface!(
            ax,
            bx,
            by,
            bz;
            color=bcolor,
            colormap=RESISTIVITY_COLORMAP,
            colorrange=orthoslice_colorrange,
            nan_color=(:white, 0.0),
            shading=false,
            transparency=false,
        )

        boundary_col_index = cut_x_side == :west ? 1 : length(cube.cols)
        sx, sy, sz = mesh_ns_at_col(cube, boundary_col_index; depths_m=vertical_depths)
        scolor = interpolate_section_depths(ns_section_at_col(cube, boundary_col_index), cube.depths_m, vertical_depths)
        mask_ns_cutaway!(sz, scolor, cube, slice_depth; y_side=cut_y_side, depths_m=vertical_depths)
        surface!(
            ax,
            sx,
            sy,
            sz;
            color=scolor,
            colormap=RESISTIVITY_COLORMAP,
            colorrange=orthoslice_colorrange,
            nan_color=(:white, 0.0),
            shading=false,
            transparency=false,
        )

        ex, ey, ez = mesh_ew(cube; depths_m=vertical_depths)
        ecolor = interpolate_section_depths(ew_section(cube), cube.depths_m, vertical_depths)
        surface!(
            ax,
            ex,
            ey,
            ez;
            color=ecolor,
            colormap=RESISTIVITY_COLORMAP,
            colorrange=orthoslice_colorrange,
            nan_color=(:white, 0.0),
            shading=false,
            transparency=false,
        )

        nx, ny, nz = mesh_ns(cube; depths_m=vertical_depths)
        ncolor = interpolate_section_depths(ns_section(cube), cube.depths_m, vertical_depths)
        surface!(
            ax,
            nx,
            ny,
            nz;
            color=ncolor,
            colormap=RESISTIVITY_COLORMAP,
            colorrange=orthoslice_colorrange,
            nan_color=(:white, 0.0),
            shading=false,
            transparency=false,
        )

        xlims!(ax, first(cube.x_coords_km), last(cube.x_coords_km))
        ylims!(ax, first(cube.y_coords_km), last(cube.y_coords_km))
        zlims!(ax, ORTHO_Z_LIMITS_M...)
    end

    colorbar_grid = GridLayout()
    fig[1, 4] = colorbar_grid
    Label(
        colorbar_grid[1, 1],
        "logρ";
        fontsize=18,
        font=:bold,
        tellwidth=false,
        tellheight=true,
        halign=:center,
    )
    Colorbar(
        colorbar_grid[2, 1];
        colormap=RESISTIVITY_COLORMAP,
        limits=orthoslice_colorrange,
        label="",
        width=22,
        height=Relative(0.70),
        ticks=ORTHO_COLORBAR_TICKS,
        ticklabelsize=ORTHO_COLORBAR_TICKLABELSIZE,
        lowclip=RESISTIVITY_LOWCLIP_COLOR,
        highclip=RESISTIVITY_HIGHCLIP_COLOR,
    )
    rowgap!(colorbar_grid, 6)
    colgap!(fig.layout, 44)

    mkpath(output_dir)
    png_path = joinpath(output_dir, "$(output_prefix)_resistivity_cutaway_3D.png")
    save(png_path, fig; px_per_unit=ORTHO_PNG_PX_PER_UNIT)
    println(png_path)
    return png_path
end

function load_cubes(
    targets::AbstractVector{TargetPoint},
    half_window_km::Integer;
    tif_path::String,
    depth_path::String,
    lookup_path::String,
    idomain_path::String,
    topo_path::String,
)
    depths_m = read_depths(depth_path)
    image = TiffImages.load(tif_path)
    idomain = Int8.(readdlm(idomain_path))
    lookup = read_lookup(lookup_path)
    elevations = read_surface_elevations(topo_path)
    validate_targets(targets, lookup)
    terrain_grid = build_terrain_grid(lookup, elevations, size(image, 1), size(image, 2))

    ndepths = 1 + length(getfield(image[1, 1], :extra))
    ndepths == length(depths_m) || error("TIF has $ndepths bands but depth file has $(length(depths_m)) levels")

    return [
        crop_depth(read_local_cube(image, idomain, depths_m, elevations, terrain_grid, target, half_window_km))
        for target in targets
    ]
end

function main(args=ARGS)
    parsed = parse_args(args)
    targets = parsed.targets_path === nothing ? TARGETS : read_targets(
        parsed.targets_path;
        target_set=parsed.target_set,
    )
    cubes = load_cubes(
        targets,
        parsed.half_window_km;
        tif_path=parsed.tif_path,
        depth_path=parsed.depth_path,
        lookup_path=parsed.lookup_path,
        idomain_path=parsed.idomain_path,
        topo_path=parsed.topo_path,
    )
    colorrange = parsed.fixed_colorrange === nothing ? quantile_colorrange(cubes) : parsed.fixed_colorrange
    section_paths = draw_sections(
        cubes,
        parsed.output_dir;
        colorrange,
        output_prefix=parsed.output_prefix,
        write_pdf=!parsed.png_only,
    )
    orthoslice_path = parsed.sections_only ? nothing : draw_orthoslices(
        cubes,
        parsed.output_dir;
        colorrange,
        view_azimuth_pi=parsed.view_azimuth_pi,
        view_elevation_pi=parsed.view_elevation_pi,
        output_prefix=parsed.output_prefix,
    )
    return (; section_paths..., orthoslice_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
