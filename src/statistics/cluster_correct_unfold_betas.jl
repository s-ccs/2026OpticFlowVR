using CSV
using DataFrames
using Statistics
using Distributions
using PythonCall
using PyMNE
using CairoMakie

const pybuiltins = pyimport("builtins")

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))

const COEF_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "all_subjects",
)

const OUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "cluster_stats_4cond_posterior_100_1200ms_pymne",
)

mkpath(OUT_ROOT)

# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

const ALPHA_CLUSTER = 0.05
const CLUSTER_FORMING_ALPHA = 0.05
const N_PERMUTATIONS = 4096
const RANDOM_SEED = 42
const SFREQ = 250.0

const POSTERIOR_ROI = [
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",

    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",

    "O1",
    "Oz",
    "O2",
]

const TIME_MIN = 0.100
const TIME_MAX = 1.200
const SAMPLE_INTERVAL = 1.0 / SFREQ

const COEFFICIENTS = [
    "condition: Random",
    "condition: Rotation",
    "condition: Spiral",

    # Main spline effects (Forward)
    "spl(speed,1)",
    "spl(speed,2)",
    "spl(speed,3)",
    "spl(speed,4)",

    # Random x spline
    "condition: Random & spl(speed,1)",
    "condition: Random & spl(speed,2)",
    "condition: Random & spl(speed,3)",
    "condition: Random & spl(speed,4)",

    # Rotation x spline
    "condition: Rotation & spl(speed,1)",
    "condition: Rotation & spl(speed,2)",
    "condition: Rotation & spl(speed,3)",
    "condition: Rotation & spl(speed,4)",

    # Spiral x spline
    "condition: Spiral & spl(speed,1)",
    "condition: Spiral & spl(speed,2)",
    "condition: Spiral & spl(speed,3)",
    "condition: Spiral & spl(speed,4)",
]

const np = pyimport("numpy")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function safe_name(text::AbstractString)
    return replace(
        text,
        "condition: " => "condition_",
        " " => "_",
        "&" => "and",
        ":" => "",
        "/" => "_",
    )
end

function load_all_coefficients()
    files = sort(
        filter(
            file -> endswith(file, "_coeftable.csv"),
            readdir(COEF_ROOT; join=true),
        ),
    )

    isempty(files) &&
        error("No coefficient tables found in $COEF_ROOT")

    tables = DataFrame[]

    for file in files
        println("Loading ", basename(file))
        push!(tables, CSV.read(file, DataFrame))
    end

    return vcat(tables...)
end


"""
Construct an array with dimensions:
subjects x times x channels
for one Unfold coefficient
"""
function make_beta_array(df::DataFrame, coefname::String)
    mask =
        (df.coefname .== coefname) .&
        in.(df.channel_name, Ref(POSTERIOR_ROI)) .&
        (df.time .>= TIME_MIN) .&
        (df.time .<= TIME_MAX)

    sub_df = df[mask, :]

    isempty(sub_df) &&
        error("No rows found for coefficient: $coefname")

    subjects = sort(unique(String.(sub_df.subject)))
    times = sort(unique(Float64.(sub_df.time)))

    channel_table = unique(
        select(sub_df, [:channel, :channel_name]),
    )

    sort!(channel_table, :channel)

    channel_names = String.(channel_table.channel_name)

    subject_index = Dict(
        subject => index
        for (index, subject) in enumerate(subjects)
    )

    time_index = Dict(
        time => index
        for (index, time) in enumerate(times)
    )

    channel_index = Dict(
        channel => index
        for (index, channel) in enumerate(channel_names)
    )

    X = fill(
        NaN,
        length(subjects),
        length(times),
        length(channel_names),
    )

    for row in eachrow(sub_df)
        si = subject_index[String(row.subject)]
        ti = time_index[Float64(row.time)]
        ci = channel_index[String(row.channel_name)]

        X[si, ti, ci] = Float64(row.estimate_uV)
    end

    if any(isnan, X)
        error("Missing beta values for coefficient: $coefname")
    end

    return X, subjects, times, channel_names
end

"""
Create EEG-channel adjacency using MNE-Python
The returned adjacency object is a Python/SciPy sparse matrix
which can be passed directly back into the MNE cluster function
"""
function make_channel_adjacency(channel_names::Vector{String})
    # Convert Julia Vector{String} to a native Python list
    py_channel_names = pybuiltins.list(channel_names)

    info = PyMNE.create_info(
        ch_names=py_channel_names,
        sfreq=SFREQ,
        ch_types="eeg",
    )

    montage = PyMNE.channels.make_standard_montage("standard_1020")
    info.set_montage(montage; on_missing="ignore")
    adjacency_result = PyMNE.channels.find_ch_adjacency(info; ch_type="eeg")
    adjacency = adjacency_result[0]
    adjacency_names = pyconvert(Vector{String}, adjacency_result[1])

    if adjacency_names != channel_names
        println(
            "Warning: adjacency channel order differs from beta-array order",
        )
        println("Beta-array channels: ", channel_names)
        println("Adjacency channels:  ", adjacency_names)
    end

    return adjacency
end

function plot_cluster_mask(
    coefname::String,
    cluster_index::Int,
    cluster_mask::Matrix{Bool},
    T_obs::Matrix{Float64},
    times::Vector{Float64},
    channel_names::Vector{String},
    p_cluster::Float64,
    is_significant::Bool,
)
    size(cluster_mask) == size(T_obs) ||
        error("Cluster mask and T_obs have different dimensions.")

    # Show only t-values belonging to this cluster
    masked_t = fill(NaN, size(T_obs))
    masked_t[cluster_mask] .= T_obs[cluster_mask]

    cluster_values = T_obs[cluster_mask]
    isempty(cluster_values) &&
        return

    colour_limit = maximum(abs, cluster_values)

    # Avoid a zero-width colour range
    colour_limit =
        colour_limit == 0 ? 1.0 : colour_limit

    time_ms = times .* 1000
    channel_indices = collect(1:length(channel_names))

    status =
        is_significant ?
        "significant" :
        "not significant"

    fig = Figure(size=(1100, 700))

    ax = Axis(
        fig[1, 1],
        xlabel="Time (ms)",
        ylabel="Channel",
        title=(
            "$coefname — cluster $cluster_index\n" *
            "p = $(round(p_cluster; digits=4)), $status"
        ),
        yticks=(channel_indices, channel_names),
    )

    hm = heatmap!(
        ax,
        time_ms,
        channel_indices,
        masked_t;
        colormap=:RdBu,
        colorrange=(-colour_limit, colour_limit),
    )

    # Mark the analysed boundaries
    vlines!(
        ax,
        [first(time_ms), last(time_ms)];
        color=:black,
        linestyle=:dash,
        linewidth=1.5,
    )

    Colorbar(
        fig[1, 2],
        hm,
        label="Observed t-value",
    )

    cluster_plot_dir = joinpath(
        OUT_ROOT,
        "cluster_masks",
        safe_name(coefname),
    )

    mkpath(cluster_plot_dir)

    outfile = joinpath(
        cluster_plot_dir,
        "cluster_$(lpad(cluster_index, 3, '0'))_" *
        "$(status == "significant" ? "significant" : "nonsignificant").png",
    )

    save(outfile, fig; px_per_unit=2)
end

function run_cluster_test(df::DataFrame, coefname::String)
    println("\n", "="^80)
    println("Cluster test for: ", coefname)

    X, subjects, times, channel_names =
        make_beta_array(df, coefname)

    println(
        "X shape: ",
        size(X),
        " = subjects x times x channels",
    )

    n_subjects = size(X, 1)
    degrees_freedom = n_subjects - 1
    t_distribution = TDist(degrees_freedom)
    t_threshold = quantile(
        t_distribution,
        1 - CLUSTER_FORMING_ALPHA / 2,
    )

    println(
        "Cluster-forming threshold: |t| > ",
        round(t_threshold; digits=3),
    )
    adjacency = make_channel_adjacency(channel_names)

    # PythonCall converts the Julia Float64 array to a NumPy-compatible
    # object when it is passed to MNE
    cluster_result =
        PyMNE.stats.spatio_temporal_cluster_1samp_test(
            X;
            threshold=t_threshold,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            adjacency=adjacency,
            out_type="mask",
            seed=RANDOM_SEED,
            n_jobs=1,
        )

    T_obs = pyconvert(
        Array{Float64, 2},
        cluster_result[0],
    )

    clusters_py = cluster_result[1]
    cluster_p_values = pyconvert(
        Vector{Float64},
        cluster_result[2],
    )

    # Null distribution
    H0 = pyconvert(Vector{Float64}, cluster_result[3])

    rows = DataFrame(
        coefname=String[],
        cluster=Int[],
        p_cluster=Float64[],
        significant=Bool[],
        cluster_mass=Float64[],
        cluster_abs_mass=Float64[],
        cluster_sign=String[],
        time_start=Float64[],
        time_end=Float64[],
        duration_ms=Float64[],
        n_timepoints=Int[],
        n_channels=Int[],
        channels=String[],
        n_subjects=Int[],
        touches_left_edge=Bool[],
        touches_right_edge=Bool[],
        distance_from_left_ms=Float64[],
        distance_from_right_ms=Float64[],
    )

    significant_mask = falses(size(T_obs))
    n_clusters = pylen(clusters_py)
    cluster_masks = Array{Bool, 3}(undef, n_clusters, length(times), length(channel_names))
    name = safe_name(coefname)

    for python_index in 0:(n_clusters - 1)
        julia_index = python_index + 1
        cluster_mask = pyconvert(Array{Bool, 2}, clusters_py[python_index])
        cluster_masks[julia_index, :, :] .= cluster_mask
        p_cluster = cluster_p_values[julia_index]

        time_mask = vec(any(cluster_mask; dims=2))
        channel_mask = vec(any(cluster_mask; dims=1))
        cluster_times = times[time_mask]
        cluster_channels = channel_names[channel_mask]

        cluster_t_values = T_obs[cluster_mask]
        cluster_mass = sum(cluster_t_values)
        cluster_abs_mass = sum(abs, cluster_t_values)

        cluster_sign =
            cluster_mass > 0 ? "positive" :
            cluster_mass < 0 ? "negative" :
            "mixed"

        cluster_start = isempty(cluster_times) ? NaN : minimum(cluster_times)
        cluster_end = isempty(cluster_times) ? NaN : maximum(cluster_times)

        duration_ms =
            isempty(cluster_times) ?
            NaN :
            (cluster_end - cluster_start + SAMPLE_INTERVAL) * 1000

        touches_left_edge =
            !isempty(cluster_times) &&
            isapprox(cluster_start, first(times); atol=SAMPLE_INTERVAL / 10)
        touches_right_edge =
            !isempty(cluster_times) &&
            isapprox(cluster_end, last(times); atol=SAMPLE_INTERVAL / 10)
        distance_from_left_ms =
            isempty(cluster_times) ?
            NaN :
            (cluster_start - first(times)) * 1000
        distance_from_right_ms =
            isempty(cluster_times) ?
            NaN :
            (last(times) - cluster_end) * 1000

        is_significant = p_cluster < ALPHA_CLUSTER

        if is_significant
            significant_mask .|= cluster_mask
        end

        plot_cluster_mask(
            coefname,
            julia_index,
            cluster_mask,
            T_obs,
            times,
            channel_names,
            p_cluster,
            is_significant,
        )

        push!(
            rows,
            (
                coefname,
                julia_index,
                p_cluster,
                is_significant,
                cluster_mass,
                cluster_abs_mass,
                cluster_sign,
                cluster_start,
                cluster_end,
                duration_ms,
                length(cluster_times),
                length(cluster_channels),
                join(cluster_channels, ","),
                n_subjects,
                touches_left_edge,
                touches_right_edge,
                distance_from_left_ms,
                distance_from_right_ms,
            ),
        )
    end

    np.save(joinpath(OUT_ROOT, "$(name)_all_cluster_masks.npy"), cluster_masks)
    cluster_file = joinpath(
        OUT_ROOT,
        "$(name)_clusters.csv",
    )
    CSV.write(cluster_file, rows)

    np.save(
        joinpath(OUT_ROOT, "$(name)_T_obs.npy"),
        T_obs,
    )

    np.save(
        joinpath(OUT_ROOT, "$(name)_significant_mask.npy"),
        significant_mask,
    )

    np.save(
        joinpath(OUT_ROOT, "$(name)_H0.npy"),
        H0,
    )

    CSV.write(
        joinpath(OUT_ROOT, "times.csv"),
        DataFrame(time=times),
    )

    CSV.write(
        joinpath(OUT_ROOT, "channels.csv"),
        DataFrame(channel=channel_names),
    )

    println("Saved: ", cluster_file)
    println("Significant clusters:")

    significant_rows = rows[rows.significant, :]

    if isempty(significant_rows)
        println("None")
    else
        show(significant_rows; allcols=true)
        println()
    end

    return rows
end


function main()
    df = load_all_coefficients()
    all_results = DataFrame[]

    for coefname in COEFFICIENTS
        result = run_cluster_test(df, coefname)
        push!(all_results, result)
    end

    summary = vcat(all_results...)
    summary_file = joinpath(
        OUT_ROOT,
        "cluster_summary_all_coefficients.csv",
    )

    CSV.write(summary_file, summary)

    println("\nDone.")
    println("Saved results to: ", OUT_ROOT)
    println("Saved summary: ", summary_file)
end

main()
