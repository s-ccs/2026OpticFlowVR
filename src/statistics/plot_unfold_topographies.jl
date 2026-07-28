using CSV
using DataFrames
using Statistics
using PythonCall
using PyMNE


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))

const INPUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "all_subjects",
)

const CLUSTER_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "cluster_stats_4cond_posterior_100_1200ms_pymne",
)

const OUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "plots",
)

mkpath(OUT_ROOT)

# ---------------------------------------------------------------------
# Python modules
# ---------------------------------------------------------------------

const np = pyimport("numpy")
const plt = pyimport("matplotlib.pyplot")
const pybuiltins = pyimport("builtins")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

function make_python_list(values::Vector{String})
    result = pybuiltins.list()
    for value in values
        result.append(value)
    end
    return result
end

# ---------------------------------------------------------------------
# Load coefficient tables
# ---------------------------------------------------------------------

println("Loading coefficient tables...")

files = sort(
    filter(
        file -> endswith(file, "_coeftable.csv"),
        readdir(INPUT_ROOT; join=true),
    ),
)

isempty(files) &&
    error("No coefficient tables found in $INPUT_ROOT")

full_df = DataFrame()

for file in files
    println("Loading ", basename(file))
    append!(full_df, CSV.read(file, DataFrame))
end

println("Rows loaded: ", nrow(full_df))

# ---------------------------------------------------------------------
# Load significant Random clusters
# ---------------------------------------------------------------------

cluster_file = joinpath(
    CLUSTER_ROOT,
    "condition_Random_clusters.csv",
)

isfile(cluster_file) ||
    error("Cluster file not found: $cluster_file")

clusters = CSV.read(cluster_file, DataFrame)

significant_clusters = filter(
    row -> row.significant == true,
    clusters,
)

println(
    "Significant Random clusters: ",
    nrow(significant_clusters),
)

if isempty(significant_clusters)
    println("No significant Random clusters. No topographies created")
    exit()
end

# ---------------------------------------------------------------------
# Prepare full-scalp Random beta data
# ---------------------------------------------------------------------

random_df = filter(
    row -> row.coefname == "condition: Random",
    full_df,
)

isempty(random_df) &&
    error("No coefficient rows found for condition: Random")

channel_table = unique(
    select(random_df, [:channel, :channel_name]),
)

sort!(channel_table, :channel)
channel_names = String.(channel_table.channel_name)
println("Channels: ", length(channel_names))

# ---------------------------------------------------------------------
# MNE info and montage
# ---------------------------------------------------------------------

py_channel_names = make_python_list(channel_names)

info = PyMNE.create_info(
    ch_names=py_channel_names,
    sfreq=250.0,
    ch_types="eeg",
)
montage = PyMNE.channels.make_standard_montage("standard_1020")
info.set_montage(montage; on_missing="ignore",)

# ---------------------------------------------------------------------
# Calculate one whole-scalp beta map per significant cluster
# ---------------------------------------------------------------------

cluster_values = Vector{Vector{Float64}}()

for cluster in eachrow(significant_clusters)
    interval_df = filter(
        row ->
            row.time >= cluster.time_start &&
            row.time <= cluster.time_end,
        random_df,
    )

    # Average time samples within each subject and electrode first
    subject_channel = combine(
        groupby(
            interval_df,
            [:subject, :channel, :channel_name],
        ),
        :estimate_uV => mean => :subject_beta_uV,
    )

    # Then average subject estimates at each electrode
    group_channel = combine(
        groupby(
            subject_channel,
            [:channel, :channel_name],
        ),
        :subject_beta_uV => mean => :mean_beta_uV,
    )

    sort!(group_channel, :channel)
    observed_names = String.(group_channel.channel_name)
    observed_names == channel_names ||
        error("Channel order mismatch while preparing topography")

    push!(
        cluster_values,
        Float64.(group_channel.mean_beta_uV),
    )
end

scale_limit = maximum(abs, vcat(cluster_values...))

println(
    "Topography colour range: ±",
    round(scale_limit; digits=3),
    " µV",
)

# ---------------------------------------------------------------------
# Draw topographies
# ---------------------------------------------------------------------

n_clusters = nrow(significant_clusters)

fig, axes = plt.subplots(
    1,
    n_clusters;
    figsize=(4.8 * n_clusters, 4.8),
    squeeze=false,
)

mask_params_py = pybuiltins.dict()

mask_params_py["marker"] = "o"
mask_params_py["markerfacecolor"] = "none"
mask_params_py["markeredgecolor"] = "black"
mask_params_py["linewidth"] = 0.0
mask_params_py["markersize"] = 8.0

for index in 1:n_clusters
    cluster = significant_clusters[index, :]
    values = cluster_values[index]

    significant_channel_names = split(
        String(cluster.channels),
        ",",
    )

    channel_mask = [
        channel in significant_channel_names
        for channel in channel_names
    ]

    values_py = np.asarray(
        values;
        dtype=np.float64,
    )

    mask_py = np.asarray(
        channel_mask;
        dtype=np.bool_,
    )

    # Python array uses zero-based indexing
    ax = axes[0, index - 1]

    topomap = PyMNE.viz.plot_topomap(
        values_py,
        info;
        axes=ax,
        show=false,
        vlim=(-scale_limit, scale_limit),
        contours=6,
        mask=mask_py,
        mask_params=mask_params_py,
        sphere=(0, 0, 0, 0.110),
    )

    start_ms = round(Int, cluster.time_start * 1000)
    end_ms = round(Int, cluster.time_end * 1000)

    ax.set_title(
        "$(start_ms)-$(end_ms) ms\n" *
        "pcluster = $(round(cluster.p_cluster; digits=3))",
    )

    image = topomap[0]

    fig.colorbar(
        image;
        ax=ax,
        shrink=0.75,
        label="β (µV)",
    )
end

fig.tight_layout()
outfile = joinpath(OUT_ROOT, "condition_Random_cluster_topographies.png")
fig.savefig(outfile; dpi=300, bbox_inches="tight")
plt.close(fig)

println("Saved: ", outfile)
println("Done.")