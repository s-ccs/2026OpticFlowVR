using CSV
using DataFrames
using Statistics
using CairoMakie


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

project_root = dirname(dirname(dirname(@__FILE__)))

input_root = joinpath(
    project_root,
    "output-iclabel",
    "unfold_results",
    "all_subjects",
)

cluster_root = joinpath(
    project_root,
    "output-iclabel",
    "unfold_results",
    "cluster_stats_4cond_posterior_100_1200ms_pymne",
)

out_root = joinpath(
    project_root,
    "output-iclabel",
    "unfold_results",
    "plots",
)

mkpath(out_root)

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

posterior_channels = [
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

coefficients_to_plot = [
    "spl(speed,1)",
    "spl(speed,2)",
    "spl(speed,3)",
    "spl(speed,4)",
    "condition: Random",
    "condition: Rotation",
    "condition: Spiral",
]

# ---------------------------------------------------------------------
# Load all coefficient tables
# ---------------------------------------------------------------------

println("Loading coefficient tables...")

files = sort(
    filter(
        f -> occursin("_coeftable.csv", f),
        readdir(input_root; join=true),
    ),
)

isempty(files) &&
    error("No coefficient tables found in $input_root")

full_df = DataFrame()

for file in files
    println("Loading ", basename(file))
    append!(full_df, CSV.read(file, DataFrame))
end

println("Rows loaded: ", nrow(full_df))

roi_df = filter(
    row -> row.channel_name in posterior_channels,
    full_df,
)

println("ROI rows: ", nrow(roi_df))

# ---------------------------------------------------------------------
# ROI beta summaries
# ---------------------------------------------------------------------

subject_roi = combine(
    groupby(
        roi_df,
        [:subject, :coefname, :time],
    ),
    :estimate_uV => mean => :beta_uV,
)

group_beta = combine(
    groupby(
        subject_roi,
        [:coefname, :time],
    ),
    :beta_uV => mean => :mean_beta,
    :beta_uV => (x -> std(x) / sqrt(length(x))) => :sem,
)

function coefficient_ylabel(coef::AbstractString)
    if startswith(coef, "spl(speed,")
        return "Spline basis coefficient β (µV)"
    end

    return "β (µV)"
end

function safe_name(text::AbstractString)
    return replace(
        text,
        "condition: " => "condition_",
        " " => "_",
        ":" => "",
        "&" => "and",
        "/" => "_",
        "(" => "_",
        ")" => "",
        "," => "_",
    )
end

# ---------------------------------------------------------------------
# Individual beta plots
# ---------------------------------------------------------------------

function plot_beta(coef::String)
    data = filter(
        row -> row.coefname == coef,
        group_beta,
    )

    isempty(data) &&
        error("No group beta data found for coefficient: $coef")

    sort!(data, :time)

    fig = Figure(size=(900, 500))

    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel=coefficient_ylabel(coef),
        title=coef,
        xticks=-0.2:0.1:1.2,
    )

    band!(
        ax,
        data.time,
        data.mean_beta .- data.sem,
        data.mean_beta .+ data.sem,
        alpha=0.25,
    )

    lines!(
        ax,
        data.time,
        data.mean_beta,
        linewidth=3,
    )

    vlines!(ax, [0])
    hlines!(ax, [0])

    outfile = joinpath(
        out_root,
        "$(safe_name(coef)).png",
    )

    save(outfile, fig)
    println("Saved: ", outfile)
end

for coef in coefficients_to_plot
    plot_beta(coef)
end

# ---------------------------------------------------------------------
# Combined spline-basis coefficient time courses
# ---------------------------------------------------------------------

function plot_spline_basis_comparison()
    spline_coefs = [
        "spl(speed,1)",
        "spl(speed,2)",
        "spl(speed,3)",
        "spl(speed,4)",
    ]

    fig = Figure(size=(950, 550))

    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel="Spline basis coefficient β (µV)",
        title="Spline basis coefficients for speed",
        xticks=-0.2:0.1:1.2,
    )

    for coef in spline_coefs
        data = filter(
            row -> row.coefname == coef,
            group_beta,
        )

        isempty(data) &&
            error("No group beta data found for coefficient: $coef")

        sort!(data, :time)

        label = replace(
            coef,
            "spl(speed," => "Basis ",
            ")" => "",
        )

        lines!(
            ax,
            data.time,
            data.mean_beta;
            linewidth=3,
            label=label,
        )
    end

    vlines!(
        ax,
        [0];
        color=:black,
        linewidth=1,
    )

    hlines!(
        ax,
        [0];
        color=:black,
        linewidth=1,
    )

    axislegend(ax; position=:lt)

    outfile = joinpath(
        out_root,
        "spline_basis_coefficients_combined.png",
    )

    save(outfile, fig)
    println("Saved: ", outfile)
end

plot_spline_basis_comparison()

# ---------------------------------------------------------------------
# Combined condition plot
# ---------------------------------------------------------------------

function plot_condition_comparison()
    condition_coefs = [
        "condition: Random",
        "condition: Rotation",
        "condition: Spiral",

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

    fig = Figure(size=(950, 550))

    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel="β (µV)",
        title="Condition effects relative to Forward, controlling for nonlinear speed",
        xticks=-0.2:0.1:1.2,
    )

    for coef in condition_coefs
        data = filter(
            row -> row.coefname == coef,
            group_beta,
        )

        sort!(data, :time)

        label = replace(coef, "condition: " => "")

        lines!(
            ax,
            data.time,
            data.mean_beta,
            linewidth=3,
            label=label,
        )
    end

    vlines!(ax, [0])
    hlines!(ax, [0])
    axislegend(ax; position=:lt)

    outfile = joinpath(
        out_root,
        "condition_effects_combined.png",
    )

    save(outfile, fig)
    println("Saved: ", outfile)
end

plot_condition_comparison()

# ---------------------------------------------------------------------
# Load significant Random clusters
# ---------------------------------------------------------------------

random_cluster_file = joinpath(
    cluster_root,
    "condition_Random_clusters.csv",
)

random_clusters = CSV.read(
    random_cluster_file,
    DataFrame,
)

significant_random_clusters = filter(
    row -> row.significant == true,
    random_clusters,
)

println(
    "Significant Random clusters: ",
    nrow(significant_random_clusters),
)


# ---------------------------------------------------------------------
# Random beta plot with significant intervals
# ---------------------------------------------------------------------

function plot_random_with_clusters()
    data = filter(
        row -> row.coefname == "condition: Random",
        group_beta,
    )
    sort!(data, :time)

    fig = Figure(size=(950, 550))
    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel="β (µV)",
        title="Random relative to Forward",
        xticks=-0.2:0.1:1.2,
    )

    band!(
        ax,
        data.time,
        data.mean_beta .- data.sem,
        data.mean_beta .+ data.sem,
        alpha=0.25,
    )

    lines!(
        ax,
        data.time,
        data.mean_beta,
        linewidth=3,
    )

    # Mark each significant cluster interval
    for row in eachrow(significant_random_clusters)
        vspan!(
            ax,
            row.time_start,
            row.time_end,
            alpha=0.15,
        )
    end

    vlines!(ax, [0])
    hlines!(ax, [0])

    outfile = joinpath(
        out_root,
        "condition_Random_significant_clusters.png",
    )

    save(outfile, fig)
    println("Saved: ", outfile)
end


plot_random_with_clusters()
println("Done.")