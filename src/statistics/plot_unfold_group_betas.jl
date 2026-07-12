using CSV
using DataFrames
using Statistics
using CairoMakie


project_root = dirname(dirname(dirname(@__FILE__)))

input_root = joinpath(
    project_root,
    "output",
    "unfold_results",
    "all_subjects",
)

out_root = joinpath(
    project_root,
    "output",
    "unfold_results",
    "plots",
)

mkpath(out_root)

println("Loading coefficient tables...")

files = filter(
    f -> occursin("_coeftable.csv", f),
    readdir(input_root, join=true),
)

df = DataFrame()
for file in files
    println("Loading ", basename(file))
    append!(
        df,
        CSV.read(file, DataFrame),
    )
end

println("Rows loaded: ", nrow(df))

posterior_channels = [
    "O1",
    "O2",
    "POz",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
]

df = filter(
    row -> row.channel_name in posterior_channels,
    df,
)

println("After ROI selection:")
println(nrow(df), " rows")

# first average channels within each subject
subject_roi = combine(
    groupby(
        df,
        [
            :subject,
            :coefname,
            :time,
        ],
    ),
    :estimate_uV => mean => :beta_uV,
)

# group average
group_beta = combine(
    groupby(
        subject_roi,
        [
            :coefname,
            :time,
        ],
    ),

    :beta_uV => mean => :mean_beta,
    :beta_uV => (x -> std(x) / sqrt(length(x))) => :sem,
)

function plot_beta(coef)
    data = filter(
        row -> row.coefname == coef,
        group_beta,
    )

    sort!(data, :time)

    is_speed_effect = occursin("speed_centered", coef)

    ylabel = if is_speed_effect
        "β (µV / m/s)"
    else
        "β (µV)"
    end
    fig = Figure(size=(900, 500))

    ax = Axis(
        fig[1, 1],
        xlabel="Time (s)",
        ylabel=ylabel,
        title=coef,
    )

    lines!(
        ax,
        data.time,
        data.mean_beta,
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

    vlines!(ax, [0],)
    hlines!(ax, [0],)

    safe_name = replace(
        coef,
        " " => "_",
        ":" => "",
        "&" => "and",
        "/" => "_",
    )

    outfile = joinpath(
        out_root,
        "$(safe_name).png",
    )

    save(outfile, fig)
    println("Saved: ", outfile)
end

coefficients_to_plot = [
    "speed_centered",

    "condition: Random",
    "condition: Rotation",
    "condition: Spiral",

    "condition: Random & speed_centered",
    "condition: Rotation & speed_centered",
    "condition: Spiral & speed_centered",
]

for coef in coefficients_to_plot
    plot_beta(coef)
end

println("Done.")