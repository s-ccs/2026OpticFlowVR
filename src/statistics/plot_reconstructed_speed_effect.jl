using CSV
using DataFrames
using Statistics
using CairoMakie
using StatsModels
using CategoricalArrays
using Unfold
using BSplineKit

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))
const COEF_ROOT = joinpath(PROJECT_ROOT, "output-iclabel", "unfold_results", "all_subjects")
const CLUSTER_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "cluster_stats_4cond_posterior_100_1200ms_pymne",
)
const EXPORT_ROOT = joinpath(PROJECT_ROOT, "output-iclabel", "unfold_export")
const OUT_ROOT = joinpath(PROJECT_ROOT, "output-iclabel", "unfold_results", "plots")
mkpath(OUT_ROOT)

const SPEEDS = Float64[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
const CONDITION_LEVELS = ["Forward", "Random", "Rotation", "Spiral"]
const PREDICTION_CONDITION = "Forward"
const POSTERIOR_ROI = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
const FORMULA = @formula(0 ~ 1 + condition + spl(speed, 5))

function coefficient_files()
    files = sort(filter(path -> endswith(path, "_coeftable.csv"), readdir(COEF_ROOT; join=true)))
    isempty(files) && error("No coefficient tables found in $COEF_ROOT")
    return files
end

subject_from_filename(path::AbstractString) = replace(basename(path), "_coeftable.csv" => "")

function prepare_events(events::DataFrame)
    :condition in propertynames(events) || error("Events table has no :condition column.")
    :speed in propertynames(events) || error("Events table has no :speed column.")
    events.condition = categorical(String.(events.condition))
    levels!(events.condition, CONDITION_LEVELS)
    events.speed = Float64.(events.speed)
    return events
end

function prediction_design(events::DataFrame)
    applied_formula = apply_schema(FORMULA, schema(FORMULA, events), UnfoldModel)

    prediction_grid = DataFrame(
        condition=categorical(fill(PREDICTION_CONDITION, length(SPEEDS)); levels=CONDITION_LEVELS),
        speed=SPEEDS,
    )

    X_prediction = Matrix{Float64}(modelmatrix(applied_formula.rhs, prediction_grid))
    predictor_names = String.(coefnames(applied_formula.rhs))

    size(X_prediction, 2) == length(predictor_names) || error(
        "Design-matrix columns ($(size(X_prediction, 2))) do not match coefficient names ($(length(predictor_names)))."
    )

    return X_prediction, predictor_names
end

function subject_beta_matrix(coefficient_table::DataFrame, predictor_names::Vector{String})
    roi = filter(row -> String(row.channel_name) in POSTERIOR_ROI, coefficient_table)
    isempty(roi) && error("No rows remained after posterior ROI selection.")

    subject_roi = combine(groupby(roi, [:coefname, :time]), :estimate_uV => mean => :beta_uV)
    times = sort(unique(Float64.(subject_roi.time)))

    beta = fill(NaN, length(predictor_names), length(times))
    coefficient_index = Dict(name => i for (i, name) in enumerate(predictor_names))
    time_index = Dict(time => i for (i, time) in enumerate(times))

    available_names = Set(String.(subject_roi.coefname))
    missing_names = filter(name -> !(name in available_names), predictor_names)
    isempty(missing_names) || error("Missing model terms: " * join(missing_names, ", "))

    for row in eachrow(subject_roi)
        name = String(row.coefname)
        haskey(coefficient_index, name) || continue
        beta[coefficient_index[name], time_index[Float64(row.time)]] = Float64(row.beta_uV)
    end

    any(isnan, beta) && error("Missing coefficient x time values after matrix construction.")
    return beta, times
end

function reconstruct_subject(coefficient_file::AbstractString)
    subject = subject_from_filename(coefficient_file)
    events_file = joinpath(EXPORT_ROOT, subject, "$(subject)_events.csv")
    isfile(events_file) || error("Events file not found for $subject: $events_file")

    coefficient_table = CSV.read(coefficient_file, DataFrame)
    events = prepare_events(CSV.read(events_file, DataFrame))
    X_prediction, predictor_names = prediction_design(events)
    beta, times = subject_beta_matrix(coefficient_table, predictor_names)
    predicted = X_prediction * beta

    return subject, predicted, times, predictor_names
end

function load_significant_speed_clusters()
    cluster_file = joinpath(
        CLUSTER_ROOT,
        "spl(speed,1)_clusters.csv",
    )

    isfile(cluster_file) ||
        error("Speed cluster file not found: $cluster_file")

    clusters = CSV.read(cluster_file, DataFrame)

    significant = filter(
        row -> row.significant == true,
        clusters,
    )

    isempty(significant) &&
        error(
            "No significant spl(speed,1) clusters found in: " *
            cluster_file
        )

    println("Significant spl(speed,1) clusters:")
    show(significant; allcols=true)
    println()

    return significant
end

function main()
    println("Reconstructing spline predictions...")
    subjects = String[]
    subject_predictions = Matrix{Float64}[]
    reference_times = nothing
    reference_predictor_names = nothing

    for file in coefficient_files()
        subject = subject_from_filename(file)
        println("Reconstructing ", subject)
        subject, predicted, times, predictor_names = reconstruct_subject(file)

        if isnothing(reference_times)
            reference_times = times
            reference_predictor_names = predictor_names
        else
            times == reference_times || error("Time vector differs for $subject.")
            predictor_names == reference_predictor_names || error("Predictor ordering differs for $subject.")
        end

        push!(subjects, subject)
        push!(subject_predictions, predicted)
    end

    times = reference_times::Vector{Float64}
    predictor_names = reference_predictor_names::Vector{String}
    println("Subjects reconstructed: ", length(subjects))
    println("Model coefficients: ", predictor_names)

    predictions = Array{Float64}(undef, length(subject_predictions), length(SPEEDS), length(times))
    for i in eachindex(subject_predictions)
        predictions[i, :, :] = subject_predictions[i]
    end

    group_mean = dropdims(mean(predictions; dims=1), dims=1)
    group_sem = dropdims(std(predictions; dims=1, corrected=true), dims=1) ./ sqrt(size(predictions, 1))
    significant_speed_clusters = load_significant_speed_clusters()

    prediction_rows = DataFrame(speed=Float64[], time=Float64[], mean_prediction_uV=Float64[], sem_uV=Float64[], n_subjects=Int[])
    for speed_i in eachindex(SPEEDS), time_i in eachindex(times)
        push!(prediction_rows, (
            SPEEDS[speed_i], times[time_i], group_mean[speed_i, time_i],
            group_sem[speed_i, time_i], length(subjects),
        ))
    end

    prediction_csv = joinpath(OUT_ROOT, "reconstructed_speed_effect.csv")
    CSV.write(prediction_csv, prediction_rows)
    println("Saved: ", prediction_csv)

    fig_lines = Figure(size=(1050, 650))
    ax_lines = Axis(
        fig_lines[1, 1],
        xlabel="Time (s)",
        ylabel="Model-predicted amplitude (µV)",
        title="Reconstructed nonlinear speed effect\nForward condition, posterior ROI",
        xticks = -0.2:0.1:0.8,
    )
    for cluster in eachrow(significant_speed_clusters)
        vspan!(ax_lines, cluster.time_start, cluster.time_end; color=(:gray, 0.18))
    end
    for speed_i in eachindex(SPEEDS)
        lines!(ax_lines, times, vec(group_mean[speed_i, :]); linewidth=2.5, label="$(SPEEDS[speed_i]) m/s")
    end
    vlines!(ax_lines, [0.0]; color=:black, linewidth=1)
    hlines!(ax_lines, [0.0]; color=:black, linewidth=1)
    axislegend(ax_lines; position=:lt, nbanks=2)
    line_file = joinpath(OUT_ROOT, "reconstructed_speed_effect_with_cluster.png")
    save(line_file, fig_lines; px_per_unit=2)
    println("Saved: ", line_file)

    fig_heatmap = Figure(size=(1050, 650))
    ax_heatmap = Axis(
        fig_heatmap[1, 1],
        xlabel="Time (s)",
        ylabel="Speed (m/s)",
        title="Reconstructed nonlinear speed effect\nForward condition, posterior ROI",
        yticks=(SPEEDS, string.(SPEEDS)),
        xticks = -0.2:0.1:0.8,
    )
    heatmap_values = permutedims(group_mean, (2, 1))
    limit = maximum(abs, heatmap_values)
    hm = heatmap!(ax_heatmap, times, SPEEDS, heatmap_values; colormap=:RdBu, colorrange=(-limit, limit))
    for cluster in eachrow(significant_speed_clusters)
        vlines!(
            ax_heatmap,
            [
                cluster.time_start,
                cluster.time_end,
            ];
        color=:black, linewidth=2, linestyle=:dash)
    end
    vlines!(ax_heatmap, [0.0]; color=:black, linewidth=1)
    Colorbar(fig_heatmap[1, 2], hm, label="Model-predicted amplitude (µV)")
    heatmap_file = joinpath(OUT_ROOT, "reconstructed_speed_heatmap_with_cluster.png")
    save(heatmap_file, fig_heatmap; px_per_unit=2)
    println("Saved: ", heatmap_file)

    println("Done.")

end

main()
