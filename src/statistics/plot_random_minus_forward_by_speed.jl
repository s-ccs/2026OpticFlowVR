using CSV
using DataFrames
using Statistics
using CairoMakie
using StatsModels
using CategoricalArrays
using Unfold
using BSplineKit

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))

const COEF_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "all_subjects_conditionSpline",
)

const EXPORT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_export",
)

const OUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "conditionSpline_plots",
)

mkpath(OUT_ROOT)

const SPEEDS = Float64[0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

const CONDITION_LEVELS = [
    "Forward",
    "Random",
    "Rotation",
    "Spiral",
]

const POSTERIOR_ROI = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

const FORMULA = @formula(
    0 ~ 1 + condition * spl(speed, 5)
)

const SUMMARY_TIME_MIN = 0.372
const SUMMARY_TIME_MAX = 0.540

function coefficient_files()
    files = sort(filter(
        path -> endswith(path, "_coeftable.csv"),
        readdir(COEF_ROOT; join=true),
    ))

    isempty(files) && error(
        "No coefficient tables found in $COEF_ROOT. " *
        "Check the COEF_ROOT folder name."
    )

    return files
end

subject_from_filename(path::AbstractString) =
    replace(basename(path), "_coeftable.csv" => "")

function prepare_events(events::DataFrame)
    events.condition = categorical(String.(events.condition))
    levels!(events.condition, CONDITION_LEVELS)
    events.speed = Float64.(events.speed)
    return events
end

function contrast_design(events::DataFrame)
    applied_formula = apply_schema(
        FORMULA,
        schema(FORMULA, events),
        UnfoldModel,
    )

    forward_grid = DataFrame(
        condition=categorical(
            fill("Forward", length(SPEEDS));
            levels=CONDITION_LEVELS,
        ),
        speed=SPEEDS,
    )

    random_grid = DataFrame(
        condition=categorical(
            fill("Random", length(SPEEDS));
            levels=CONDITION_LEVELS,
        ),
        speed=SPEEDS,
    )

    X_forward = Matrix{Float64}(
        modelmatrix(applied_formula.rhs, forward_grid),
    )

    X_random = Matrix{Float64}(
        modelmatrix(applied_formula.rhs, random_grid),
    )

    predictor_names = String.(coefnames(applied_formula.rhs))

    size(X_forward, 2) == length(predictor_names) ||
        error("Forward design matrix does not match coefficient names.")

    size(X_random, 2) == length(predictor_names) ||
        error("Random design matrix does not match coefficient names.")

    return X_random - X_forward, predictor_names
end

function subject_beta_matrix(coefficient_table::DataFrame, predictor_names::Vector{String})
    roi = filter(
        row -> String(row.channel_name) in POSTERIOR_ROI,
        coefficient_table,
    )

    isempty(roi) && error("No rows remained after ROI selection.")

    subject_roi = combine(
        groupby(roi, [:coefname, :time]),
        :estimate_uV => mean => :beta_uV,
    )

    times = sort(unique(Float64.(subject_roi.time)))

    beta = fill(
        NaN,
        length(predictor_names),
        length(times),
    )

    coef_index = Dict(
        name => i for (i, name) in enumerate(predictor_names)
    )

    time_index = Dict(
        time => i for (i, time) in enumerate(times)
    )

    available_names = Set(String.(subject_roi.coefname))
    missing_names = filter(
        name -> !(name in available_names),
        predictor_names,
    )

    isempty(missing_names) || error(
        "Coefficient table is missing model terms: " *
        join(missing_names, ", ")
    )

    for row in eachrow(subject_roi)
        name = String(row.coefname)
        haskey(coef_index, name) || continue
        beta[
            coef_index[name],
            time_index[Float64(row.time)],
        ] = Float64(row.beta_uV)
    end

    any(isnan, beta) && error(
        "Missing coefficient x time values after matrix construction."
    )

    return beta, times
end

function reconstruct_subject(coefficient_file::AbstractString)
    subject = subject_from_filename(coefficient_file)

    events_file = joinpath(
        EXPORT_ROOT,
        subject,
        "$(subject)_events.csv",
    )

    isfile(events_file) ||
        error("Events file not found for $subject: $events_file")

    coefficient_table = CSV.read(coefficient_file, DataFrame)
    events = prepare_events(CSV.read(events_file, DataFrame))

    design, predictor_names = contrast_design(events)
    beta, times = subject_beta_matrix(
        coefficient_table,
        predictor_names,
    )

    contrast = design * beta

    size(contrast) == (length(SPEEDS), length(times)) ||
        error("Unexpected contrast size: $(size(contrast))")

    return subject, contrast, times, predictor_names
end

function main()
    println("Reconstructing Random-minus-Forward contrasts...")

    subjects = String[]
    subject_contrasts = Matrix{Float64}[]
    reference_times = nothing
    reference_predictor_names = nothing

    for file in coefficient_files()
        subject = subject_from_filename(file)
        println("Reconstructing ", subject)

        subject, contrast, times, predictor_names =
            reconstruct_subject(file)

        if isnothing(reference_times)
            reference_times = times
            reference_predictor_names = predictor_names
        else
            times == reference_times ||
                error("Time vector differs for $subject.")

            predictor_names == reference_predictor_names ||
                error("Predictor ordering differs for $subject.")
        end

        push!(subjects, subject)
        push!(subject_contrasts, contrast)
    end

    times = reference_times::Vector{Float64}

    contrasts = Array{Float64}(
        undef,
        length(subject_contrasts),
        length(SPEEDS),
        length(times),
    )

    for subject_i in eachindex(subject_contrasts)
        contrasts[subject_i, :, :] =
            subject_contrasts[subject_i]
    end

    group_mean = dropdims(mean(contrasts; dims=1), dims=1)
    group_sem = dropdims(
        std(contrasts; dims=1, corrected=true),
        dims=1,
    ) ./ sqrt(size(contrasts, 1))

    rows = DataFrame(
        speed=Float64[],
        time=Float64[],
        mean_random_minus_forward_uV=Float64[],
        sem_uV=Float64[],
        n_subjects=Int[],
    )

    for speed_i in eachindex(SPEEDS)
        for time_i in eachindex(times)
            push!(rows, (
                SPEEDS[speed_i],
                times[time_i],
                group_mean[speed_i, time_i],
                group_sem[speed_i, time_i],
                length(subjects),
            ))
        end
    end

    CSV.write(
        joinpath(OUT_ROOT, "random_minus_forward_by_speed.csv"),
        rows,
    )

    interval_mask = (
        (times .>= SUMMARY_TIME_MIN) .&
        (times .<= SUMMARY_TIME_MAX)
    )

    interval_rows = DataFrame(
        subject=String[],
        speed=Float64[],
        mean_random_minus_forward_uV=Float64[],
    )

    for subject_i in eachindex(subjects)
        for speed_i in eachindex(SPEEDS)
            push!(interval_rows, (
                subjects[subject_i],
                SPEEDS[speed_i],
                mean(contrasts[subject_i, speed_i, interval_mask]),
            ))
        end
    end

    CSV.write(
        joinpath(
            OUT_ROOT,
            "random_minus_forward_late_window_by_subject.csv",
        ),
        interval_rows,
    )

    ticks = collect(-0.2:0.1:0.8)

    fig_lines = Figure(size=(1100, 650))
    ax_lines = Axis(
        fig_lines[1, 1],
        xlabel="Time (s)",
        ylabel="Random - Forward (µV)",
        title="Reconstructed Random-Forward contrast by speed\nPosterior ROI",
        xticks=ticks,
    )

    for speed_i in eachindex(SPEEDS)
        lines!(
            ax_lines,
            times,
            vec(group_mean[speed_i, :]);
            linewidth=2.5,
            label="$(SPEEDS[speed_i]) m/s",
        )
    end

    vlines!(ax_lines, [0.0]; color=:black, linewidth=1)
    hlines!(ax_lines, [0.0]; color=:black, linewidth=1)
    axislegend(ax_lines; position=:lt, nbanks=2)

    save(
        joinpath(OUT_ROOT, "random_minus_forward_by_speed.png"),
        fig_lines;
        px_per_unit=2,
    )

    fig_heatmap = Figure(size=(1100, 650))
    ax_heatmap = Axis(
        fig_heatmap[1, 1],
        xlabel="Time (s)",
        ylabel="Speed (m/s)",
        title="Reconstructed Random-Forward contrast\nPosterior ROI",
        xticks=ticks,
        yticks=(SPEEDS, string.(SPEEDS)),
    )

    heatmap_values = permutedims(group_mean, (2, 1))
    limit = maximum(abs, heatmap_values)

    hm = heatmap!(
        ax_heatmap,
        times,
        SPEEDS,
        heatmap_values;
        colormap=:RdBu,
        colorrange=(-limit, limit),
    )

    vlines!(ax_heatmap, [0.0]; color=:black, linewidth=1)

    Colorbar(
        fig_heatmap[1, 2],
        hm,
        label="Random - Forward (µV)",
    )

    save(
        joinpath(OUT_ROOT, "random_minus_forward_heatmap.png"),
        fig_heatmap;
        px_per_unit=2,
    )

    interval_group = combine(
        groupby(interval_rows, :speed),
        :mean_random_minus_forward_uV => mean => :mean_contrast_uV,
        :mean_random_minus_forward_uV =>
            (x -> std(x) / sqrt(length(x))) => :sem_uV,
    )

    sort!(interval_group, :speed)

    fig_interval = Figure(size=(800, 550))
    ax_interval = Axis(
        fig_interval[1, 1],
        xlabel="Speed (m/s)",
        ylabel="Mean Random - Forward (µV)",
        title="Random-Forward contrast, 372-540 ms",
        xticks=(SPEEDS, string.(SPEEDS)),
    )

    errorbars!(
        ax_interval,
        interval_group.speed,
        interval_group.mean_contrast_uV,
        interval_group.sem_uV;
        whiskerwidth=10,
    )

    scatterlines!(
        ax_interval,
        interval_group.speed,
        interval_group.mean_contrast_uV;
        linewidth=2.5,
        markersize=12,
    )

    hlines!(ax_interval, [0.0]; color=:black, linewidth=1)

    save(
        joinpath(
            OUT_ROOT,
            "random_minus_forward_late_window_by_speed.png",
        ),
        fig_interval;
        px_per_unit=2,
    )

    println("Saved outputs to: ", OUT_ROOT)
end

main()
