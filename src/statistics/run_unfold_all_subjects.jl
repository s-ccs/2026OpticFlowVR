using Unfold
using DataFrames
using CSV
using NPZ
using StatsModels
using CategoricalArrays
using BSplineKit

project_root = dirname(dirname(dirname(@__FILE__)))
export_root = joinpath(project_root, "output-iclabel", "unfold_export")
out_root = joinpath(project_root, "output-iclabel", "unfold_results", "all_subjects_conditionSpline")
mkpath(out_root)

subjects = [
    "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008",
    # "sub-009",  # excluded, missing condition x speed cells
    "sub-010", "sub-011", "sub-012", "sub-013", "sub-014", "sub-015", "sub-016",
    "sub-018", "sub-019", "sub-020", "sub-021", "sub-022", "sub-023", "sub-024",
    "sub-026", "sub-027", "sub-029", "sub-030", "sub-031", "sub-032", "sub-034",
    "sub-035",
]

condition_levels = [
    "Forward",
    "Random",
    "Rotation",
    "Spiral",
]

f = @formula(0 ~ 1 + condition * spl(speed, 5))

summary_rows = DataFrame(
    subject = String[],
    n_channels = Int[],
    n_times = Int[],
    n_trials = Int[],
    status = String[],
)

for subject in subjects
    println("\n", "=" ^ 80)
    println("Running Unfold model for ", subject)

    try
        data_file = joinpath(export_root, subject, "$(subject)_data.npy")
        events_file = joinpath(export_root, subject, "$(subject)_events.csv")
        times_file = joinpath(export_root, subject, "$(subject)_times.csv")
        channels_file = joinpath(export_root, subject, "$(subject)_channels.csv")

        data = npzread(data_file)
        events = CSV.read(events_file, DataFrame)
        times = CSV.read(times_file, DataFrame).time
        channels = CSV.read(channels_file, DataFrame).channel

        println("Data size: ", size(data), " = channels x dtimes x trials")
        println("Events rows: ", nrow(events))

        if size(data, 3) != nrow(events)
            error("Number of trials in data does not match number of event rows.")
        end

        events.condition = categorical(events.condition)
        levels!(events.condition, condition_levels)
        events.speed = Float64.(events.speed)
        # events.speed_centered = events.speed .- 1.4

        println("Fitting model: ", f)

        m = fit(UnfoldModel, f, events, data, times)
        ct = coeftable(m)
        # Add useful columns
        ct.subject .= subject
        ct.channel_name = [channels[ch] for ch in ct.channel]
        # Convert beta estimates from volts to microvolts
        ct.estimate_uV = ct.estimate .* 1_000_000

        # Reorder columns for readability
        ct = ct[:, [
            :subject,
            :channel,
            :channel_name,
            :time,
            :coefname,
            :estimate,
            :estimate_uV,
            :eventname,
            :group,
            :stderror,
        ]]

        out_file = joinpath(out_root, "$(subject)_coeftable.csv")

        CSV.write(
            out_file,
            ct;
            transform = (col, val) -> something(val, missing),
        )

        println("Saved: ", out_file)

        push!(
            summary_rows,
            (
                subject,
                size(data, 1),
                size(data, 2),
                size(data, 3),
                "OK",
            ),
        )

    catch err
        println("ERROR for ", subject)
        println(err)

        push!(
            summary_rows,
            (
                subject,
                -1,
                -1,
                -1,
                "FAILED: $(err)",
            ),
        )
    end
end

summary_file = joinpath(out_root, "model_fit_summary.csv")
CSV.write(summary_file, summary_rows)

println("\n", "=" ^ 80)
println("Done.")
println("Saved summary: ", summary_file)
println(summary_rows)
