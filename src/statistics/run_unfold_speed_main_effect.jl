using Unfold
using DataFrames
using CSV
using NPZ
using StatsModels
using CategoricalArrays
using BSplineKit

# Fits the subject-level model:
#   0 ~ 1 + condition + spl(speed, 5)

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))
const EXPORT_ROOT = joinpath(PROJECT_ROOT, "output-iclabel", "unfold_export")
const OUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "speed_main_effect",
    "all_subjects",
)
mkpath(OUT_ROOT)

const SUBJECTS = [
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-005",
    "sub-006",
    # "sub-007",  # excluded: abnormal signal quality
    "sub-008",
    # "sub-009",  # excluded: missing condition x speed cells
    "sub-010",
    "sub-011",
    "sub-012",
    "sub-013",
    "sub-014",
    "sub-015",
    "sub-016",
    # "sub-018",  # excluded: ERP not reproducible across trials
    "sub-019",
    # "sub-020",  # excluded: posterior ROI channels interpolated
    "sub-021",
    "sub-022",
    "sub-023",
    "sub-024",
    "sub-026",
    "sub-027",
    "sub-029",
    "sub-030",
    "sub-031",
    "sub-032",
    "sub-034",
    "sub-035",
    "sub-036",
    "sub-037",
    "sub-038",
    "sub-039",
    "sub-040",
]

const CONDITION_LEVELS = ["Forward", "Random", "Rotation", "Spiral"]
const FORMULA = @formula(0 ~ 1 + condition + spl(speed, 5))

summary_rows = DataFrame(
    subject=String[],
    n_channels=Int[],
    n_times=Int[],
    n_trials=Int[],
    n_spline_coefficients=Int[],
    status=String[],
)

for subject in SUBJECTS
    println("\n", "="^80)
    println("Running speed-main-effect Unfold model for ", subject)

    try
        data_file = joinpath(EXPORT_ROOT, subject, "$(subject)_data.npy")
        events_file = joinpath(EXPORT_ROOT, subject, "$(subject)_events.csv")
        times_file = joinpath(EXPORT_ROOT, subject, "$(subject)_times.csv")
        channels_file = joinpath(EXPORT_ROOT, subject, "$(subject)_channels.csv")

        for path in (data_file, events_file, times_file, channels_file)
            isfile(path) || error("Missing input file: $path")
        end

        data = npzread(data_file)
        events = CSV.read(events_file, DataFrame)
        times = Float64.(CSV.read(times_file, DataFrame).time)
        channels = String.(CSV.read(channels_file, DataFrame).channel)

        println("Data size: ", size(data), " = channels x times x trials")
        println("Events rows: ", nrow(events))

        size(data, 3) == nrow(events) ||
            error("Number of trials in data does not match number of event rows.")

        events.condition = categorical(String.(events.condition))
        levels!(events.condition, CONDITION_LEVELS)
        events.speed = Float64.(events.speed)

        println("Fitting model: ", FORMULA)
        m = fit(UnfoldModel, FORMULA, events, data, times)
        ct = coeftable(m)

        ct.subject .= subject
        ct.channel_name = [channels[ch] for ch in ct.channel]
        ct.estimate_uV = Float64.(ct.estimate)

        spline_names = sort(unique(String.(ct.coefname[startswith.(String.(ct.coefname), "spl(speed,")])))
        println("Spline coefficients: ", spline_names)

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

        out_file = joinpath(OUT_ROOT, "$(subject)_coeftable.csv")
        CSV.write(out_file, ct; transform=(col, val) -> something(val, missing))
        println("Saved: ", out_file)

        push!(summary_rows, (
            subject,
            size(data, 1),
            size(data, 2),
            size(data, 3),
            length(spline_names),
            "OK",
        ))

    catch err
        println("ERROR for ", subject)
        showerror(stdout, err)
        println()

        push!(summary_rows, (subject, -1, -1, -1, -1, "FAILED: $(err)"))
    end
end

summary_file = joinpath(OUT_ROOT, "model_fit_summary.csv")
CSV.write(summary_file, summary_rows)

println("\n", "="^80)
println("Done.")
println("Saved summary: ", summary_file)
println(summary_rows)
