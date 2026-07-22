using DataFrames
using CSV
using Statistics
using HypothesisTests

project_root = dirname(dirname(dirname(@__FILE__)))

input_root = joinpath(
    project_root,
    "output-iclabel",
    "unfold_results",
    "all_subjects_conditionSpline",
)

out_root = joinpath(
    project_root,
    "output-iclabel",
    "unfold_results",
    "group_stats_conditionSpline",
)

mkpath(out_root)

println("Loading subject coefficient tables...")

files = filter(
    f -> occursin("_coeftable.csv", f),
    readdir(input_root, join=true),
)

all_data = DataFrame()

for file in files
    println("Loading: ", basename(file))
    df = CSV.read(file, DataFrame)
    append!(all_data, df)
end


println()
println("Loaded rows: ", nrow(all_data))
println("Subjects:")
println(unique(all_data.subject))
println("Coefficients:")
println(unique(all_data.coefname))

grouped = groupby(
    all_data,
    [
        :coefname,
        :channel,
        :channel_name,
        :time,
    ],
)

results = DataFrame(
    coefname = String[],
    channel = Int[],
    channel_name = String[],
    time = Float64[],
    mean_beta_uV = Float64[],
    t = Float64[],
    p_uncorrected = Float64[],
    n_subjects = Int[],
)

println()
println("Running second-level tests...")

for g in grouped

    betas = g.estimate_uV
    test = OneSampleTTest(betas, 0.0)

    push!(
        results,
        (
            first(g.coefname),
            first(g.channel),
            first(g.channel_name),
            first(g.time),
            mean(betas),
            test.t,
            pvalue(test),
            length(betas),
        )
    )
end

sort!(
    results,
    [
        :coefname,
        :channel,
        :time,
    ],
)

outfile = joinpath(
    out_root,
    "group_ttests_uncorrected.csv",
)

CSV.write(outfile, results)

println()
println("Saved:")
println(outfile)
println()
println("Preview:")
println(first(results, 20))
