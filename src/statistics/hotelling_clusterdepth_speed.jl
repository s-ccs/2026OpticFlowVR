using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Distributions
using HypothesisTests
using ClusterDepth
using Random

# Steps:
# 1. Load subject coefficients from 0 ~ 1 + condition + spl(speed, 5)
# 2. Average each spline beta across the predefined posterior ROI
# 3. At every time point jointly test the spline beta vector against zero
#    with a one-sample Hotelling T^2 test
# 4. Correct across time with ClusterDepth.jl using T^2 as the statistic and
#    subject-wise sign flips of the entire spline-beta vector.

const PROJECT_ROOT = dirname(dirname(dirname(@__FILE__)))
const INPUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "speed_main_effect",
    "all_subjects",
)
const OUT_ROOT = joinpath(
    PROJECT_ROOT,
    "output-iclabel",
    "unfold_results",
    "speed_main_effect",
    "hotelling_clusterdepth",
)
mkpath(OUT_ROOT)

const POSTERIOR_ROI = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

const TIME_MIN = 0.100
const TIME_MAX = 1.200

const CLUSTER_FORMING_ALPHA = 0.05
const FAMILYWISE_ALPHA = 0.05
const N_PERMUTATIONS = 5000
const RANDOM_SEED = 42

function load_coefficients()
    files = sort(filter(f -> endswith(f, "_coeftable.csv"), readdir(INPUT_ROOT; join=true)))
    isempty(files) && error("No coefficient tables found in $INPUT_ROOT")

    dfs = DataFrame[]
    for file in files
        println("Loading ", basename(file))
        push!(dfs, CSV.read(file, DataFrame))
    end
    return vcat(dfs...)
end

function spline_names(df::DataFrame)
    names = sort(unique(String.(df.coefname[startswith.(String.(df.coefname), "spl(speed,")])))
    isempty(names) && error("No spline coefficients were found.")
    return names
end

function prepare_roi_betas(df::DataFrame, spline_coefs::Vector{String})
    mask =
        in.(String.(df.channel_name), Ref(POSTERIOR_ROI)) .&
        in.(String.(df.coefname), Ref(spline_coefs)) .&
        (Float64.(df.time) .>= TIME_MIN) .&
        (Float64.(df.time) .<= TIME_MAX)

    roi = df[mask, :]
    isempty(roi) && error("No rows remained after ROI/time/spline selection.")

    # Average beta estimates over posterior channels within participant
    # Because the same regression model is fit at each channel, averaging the
    # channel-wise betas produces the beta for the channel-averaged response
    subject_roi = combine(
        groupby(roi, [:subject, :coefname, :time]),
        :estimate_uV => mean => :beta_uV,
    )

    subjects = sort(unique(String.(subject_roi.subject)))
    times = sort(unique(Float64.(subject_roi.time)))

    subject_index = Dict(s => i for (i, s) in enumerate(subjects))
    time_index = Dict(t => i for (i, t) in enumerate(times))
    coef_index = Dict(c => i for (i, c) in enumerate(spline_coefs))

    # ClusterDepth expects the last dimension to be subjects.  Use a
    # singleton 'channel' dimension so the statistic returned to ClusterDepth
    # is 1 x time, while preserving spline coefficients as an internal dimension:
    #   1 x time x spline-coefficient x subject
    data = fill(NaN, 1, length(times), length(spline_coefs), length(subjects))

    for row in eachrow(subject_roi)
        si = subject_index[String(row.subject)]
        ti = time_index[Float64(row.time)]
        ci = coef_index[String(row.coefname)]
        data[1, ti, ci, si] = Float64(row.beta_uV)
    end

    any(isnan, data) && error("Missing ROI beta values after constructing the spline-beta array.")

    return data, subjects, times
end

# Compute one-sample Hotelling's T^2 directly. This is mathematically equivalent
# to OneSampleHotellingT2Test(X, zeros(p)) and is used inside the permutation
# loop to avoid constructing many test objects.
function hotelling_t2(X::AbstractMatrix{<:Real})
    # X: subjects × spline coefficients
    n, p = size(X)
    n > p || error("Hotelling T^2 requires n_subjects > n_coefficients; got n=$n, p=$p")

    μ = vec(mean(X; dims=1))
    S = cov(X; corrected=true)

    # For a regular covariance matrix this equals n * μ' * inv(S) * μ.
    # Prefer a linear solve; fall back to a pseudoinverse if the estimated
    # covariance happens to be singular at a time point/permutation (Hotelling T^2).
    solved = try
        S \ μ
    catch err
        if err isa SingularException || err isa PosDefException
            pinv(S) * μ
        else
            rethrow()
        end
    end
    return n * dot(μ, solved)
end

# Custom statistic for ClusterDepth.jl
# Input dimensions: 1 x time x coefficients x subjects
# Output dimensions: 1 x time
function hotelling_stat(data::AbstractArray)
    size(data, 1) == 1 || error("Expected singleton channel dimension.")
    n_times = size(data, 2)
    n_coefs = size(data, 3)
    n_subjects = size(data, 4)

    out = Matrix{Float64}(undef, 1, n_times)

    for ti in 1:n_times
        # coefficient x subject -> subject x coefficient
        X = permutedims(Array(@view data[1, ti, :, :]), (2, 1))
        size(X) == (n_subjects, n_coefs) || error("Unexpected Hotelling matrix shape.")
        out[1, ti] = hotelling_t2(X)
    end

    return out
end

function analytic_hotelling_results(data, times)
    n_subjects = size(data, 4)
    n_coefs = size(data, 3)
    zero_vector = zeros(n_coefs)

    rows = DataFrame(
        time=Float64[],
        T2=Float64[],
        p_uncorrected=Float64[],
    )

    for (ti, time) in enumerate(times)
        X = permutedims(Array(@view data[1, ti, :, :]), (2, 1))
        test = OneSampleHotellingT2Test(X, zero_vector)
        T2 = hotelling_t2(X)
        push!(rows, (time, T2, pvalue(test)))
    end

    return rows
end

function hotelling_t2_threshold(n_subjects::Int, n_coefs::Int, alpha::Float64)
    # Under H0:
    #   ((n-p)/(p(n-1))) T^2 ~ F(p, n-p)
    fcrit = quantile(FDist(n_coefs, n_subjects - n_coefs), 1 - alpha)
    return (n_coefs * (n_subjects - 1) / (n_subjects - n_coefs)) * fcrit
end

function contiguous_intervals(mask::AbstractVector{Bool}, times::AbstractVector{<:Real}, pvals::AbstractVector{<:Real},)
    rows = DataFrame(
        cluster=Int[],
        time_start=Float64[],
        time_end=Float64[],
        duration_ms=Float64[],
        min_p_corrected=Float64[],
        n_timepoints=Int[],
    )

    i = 1
    cluster_id = 0
    dt = length(times) > 1 ? median(diff(times)) : 0.0

    while i <= length(mask)
        if !mask[i]
            i += 1
            continue
        end

        start_i = i
        while i < length(mask) && mask[i + 1]
            i += 1
        end
        end_i = i
        cluster_id += 1

        push!(rows, (
            cluster_id,
            times[start_i],
            times[end_i],
            (times[end_i] - times[start_i] + dt) * 1000,
            minimum(pvals[start_i:end_i]),
            end_i - start_i + 1,
        ))
        i += 1
    end

    return rows
end

function main()
    df = load_coefficients()
    splines = spline_names(df)
    println("Spline coefficients jointly tested: ", splines)

    data, subjects, times = prepare_roi_betas(df, splines)
    n_subjects = length(subjects)
    n_coefs = length(splines)

    println("Subjects: ", n_subjects)
    println("Time points: ", length(times), " (", first(times), " to ", last(times), " s)")
    println("Spline coefficients: ", n_coefs)
    println("ROI channels: ", join(POSTERIOR_ROI, ", "))
    println("Data shape for ClusterDepth: ", size(data), " = 1 x time x spline x subject")

    # Save the uncorrected Hotelling test at each time point as a
    # transparent diagnostic and for plotting/reporting
    results = analytic_hotelling_results(data, times)

    τ = hotelling_t2_threshold(n_subjects, n_coefs, CLUSTER_FORMING_ALPHA)
    println("Cluster-forming alpha: ", CLUSTER_FORMING_ALPHA)
    println("Hotelling T² cluster-forming threshold: ", round(τ; digits=4))
    println("Cluster-depth permutations: ", N_PERMUTATIONS)

    rng = MersenneTwister(RANDOM_SEED)

    # ClusterDepth's built-in sign permutation flips the complete
    # final-dimension slice. Because subjects are the final dimension here,
    # all spline coefficients and all time points of a participant receive the
    # same random sign on each permutation, preserving their covariance
    p_corrected_matrix = clusterdepth(
        rng,
        data;
        τ=τ,
        statfun=hotelling_stat,
        perm_type=:sign,
        side_type=:positive,  # T^2 is non-negative
        nperm=N_PERMUTATIONS,
        pval_type=:troendle,
    )

    p_corrected = vec(Array(p_corrected_matrix))
    length(p_corrected) == length(times) ||
        error("Corrected p-value vector length does not match time vector.")

    results.p_clusterdepth = p_corrected
    results.significant = p_corrected .< FAMILYWISE_ALPHA

    results_file = joinpath(OUT_ROOT, "hotelling_t2_clusterdepth_timeseries.csv")
    CSV.write(results_file, results)

    clusters = contiguous_intervals(results.significant, times, p_corrected)
    clusters_file = joinpath(OUT_ROOT, "significant_speed_clusters.csv")
    CSV.write(clusters_file, clusters)

    settings = DataFrame(
        key=[
            "model",
            "roi",
            "time_min_s",
            "time_max_s",
            "n_subjects",
            "n_spline_coefficients",
            "spline_coefficients",
            "cluster_forming_alpha",
            "hotelling_T2_threshold",
            "n_permutations",
            "familywise_alpha",
            "random_seed",
        ],
        value=[
            "0 ~ 1 + condition + spl(speed, 5)",
            join(POSTERIOR_ROI, ","),
            string(TIME_MIN),
            string(TIME_MAX),
            string(n_subjects),
            string(n_coefs),
            join(splines, ","),
            string(CLUSTER_FORMING_ALPHA),
            string(τ),
            string(N_PERMUTATIONS),
            string(FAMILYWISE_ALPHA),
            string(RANDOM_SEED),
        ],
    )
    CSV.write(joinpath(OUT_ROOT, "analysis_settings.csv"), settings)

    println("\nSaved: ", results_file)
    println("Saved: ", clusters_file)
    println("\nSignificant corrected intervals (p < ", FAMILYWISE_ALPHA, "):")
    if isempty(clusters)
        println("None")
    else
        show(clusters; allcols=true)
        println()
    end
end

main()
