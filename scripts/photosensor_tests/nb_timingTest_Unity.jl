### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

begin
	using XDF
	using CairoMakie
	using DataFrames
	using Printf
	using Unfold
	using PlutoLinks
	using Statistics
	using PlutoUI
end

md"# Unity timing test"

md"### Import"

function ingredients(path::String)
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end;

begin 
	LSLimport = ingredients("./LslTools.jl")
	LslTools = LSLimport.LslTools
end;

begin
	baseDirectory = raw"C:\Users\etest\UniStuttgart\sem4\timing_tests\sub-001-20260430T133609Z-3-001\sub-001\ses-001\eeg"

	file_names = [
		# "sub-001_ses-001_task-Default_run-002_eeg.xdf",
		# "sub-001_ses-001_task-Default_run-003_eeg.xdf",
		# "sub-001_ses-001_task-Default_run-004_eeg.xdf",
		# "sub-001_ses-001_task-Default_run-005_eeg.xdf",
		# "sub-001_ses-001_task-Default_run-006_eeg.xdf",
		"sub-001_ses-001_task-Default_run-007_eeg.xdf",
		"sub-001_ses-001_task-Default_run-009_eeg.xdf",
	]

	# Build full paths
	xdf_files = [joinpath(baseDirectory, f) for f in file_names]

	# recording_labels = ["5 min", "15 min", "30 min", "45 min", "60 min"]
	recording_labels = ["37 min", "60 min"]

	xdf_files 
end

begin
	# extract marker labels from Unity marker stream
	function get_marker_text(stim)
		raw_marker = stim["data"]
		ndims(raw_marker) == 2 ? string.(raw_marker[:, 1]) : string.(raw_marker)
	end

	# find the index of the timestamp in sorted_items that is closest to time t
	function nearest_index(sorted_times, t)
		i = searchsortedfirst(sorted_times, t)

		if i <= 1
			return 1
		elseif i > length(sorted_times)
			return length(sorted_times)
		else
			before = i - 1
			after = i
			return abs(sorted_times[before] - t) <= abs(sorted_times[after] - t) ? before : after
		end
	end

	function classify_latency(pairs; drift_threshold_ms=5.0, stable_residual_ms=10.0, unstable_residual_ms=20.0)
		# x = time since reconding start in seconds
		x = pairs.time_from_start_s
		# y = measured delay btw Unity marker and photodiode onset in ms
		y = pairs.latency_ms

		# create design matrix for linear regression
		# latency_ms = intercept + slope * time
		X = hcat(ones(length(x)), x)
		β = X \ y

		intercept_ms = β[1]
		slope_ms_per_s = β[2]
		slope_ms_per_min = slope_ms_per_s * 60
		total_drift_ms = slope_ms_per_s * maximum(x)

		yhat = X * β # predicted latency from the fitted line
		residuals = y .- yhat # diff btw actual and fitted latency

		# compute standard deviation and median absolute deviation of residuals
		residual_sd_ms = std(residuals)
		residual_mad_ms = median(abs.(residuals .- median(residuals)))

		classification =
			if abs(total_drift_ms) < drift_threshold_ms && residual_sd_ms < stable_residual_ms
				"constant / fixed offset only"
			elseif abs(total_drift_ms) >= drift_threshold_ms && residual_sd_ms < unstable_residual_ms
				"linear / correctable drift"
			else
				"unstable / jitter, dropped frames, or bad synchronization"
			end

		return (
			intercept_ms = intercept_ms,
			slope_ms_per_min = slope_ms_per_min,
			total_drift_ms = total_drift_ms,
			residual_sd_ms = residual_sd_ms,
			residual_mad_ms = residual_mad_ms,
			classification = classification
		)
	end

	function find_photo_edge_after_marker(photo_data, eeg_time, marker_time, 			sfreq;
		        search_start_ms = 0,
		        search_end_ms = 70,
		        baseline_ms = 15,
		        frac = 0.5)
	
	    i0 = searchsortedfirst(eeg_time, marker_time + search_start_ms / 1000)
	    i1 = searchsortedlast(eeg_time, marker_time + search_end_ms / 1000)
	
	    b0 = searchsortedfirst(eeg_time, marker_time - baseline_ms / 1000)
	    b1 = searchsortedlast(eeg_time, marker_time)
	
	    if i1 <= i0 || b1 <= b0
	        return missing
	    end
	
	    baseline = median(photo_data[b0:b1])
	    local_max = maximum(photo_data[i0:i1])
	
	    # dynamic threshold for this marker
	    local_thresh = baseline + frac * (local_max - baseline)
	
	    above = photo_data[i0:i1] .> local_thresh
	    crossings = findall(diff(above) .== 1)
	
	    if isempty(crossings)
	        return missing
	    end
	
	    edge_idx = i0 + first(crossings)
	    return eeg_time[edge_idx]
	end

	function analyze_xdf_latency(filepath, label; thresh=20000)
		streams = read_xdf(filepath)

		streams_sync = deepcopy(streams)
		streams_sync = LslTools.dejitter!(streams_sync, "eegosport")
		streams_sync = LslTools.sync_to_continuous!(streams_sync, "eegosport")

		eeg = LslTools.get_stream(streams_sync, "eegosport")
		stim = LslTools.get_stream(streams_sync, "UnityMarkers")

		# table with all Unity markers, e.g.
		# 12.345 | FULLSCREEN_PHOTO_ON
		# 12.478 | FULLSCREEN_PHOTO_OFF
		evts_all = DataFrame(
			onset = stim["time"],
			type = get_marker_text(stim)
		)

		# EEG sample index offset
		# (marker_time - first_eeg_time) * eeg_sampling_rate
		evts_all.latency = round.(Int, (evts_all.onset .- eeg["time"][1]) .* eeg["srate"])

		evts_on = evts_all[occursin.("FULLSCREEN_PHOTO_ON", evts_all.type), :]
		evts_off = evts_all[occursin.("FULLSCREEN_PHOTO_OFF", evts_all.type), :]
		evts_pulse = evts_all[occursin.("FULLSCREEN_PHOTO_PULSE", evts_all.type), :]

		photo_data = eeg["data"][:, end-2]
		sfreq = eeg["srate"]
		
		# ------------------------------------------------------------
		# 1. Detect ALL photodiode rising edges, independent of Unity
		# ------------------------------------------------------------
		
		photo_is_on = photo_data .> thresh
		ix_photo_on_raw = findall(diff(photo_is_on) .== 1)
		
		# For ON-to-ON, minimum possible interval is:
		# 5 ON frames + 5 OFF frames = 10 frames at 120 Hz = 83.3 ms.
		# Use a slightly smaller refractory period to remove duplicate threshold crossings.
		min_photo_gap_s = 0.06
		min_photo_gap_samples = round(Int, min_photo_gap_s * sfreq)
		
		keep = vcat(true, diff(ix_photo_on_raw) .>= min_photo_gap_samples)
		ix_photo_on = ix_photo_on_raw[keep]
		
		photo_on_times_all = eeg["time"][ix_photo_on]
		
		# ------------------------------------------------------------
		# 2. Match Unity ON markers to photodiode rising edges only
		# ------------------------------------------------------------
		
		unity_on_times = evts_on.onset
		
		pairs = DataFrame(
		    unity_time = Float64[],
		    photo_time = Float64[],
		    latency_ms = Float64[],
		    unity_index = Int[]
		)
		
		for (i, t) in enumerate(evts_on.onset)
		    photo_time = find_photo_edge_after_marker(
		        photo_data,
		        eeg["time"],
		        t,
		        eeg["srate"]
		    )
		
		    if !ismissing(photo_time)
		        push!(pairs, (
		            t,
		            photo_time,
		            (photo_time - t) * 1000,
		            i
		        ))
		    end
		end
		
		pairs.time_from_start_s = pairs.unity_time .- first(pairs.unity_time)
		
		stats = classify_latency(pairs)

		summary = DataFrame(
			label = [label],
			duration_min = [(size(eeg["data"], 1) / eeg["srate"]) / 60],
			eeg_srate = [eeg["srate"]],
			n_unity_on = [nrow(evts_on)],
			n_photo_on = [length(photo_on_times_all)],
			n_pairs = [nrow(pairs)],
			median_latency_ms = [median(pairs.latency_ms)],
			iqr_latency_ms = [quantile(pairs.latency_ms, 0.75) - quantile(pairs.latency_ms, 0.25)],
			initial_latency_ms = [stats.intercept_ms],
			drift_ms_per_min = [stats.slope_ms_per_min],
			total_drift_ms = [stats.total_drift_ms],
			residual_sd_ms = [stats.residual_sd_ms],
			residual_mad_ms = [stats.residual_mad_ms],
			classification = [stats.classification],
			file = [filepath]
		)

		return (
			label = label,
			filepath = filepath,
			eeg = eeg,
			stim = stim,
			evts_all = evts_all,
			evts_on = evts_on,
			evts_off = evts_off,
			evts_pulse = evts_pulse,
			photo_on_times_all = photo_on_times_all,
			ix_photo_on = ix_photo_on,
			ix_photo_on_raw = ix_photo_on_raw,
			pairs = pairs,
			summary = summary
		)
	end
end

md"### Inspect length of data"

md"### Create dataframe for photoOn events"

md"### Check sample drops"

md"""### Inter-onset intervals"""

md"### Epoch"

md"### Plot markers"

md"""### Epoch based on photoresistor data"""

begin
	results = [
		analyze_xdf_latency(xdf_files[i], recording_labels[i]; thresh=20000)
		for i in eachindex(xdf_files)
	]

	summary_all = vcat([r.summary for r in results]...)

	summary_all
end

begin
	selected = results[1]   # change to results[2], results[3], etc.

	eeg = selected.eeg
	stim = selected.stim
	evts_all = selected.evts_all
	evts_on = selected.evts_on
	evts_off = selected.evts_off
	evts_pulse = selected.evts_pulse
	evts = evts_on
	pairs = selected.pairs
end

begin
	println("Samples of EEG data: ", size(eeg["data"],1))
	println("Number of Unity markers: ", length(stim["data"]))
	println("Time of EEG recording: ", (size(eeg["data"],1)/eeg["srate"])/60, " min")
end

combine(groupby(evts_all, :type), nrow => :count)

let
	ixT = 1:500
	f = Figure()
	ax = Axis(f[1,1], title="EEG time vs. sample counter")
	scatter!(eeg["time"][ixT],eeg["data"][ixT,end])
	ax.xlabel = "Time [s]"
	ax.ylabel = "Sample count"
	current_figure()
end

# A few outlier are not visible due to the ylims. These are between blocks and are not representative so can easily be dismissed.
let
	f = Figure()
	ax = Axis(f[1,1], title="Inter-onset interval of FULLSCREEN_PHOTO_ON events")
	scatter!(diff(evts.onset))
	ax.xlabel = "ON events"
	ax.ylabel = "Time [s]"
	ylims!(0,0.5)
	f
end

data_e,times = Unfold.epoch(
	data = allowmissing(eeg["data"][:,end-2:end-1]'),
	tbl=evts,
	τ=(-0.3,0.3),
	sfreq=Int(round(eeg["srate"]))
);

# Unity marker -> epoch photodiode signal
let
	data_tmp = copy(data_e)
	#data_tmp[ismissing.(data_tmp)] .= NaN
	#data_tmp = disallowmissing(data_tmp)
	
	CairoMakie.image(times[1]..times[end],1..size(data_tmp,3),data_tmp[1,:,:])
	vlines!(current_axis(),[0])
 	xlims!(current_axis(),-0.02,0.1)
	current_axis().title = "Photoresistor epoched to photoOn events"
	current_axis().xlabel = "Time [s]"
	current_axis().ylabel = "Epoch"
	current_figure()
end

begin
	f1 = Figure(size=(900, 400))
	ax1 = Axis(f1[1, 1], title="Photoresistor epochs")

	for ep in 1:20
		lines!(ax1, times, data_e[1, :, ep])
	end

	ax1.xlabel = "Time [s]"
	ax1.ylabel = "Photoresistor signal"

	f1
end
# single epoch
# lines(data_tmp[1,:,6])

begin
	photo_data = eeg["data"][:, end-2]
	thresh = 20000
	sfreq = Int(round(eeg["srate"]))

	# detect photodiode rising edges
	ix_photo_on = findall(diff(photo_data .> thresh) .== 1)

	# remove duplicate threshold crossings
	min_gap_samples = round(Int, 0.06 * sfreq)

	mask = vcat(true, diff(ix_photo_on) .>= min_gap_samples)
	ix_photo_on = ix_photo_on[mask]

	evts_photo = DataFrame(
		type = fill("photo_ON", length(ix_photo_on)),
		latency = ix_photo_on
	)

	data_e_photo, times_photo = Unfold.epoch(
		data = allowmissing(eeg["data"][:, end-2:end-1]'),
		tbl = evts_photo,
		τ = (-0.05, 0.10),
		sfreq = sfreq
	)
end

# photodiode rising edge -> epoch photodiode signal
begin
	n_plot = min(5000, size(data_e_photo, 3))

	f = Figure(size=(900, 500))
	ax = Axis(f[1, 1], title="Photoresistor epochs, first $(n_plot)")

	image!(
		ax,
		times_photo[1]..times_photo[end],
		1..n_plot,
		data_e_photo[1, :, 1:n_plot]
	)

	xlims!(ax, -0.01, 0.02)

	f
end

# Expected average: mean(5:10) + mean(5:10) = 7.5 + 7.5 = 15 frames
# Median: 15 / 120 = 0.125 s
begin
	on_intervals = diff(evts_on.onset)

	println("Mean ON-to-ON interval: ", mean(on_intervals), " s")
	println("Median ON-to-ON interval: ", median(on_intervals), " s")
	println("Estimated Hz from ON-to-ON: ", 15 / median(on_intervals))
end

let
	rows = []

	for r in results
		photo_intervals_ms = diff(r.photo_on_times_all) .* 1000
		unity_intervals_ms = diff(r.evts_on.onset) .* 1000

		push!(rows, (
			label = r.label,
			n_unity_on = nrow(r.evts_on),
			n_photo_on = length(r.photo_on_times_all),
			n_pairs = nrow(r.pairs),
			unity_minus_photo = nrow(r.evts_on) - length(r.photo_on_times_all),
			median_unity_iti_ms = median(unity_intervals_ms),
			median_photo_iti_ms = median(photo_intervals_ms),
			min_photo_iti_ms = minimum(photo_intervals_ms),
			n_photo_intervals_under_70ms = sum(photo_intervals_ms .< 70)
		))
	end

	DataFrame(rows)
end

let
	r = results[1] # change to results[2] for 60 min
	photo_intervals_ms = diff(r.photo_on_times_all) .* 1000

	f = Figure(size=(800, 450))
	ax = Axis(f[1, 1], title="Photodiode rising-to-rising intervals")

	hist!(ax, photo_intervals_ms, bins=100)

	ax.xlabel = "Photodiode ON-to-ON interval [ms]"
	ax.ylabel = "Count"

	f
end

let
	f = Figure(size=(900, 600))
	ax = Axis(f[1, 1], title="Residuals after linear drift fit")

	colors = [:blue, :orange, :green, :red, :purple]

	# frame duration at target refresh rate
	frame_ms = 1000 / 120

	for (i, r) in enumerate(results)
		p = r.pairs
		max_t = maximum(p.time_from_start_s)
    	p = p[p.time_from_start_s .< max_t - 30, :]

		X = hcat(ones(nrow(p)), p.time_from_start_s)
		β = X \ p.latency_ms

		residuals = p.latency_ms .- X * β

		scatter!(
			ax,
			p.time_from_start_s ./ 60,
			residuals,
			markersize=3,
			color=colors[i],
			label=r.label
		)
	end

	# zero line
	hlines!(ax, [0], linewidth=2)

	# +1 / +2 frame reference lines
	hlines!(
		ax,
		[frame_ms, 2frame_ms],
		color=[:red, :darkred],
		linestyle=:dash,
		linewidth=2
	)

	text!(
		ax,
		maximum(vcat([r.pairs.time_from_start_s ./ 60 for r in results]...)),
		frame_ms,
		text="+1 frame ($(round(frame_ms, digits=2)) ms)",
		align=(:right, :bottom),
		color=:red
	)

	text!(
		ax,
		maximum(vcat([r.pairs.time_from_start_s ./ 60 for r in results]...)),
		2frame_ms,
		text="+2 frames ($(round(2frame_ms, digits=2)) ms)",
		align=(:right, :bottom),
		color=:darkred
	)

	ax.xlabel = "Time from start [min]"
	ax.ylabel = "Residual latency [ms]"

	axislegend(ax, position=:rb)

	f
end

let
	result_idx = 1
	p = results[result_idx].pairs[:, :]
	max_t = maximum(p.time_from_start_s)
    p = p[p.time_from_start_s .< max_t - 30, :]

	f = Figure(size=(900, 500))
	ax = Axis(f[1, 1], title="Raw latency with linear drift fit")

	scatter!(ax, p.time_from_start_s ./ 60, p.latency_ms, markersize=3)

	X = hcat(ones(nrow(p)), p.time_from_start_s)
	β = X \ p.latency_ms
	yhat = X * β

	lines!(ax, p.time_from_start_s ./ 60, yhat, linewidth=3)

	ax.xlabel = "Time from start [min]"
	ax.ylabel = "Latency photo - Unity marker [ms]"

	f
end

mean(results[1].pairs.latency_ms .> 10)