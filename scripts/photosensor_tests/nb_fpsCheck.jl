### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

using DataFrames, XDF, Statistics, CSV, CairoMakie, Printf


md""" # Unity FPS check"""

begin
	sub = 0
	run = 20

	filepath = @sprintf("./timingTest/sub-%03i/ses-001/eeg/sub-%03i_task-none_events.csv",sub,sub)
end


data = CSV.read(filepath,DataFrame)

data_LLD = filter(:event => e -> occursin("LLD", e), data);

mean_frameDuration = round(mean(data_LLD.duration);digits=4)

mean_fps = 1/mean(data_LLD.duration)

std_fps = std(data_LLD.duration)

let
	f = Figure()
	ax = Axis(f[1,1])
	hist!(ax,data_LLD.duration,bins=20)
	ax.xlabel = "time [s]"
	ax.ylabel = "count"
	ax.title = "Histogram of frame durations"
	# xlims!(ax,0,maximum(data_LLD.duration))
	f
end

let
	f = Figure()
	ax = Axis(f[1,1])
	lines!(data_LLD.frame,data_LLD.onset)
	ax.xlabel = "frames"
	ax.ylabel = "onset [s]"
	ax.title = "Frame jitter"
	f
end

let
	ix = 1:20000
	onsets_diff = diff(unique(data_LLD.onset[ix]))

	# red vlines for stimOnset events
	stim_onsets = data.onset[data.event .== "stimOnset"]
	ix_stimOnset = findall(o -> o in stim_onsets, unique(data.onset[ix]))
	
	f = Figure()
	ax = Axis(f[1,1])
	lines!(onsets_diff)
	vlines!(ax, ix_stimOnset, color=:red, linestyle = :dash)
	ylims!(0,0.01) # uncomment to see impact of trial prep (goes up to ~ 1s)
	ax.xlabel = "Samples"
	ax.ylabel = "Frame duration [s]"
	ax.title = "Frame durations"
	f
end

# same plot as above, just directly plotting frame durations
# note: frame durations are shifted by one sample
let
	f = Figure()
	ax = Axis(f[1,1])
	# lines!(data.duration[data.duration.<0.05])
	lines!(data_LLD.duration[3:end])
	ax.xlabel = "Samples"
	ax.ylabel = "Frame duration [s]"
	ax.title = "Frame durations"
	f
end