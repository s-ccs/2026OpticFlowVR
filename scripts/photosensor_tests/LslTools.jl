module LslTools
	using Printf

	function sync_to_continuous!(streams,name_main="eegosport")
		stream_main = get_stream(streams,name_main)
		stream_main["type"] == "EEG" || @error("name_main stream is not an EEG stream")

		# Fix effective sampling rate to EEG sampling rate
		n_samples = size(stream_main["data"],1)
		duration = diff(stream_main["time"][[1,end]])[1]
		srate_eff = n_samples/duration
		factor = srate_eff/stream_main["srate"]
		@info @sprintf("changing effective sampling rate from %.4f to %.4f, factor %.4f",srate_eff,stream_main["srate"],factor)
		for k = values(streams)
			k["time"] = k["time"] .* factor
		end
		
		return streams
	end

	# Returns stream with "name", tries to match name exactly if multiple are found, raises error if multiple with exact match are found
	function get_stream(streams,name)
		keylist = collect(keys(streams))
		ix = find_stream(streams,name)

		if length(ix) == 0
			@error("name $name not found")
		elseif length(ix) > 1
			# multiple substring matches -> try to find stream that matches name exactly
			ix = find_stream(streams,name;exact=true)
			# throw error if there are multiple streams with exactly this name
			length(ix)==1 || @error("name $name found multiple times")
		end
		
		return streams[keylist[ix[1]]]
	end

	# Returns list of index of streams with "name"
	function find_stream(streams,name;exact=false)
		keylist = collect(keys(streams))
		streamNames = [streams[s]["name"] for s in keylist]

		if exact
			# try to find a stream matching exactly the name
			name_ix = findall(==(name), streamNames)
		else
			# return all streams that contain the substring
			name_ix =  findall(occursin.(name,streamNames))
		end
	end

	# Dejitter stream name_main & list of secondary streams according to the main stream.
	# Secondary streams need to be type 'Marker' that have the form abc@Float with the Float being the sample number.
	# ! Assumes that the last data-channel of the main stream contains the sample number of each sample.
		 dejitter_ant!(streams,name_main) = dejitter!(streams,name_main) 
	function dejitter_ant!(streams,name_main,name_secondary::String)
		stream_main 	 = get_stream(streams,name_main)
		stream_secondary = get_stream.(Ref(streams),name_secondary)
		
		samples_starting_at_1 = stream_main["data"][:,end].-stream_main["data"][1,end].+1
		coef = dejitter_coef(stream_main;samples=samples_starting_at_1)
		
		# Get sample number from EEG trigger (e.g. "12@1001." => 1001)
		samples_secondary = (Int.(parse.(Float64,[s[2] for s in split.(stream_secondary["data"],"@")])))[:,1]

		main_offset = stream_main["data"][1,end]

		# Correct the timestamps
		stream_main["time"]      = coef[1] .+ coef[2] .* (1:length(stream_main["time"]))
		stream_secondary["time"] = coef[1] .+ coef[2] .* (samples_secondary .- main_offset .+ 1) # the +1 to start at sample 1 not 0
		
		return streams
	end

	# Dejitter a single stream
	function dejitter!(streams,name)
		stream_main = get_stream(streams,name)
		coef = dejitter_coef(stream_main)
		stream_main["time"] = coef[1] .+ coef[2] .* (1:length(stream_main["time"]))
		
		return streams
	end

	# Calculate linear coefficient for jitter-correction of regular sampled EEG
	function dejitter_coef(stream_main;samples=missing)
		nsamp = length(stream_main["time"])
		if ismissing(samples)
			samples = 1:nsamp
		end
		coef = [ones(nsamp) samples] \ stream_main["time"]
		return coef
	end

end
