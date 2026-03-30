using Revise # This must be the first line!
using Plots
using Measures
using Random

# import the generating Functions
includet("signal_generation/modulators.jl")
using .SignalGeneration

# Import plotting functions
includet("plotting_sp.jl")

# Set random seed for reproducibility
Random.seed!(1234)

# ==================
# 1. CONFIGURATION
# ==================
# Compute parameters
T_duration = 5.0                    # Duration of signal in seconds
Rs = 20.0                           # Symbol rate in symbols per second
fs_clean = 4400.0                   # Sampling frequency in Hz (high for smooth DSSS and FHSS)
num_symbols = Int(T_duration * Rs)  # Number of symbols

# =============================================================================================
# Generate sensors that will receive the mixed signals. This is the environment for FastICA.

Base.@kwdef struct Sensor
    id::Int
    location::Tuple{Float64, Float64}
end

sensors = [
    Sensor(1, (0.0, 0.0)),
    Sensor(2, (10.0, 0.0)),
    Sensor(3, (0.0, 10.0)),
    Sensor(4, (10.0, 10.0)),
    Sensor(5, (5.0, 5.0)),
    Sensor(6, (15.0, 5.0))
]


# Main target
# Strong signal moving east at 15 m/s, starting at origin
target1 = SignalGeneration.TargetEmitter(
    id = 1, location = (0.0, 0.0), velocity = (15.0, 0.0),
    fc = 511.0, fs = fs_clean, sps = 220, sps_chip = 8, β = 0.35, span = 10,
    snr_db = 15.0, hop_rate = 10.0, lfsr_bits = 7,
    hop_freq = [400.0, 800.0, 600.0, 1000.0, 500.0, 1100.0]
)

# Number of targets (including the main target)
num_targets = 5

# Loop comprehension to generate random background targets with varying parameters
background_targets = [
    TargetEmitter(
        id = i, 
        location = (rand(0.0:2000.0), rand(0.0:2000.0)), # Random X, Y
        velocity = (rand(-20.0:20.0), rand(-20.0:20.0)), # Random Vx, Vy
        fc = 511.0, fs = fs_clean, sps = 220, sps_chip = 8, β = 0.35, span = 10,
        snr_db = rand(5.0:15.0), # Random power levels
        hop_rate = 10.0, 
        lfsr_bits = rand(7:10),  # Different Gold codes
        hop_freq = [400.0, 800.0, 600.0, 1000.0, 500.0, 1100.0]
    ) for i in 2:num_targets
]

# ==========================
# 2. GENERATE CLEAN SIGNALS
# ==========================
println("Running signal pipeline...")
signal_1 = SignalGeneration.generate_signal(target1, num_symbols);
println("Signal generation complete.")

# =====================================
# 2. VISUALIZE CLEAN VS ALIASED SIGNAL
# =====================================
println("Generating clean and aliased signals for comparison...")
qpsk_signal_clean = signal_1.qpsk_clean; # Passband QPSK signal (real part)
# Generate aliased version of the clean signal by downsampling
downsample_factor = 25;
# Aliasing. Compare a high rate sampled QPSK and an aliased version (downsampled)
qpsk_signal_aliased = qpsk_signal_clean[1:downsample_factor:end];

fs_aliased = target1.fs / downsample_factor;
sps_aliased = ceil(Int, target1.sps / downsample_factor);

# RRC filters
h_rrc = SignalGeneration.create_rrc_filter(target1.sps, target1.β, target1.span);
h_rrc_aliased = SignalGeneration.create_rrc_filter(sps_aliased, target1.β, target1.span);

# Demodulate them
I_clean, Q_clean = demodulate_signal(qpsk_signal_clean, target1.sps, target1.fc, target1.fs, h_rrc);
I_aliased, Q_aliased = demodulate_signal(qpsk_signal_aliased, sps_aliased, target1.fc, fs_aliased, h_rrc_aliased);

# Plottingclean and aliased signal
zoom_limit = 0.5 # seconds

clean_vs_aliased_plot = plot_clean_vs_aliased(
    qpsk_signal_clean, qpsk_signal_aliased, target1.fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, zoom_limit)

# clean_vs_aliased_plot = plot_qpsk_clean_vs_aliased(t_clean_zoom, sig_clean_zoom, t_aliased_zoom, sig_aliased_zoom, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, fc)
display(clean_vs_aliased_plot)
# Save the plot
savefig(clean_vs_aliased_plot, "src/Images/qpsk_clean_vs_aliased.png")
println("Clean vs Aliased plot saved to Images/qpsk_clean_vs_aliased.png")

clean_plot = plot_clean(qpsk_signal_clean, I_clean, Q_clean, target1.fs, zoom_limit)
zoom_x_start = 0.20
zoom_x_end   = 0.25
zoom_y_span  = (-0.15, 0.15) # Adjust based on your signal amplitude
lens!(clean_plot, [zoom_x_start, zoom_x_end], [zoom_y_span[1], zoom_y_span[2]], 
          inset = (1, bbox(0.72, 0, 0.4, 0.3)), # Place inset in center-right
          color =:red,
          linewidth=2,
          subplot=2)
display(clean_plot)
# Save the plot
savefig(clean_plot, "src/Images/qpsk_clean.png")
println("Clean plot saved to Images/qpsk_clean.png")

# ========================
# 3. PLOT DSSS (Baseband)
# ========================
println("Plotting DSSS signal...")
I_dsss, Q_dsss = demodulate_signal(
    signal_1.dsss,           # Passband DSSS signal
    target1.sps_chip,
    target1.fc, 
    target1.fs, 
    signal_1.matched_filter_chip
)

dsss_plot = plot_clean(signal_1.dsss, I_dsss, Q_dsss, target1.fs, 0.5; const_lims=(-1.5, 1.5))
savefig(dsss_plot, "src/Images/dsss_qpsk_time_zoom.png")
println("DSSS plot saved to Images/dsss_qpsk_time_zoom.png")

# ===========================
# 4. PLOT FHSS (Spectrogram)
# ===========================
samples_per_hop = round(Int, target1.fs / target1.hop_rate)
win_size = nextpow(2, samples_per_hop)
p_fhss = spectrogram(real.(signal_1.fhss), win_size, div(win_size, 2); fs=target1.fs)

p_spectrogram = heatmap(p_fhss.time, p_fhss.freq, 10*log10.(p_fhss.power),
    title="FHSS Spectrogram", color=:viridis)
savefig(p_spectrogram, "src/Images/fhss_qpsk_spectrogram.png")
println("FHSS plot saved to Images/fhss_qpsk_spectrogram.png")

# =====================================
# 5. PLOT FINAL CHANNEL (Faded + AWGN)
# =====================================
final_plot = plot_clean(real.(signal_1.final), real.(signal_1.final), imag.(signal_1.final), target1.fs, 0.5; const_lims=(-1.5, 1.5))
savefig(final_plot, "src/Images/final_qpsk_time_zoom.png")
println("Final channel plot saved to Images/final_qpsk_time_zoom.png")

# ==============================================================================
# GENERATE AND MIX MULTIPLE SIGNALS
# ==============================================================================
println("Generating signals for all targets...")

# Generate the full pipeline for every target in the array
all_signals = vcat(signal_1, [generate_signal(t, num_symbols) for t in background_targets])
targets = vcat(target1, background_targets)

# ==============================================================================
# MULTI-SENSOR MIXING (The environment for FastICA)
# ==============================================================================
println("Mixing signals in a multi-sensor environment...")
num_sensors = length(sensors) # Must be >= num_targets for FastICA to work well
num_targets = length(targets)

# Padding signals to the exact same length
max_len = maximum([length(sig.final) for sig in all_signals])
padded_signals = [vcat(sig.final, zeros(ComplexF64, max_len - length(sig.final))) for sig in all_signals]

# 2. Create the Source Matrix S (Size: num_targets x max_len)
# We take the real part because the physical antenna receives real voltages
S = vcat([real.(sig)' for sig in padded_signals]...) 

# 3. Create the physical Mixing Matrix A (Size: num_sensors x num_targets)
# In a real environment, this represents the attenuation and phase shifts.
# For BSS simulation, a random matrix perfectly models an unknown physical environment.
# A = randn(num_sensors, num_targets) 

A = zeros(Float64, num_sensors, num_targets)
for i in 1:num_sensors
    for j in 1:num_targets
        dx = sensors[i].location[1] - targets[j].location[1]
        dy = sensors[i].location[2] - targets[j].location[2]
        dist = sqrt(dx^2 + dy^2)
        
        # Physical voltage attenuation is inversely proportional to distance (1/d)
        A[i, j] = 1.0 / max(dist, 1.0) # Max prevents division by zero if target is ON the sensor
    end
end

# Normalize A so the matrix math stays stable
A = A ./ maximum(A)

# 4. Generate the received Sensor Matrix X (Size: num_sensors x max_len)
# X = A * S
X = A * S 

# Add ambient thermal noise to the sensors (The "clutter")
sensor_noise = 0.1 .* randn(size(X))
X_noisy = X .+ sensor_noise

println("Generated Sensor Matrix X of size: ", size(X_noisy))

# ==============================================================================
# 3. PLOTTING THE CHAOS THE SENSORS RECEIVE (MIXED SIGNALS)
# ==============================================================================
println("Plotting the mixed sensor input...")
sensor_1_reading = X_noisy[1, :]

zoom_limit = 0.5 # seconds
t_mixed = (0:length(sensor_1_reading)-1) ./ fs_clean

mixed_plot_time = plot(
    t_mixed[1:Int(zoom_limit*fs_clean)], 
    sensor_1_reading[1:Int(zoom_limit*fs_clean)],   # PLotting Sensor 1
    title="Mixed Sensor Input (5 Targets, FHSS + DSSS + Channel Effects)",
    xlabel="Time (s)", ylabel="Amplitude",
    label="Combined Passband Signal",
    color=:purple,
    grid=true
)

freqs, magnitude = get_spectrum(sensor_1_reading, fs_clean)
mixed_plot_freq = plot(freqs, magnitude,
    title="Frequency Spectrum: Mixed QPSK signal",
    xlabel="Frequency (Hz)",
    ylabel="Magnitude",
    label="Signal Spectrum",
    xlims=(-fs_clean/2, fs_clean/2),
    grid=true,
    legend=:outertop)

plots = [mixed_plot_time, mixed_plot_freq]
mixed_plot = plot(plots..., layout=(2,1), size=(800,400))

display(mixed_plot)

savefig(mixed_plot, "src/Images/mixed_sensor_input.png")
println("Mixed sensor input plot saved to Images/mixed_sensor_input.png")

using HDF5

# ==============================================================================
# SAVE DATASET FOR RECEIVER PIPELINE
# ==============================================================================
dataset_path = "data/sensor_recording.h5"

println("Saving dataset to $dataset_path ...")

h5open(dataset_path, "w") do file
    # Save the matrices
    file["X_noisy"] = X_noisy
    file["S_true"]  = S
    file["A_true"]  = A  # The true mixing matrix
    
    # Save the environment metadata needed for the receiver
    file["fs"] = target1.fs
    file["num_targets"] = num_targets
    file["num_sensors"] = num_sensors
    
    # Save the true locations for your tracking algorithm later!
    locations = hcat([collect(t.location) for t in targets]...)
    file["true_locations"] = locations
end

println("Dataset saved successfully!")