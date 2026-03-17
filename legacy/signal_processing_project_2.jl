using Plots
using Measures
using Plots.Measures
using Random
using FFTW
using Statistics
using DSP
# import the generating Functions
include("generating_functions.jl")

# Import plotting functions
include("plotting_sp.jl")

# Set random seed for reproducibility
Random.seed!(1234)

# =============
# Parameters
# =============

""" sps: samples per symbol
    N: total number of samples
    num_symbols: number of symbols to generate
    β: roll-off factor for RRC filter
    span: span of the RRC filter in symbol durations
    """
fs = 4400.0          # Sampling frequency in Hz (high for smooth DSSS and FHSS)
fc = 511.0          # Carrier frequency in Hz
T_duration = 5.0   # Duration of signal in seconds
Rs = 20.0          # Symbol rate in symbols per second

N = fs * T_duration  # Total number of samples
t = (0:N-1) / fs  # Time vector
sps = Int(fs / Rs) # Samples per symbol
num_symbols = Int(T_duration * Rs)  # Number of symbols

β = 0.35  # Roll-off factor for RRC filter
span = 10  # Span of RRC filter in symbol durations

downsample_factor = 25  # Downsampling factor for aliasing effect

# Carriers
I_carrier = cos.(2π * fc .* t)
Q_carrier = sin.(2π * fc .* t)

# ======================
# Generate QPSK symbols
# ======================

I_syms, Q_syms = generate_qpsk_symbols(num_symbols);

# =======================================================
# Display a clean signal & compare it to aliased version
# =======================================================
# Generate clean QPSK signal without noise. Add carriers here
# Now we extend the code to generate n number of signals, representing n moving targets, each with a different Doppler shift. We can then apply cyclostationary detection to identify the presence of these targets and estimate their parameters (e.g., velocity from Doppler shift).
# Generate n signals
num_signals = 10  # Number of signals to generate



# Apply a filter (RRC) to the transmitted signals to mimic the pulse shaping at the transmitter. 
# This will also help visualise the aliasing effect better in the frequency domain.
h_rrc = apply_rrc_impulse_response(sps, β, span); # The RRC filter coefficients

I_train, Q_train = zeros(Float64, Int(N)), zeros(Float64, Int(N));
I_train[1:sps:end] = I_syms[1:length(1:sps:N)]
Q_train[1:sps:end] = Q_syms[1:length(1:sps:N)]

# println("Length of I_train: ", length(I_train), " Length of Q_train: ", length(Q_train))
# println("Length of h_rrc: ", length(h_rrc))

h_len = length(h_rrc)
delay = floor(Int, h_len / 2)
# Filtering part: Apply the RRC, including the delay from convoluting.
conv_indices = delay + 1 : delay + length(I_train)

I_base = conv(I_train, h_rrc)[conv_indices]
Q_base = conv(Q_train, h_rrc)[conv_indices]

# println("Length of I_base: ", length(I_base), " Length of Q_base: ", length(Q_base))
# println("Length of t: ", length(t))

# True clean signal
qpsk_signal_clean = I_base .* I_carrier .- Q_base .* Q_carrier;

# Aliasing. Compare a high rate sampled QPSK and an aliased version (downsampled)
qpsk_signal_aliased = qpsk_signal_clean[1:downsample_factor:end];

fs_aliased = fs / downsample_factor;
sps_aliased = sps / downsample_factor;

h_rrc_aliased = apply_rrc_impulse_response(sps_aliased, β, span); # RRC filter for aliased signal

I_clean, Q_clean = demodulate_signal(qpsk_signal_clean, sps, fc, fs, h_rrc);
I_aliased, Q_aliased = demodulate_signal(qpsk_signal_aliased, sps_aliased, fc, fs_aliased, h_rrc_aliased);

# Plottingclean and aliased signal
zoom_limit = 0.5 # seconds

# t_clean_zoom, sig_clean_zoom = zoom_time(qpsk_signal_clean, fs, zoom_limit)
# t_aliased_zoom, sig_aliased_zoom = zoom_time(qpsk_signal_aliased, fs_aliased, zoom_limit)

clean_vs_aliased_plot = plot_clean_vs_aliased(qpsk_signal_clean, qpsk_signal_aliased, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, zoom_limit)

# clean_vs_aliased_plot = plot_qpsk_clean_vs_aliased(t_clean_zoom, sig_clean_zoom, t_aliased_zoom, sig_aliased_zoom, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, fc)
display(clean_vs_aliased_plot)
# Save the plot
savefig(clean_vs_aliased_plot, "signal_processing_project/Images/qpsk_clean_vs_aliased.png")


clean_plot = plot_clean(qpsk_signal_clean, I_clean, Q_clean, fs, zoom_limit)
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
savefig(clean_plot, "signal_processing_project/Images/qpsk_clean.png")


# =======================================================
# Apply DSSS spreading and RRC filtering
# =======================================================

""" PN_code: Pseudo-Noise code for spreading
    β: roll-off factor for RRC filter
    span: span of the RRC filter in symbol durations
    sps_chip: samples per chip
"""

sps_chip = 8  # Samples per chip
β = 0.35  # Roll-off factor for RRC filter
span = 10  # Span of RRC filter in symbol durations

h_rrc_chip = apply_rrc_impulse_response(sps_chip, β, span); # The RRC filter coefficients

# FINISH THE PN CODE

# Here we should generate a PN code. Possibly Barker code or Gold code.
# Barker code of length 11 for spreading.
PN_code = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0]

PN_code = generate_gold_code(127)

I_spread, Q_spread = apply_dsss(I_syms, Q_syms, PN_code, num_symbols);


# Generate impulse train for I and Q
I_impulse_train = zeros(Float64, length(I_spread) * sps_chip)
Q_impulse_train = zeros(Float64, length(Q_spread) * sps_chip)

I_impulse_train[1:sps_chip:end] = I_spread
Q_impulse_train[1:sps_chip:end] = Q_spread

# Convolve with RRC filter "pulse shaping". Now we get g(t) int the text
h_chip_len = length(h_rrc_chip)
delay = floor(Int, h_chip_len / 2)
# Filtering part: Apply the RRC, including the delay from convoluting.
conv_indices = delay + 1 : delay + length(I_impulse_train)
I_shaped = conv(I_impulse_train, h_rrc_chip)[conv_indices]
Q_shaped = conv(Q_impulse_train, h_rrc_chip)[conv_indices]

# If you want that DSSS frequency plot to look "stealthy" (lower than the original), just add this line before plotting:
I_shaped = I_shaped ./ sqrt(length(PN_code))
Q_shaped = Q_shaped ./ sqrt(length(PN_code))

N_spread = length(I_shaped);
t_spread = (0:N_spread-1) ./ fs;
I_carrier_dsss = cos.(2π * fc .* t_spread);
Q_carrier_dsss = sin.(2π * fc .* t_spread);

dsss_signal = I_shaped .* I_carrier_dsss .- Q_shaped .* Q_carrier_dsss;

t_dsss_zoom, sig_dsss_zoom = zoom_time(dsss_signal, fs, zoom_limit)

plot_clean(dsss_signal, I_shaped, Q_shaped, fs, zoom_limit)


savefig("signal_processing_project/Images/dsss_qpsk_time_zoom.png")

# =============================
# Frequency Hopping (FHSS)
# =============================

""" hop_rate: rate of hopping in Hz
    hop_freq: frequency deviation for hopping in Hz
"""
hop_rate=10.0
hop_freq = [400.0, 800.0, 600.0, 1000.0, 500.0, 1100.0]
signal_fhss = apply_frequency_hopping(I_shaped, Q_shaped, fs, hop_rate, hop_freq);

# Visualise the FHSS signal in time and frequency domain, computing the spectrogram
# t_fhss_zoom = t[1:N_zoom_clean]
# signal_fhss_zoom = signal_fhss[1:N_zoom_clean]

# p_fhss_time_zoom = plot_qpsk_time(t_fhss_zoom, signal_fhss_zoom, fs, fc)
# display(p_fhss_time_zoom)

samples_per_hop = round(Int, fs / hop_rate)

win_size = nextpow(2, samples_per_hop) 

overlap = div(win_size, 2) # 50% overlap

p_fhss = spectrogram(signal_fhss, win_size, overlap; fs=fs);

p_spectrogram = heatmap(p_fhss.time, p_fhss.freq, 10*log10.(p_fhss.power),
    title="FHSS Spectrogram",
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    color=:viridis)

# Save the plot
savefig(p_spectrogram, "signal_processing_project/Images/fhss_qpsk_spectrogram.png")


# =========================================
# Channel Effects. Adding fading and noise
# =========================================

# Apply Rayleigh fading to the FHSS signal

signal_faded, channel_gain = apply_rayleigh_fading(signal_fhss; doppler=true);

# Add AWGN noise to the faded signal
# SNR_dB = -5.0 # Signal-to-noise ratio in dB.
signal_final = apply_awgn(signal_faded; SNR_dB=-5.0);

# Plot the final received signal in time, frequency and constellation diagram

I_final = real(signal_final);
Q_final = imag(signal_final);

# Plotting the real part of the signal to visualise the fading and noise effects in time domain. 
p_final_time_zoom = plot_clean(real(signal_final), I_final, Q_final, fs, zoom_limit)
display(p_final_time_zoom)

savefig(p_final_time_zoom, "signal_processing_project/Images/final_qpsk_time_zoom.png")


