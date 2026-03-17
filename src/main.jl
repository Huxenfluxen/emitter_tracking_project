using Plots
using Measures
using Random

# import the generating Functions
include("signal_generation/modulators.jl")
using .SignalGeneration

# Import plotting functions
include("plotting_sp.jl")

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

target1 = SignalGeneration.TargetEmitter(
    id = 1, location = (0.0, 0.0), velocity = (15.0, 0.0),
    fc = 511.0, fs = 4400.0, sps = 8, β = 0.35, span = 10,
    snr_db = -5.0, hop_rate = 10.0, lfsr_bits = 7,
    hop_freq = [400.0, 800.0, 600.0, 1000.0, 500.0, 1100.0]
)


# ==========================
# 2. GENERATE CLEAN SIGNALS
# ==========================
println("Running signal pipeline...")
signals = SignalGeneration.generate_signal(target1, num_symbols)

# ===========================
# 2. CLEAN VS ALIASED SIGNAL
# ===========================
# Generate aliased version of the clean signal by downsampling
downsample_factor = 25
# Aliasing. Compare a high rate sampled QPSK and an aliased version (downsampled)
qpsk_signal_aliased = qpsk_signal_clean[1:downsample_factor:end];

fs_aliased = target1.fs / downsample_factor
sps_aliased = Int(target1.sps / downsample_factor)

# RRC filter for aliased signal
h_rrc_aliased = SignalGeneration.create_rrc_filter(sps_aliased, β, span);

# Demodulate them
I_clean, Q_clean = demodulate_signal(qpsk_signal_clean, sps, fc, fs, h_rrc);
I_aliased, Q_aliased = demodulate_signal(qpsk_signal_aliased, sps_aliased, fc, fs_aliased, h_rrc_aliased);

# Plottingclean and aliased signal
zoom_limit = 0.5 # seconds

clean_vs_aliased_plot = plot_clean_vs_aliased(
    qpsk_signal_clean, qpsk_signal_aliased, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, zoom_limit)

# clean_vs_aliased_plot = plot_qpsk_clean_vs_aliased(t_clean_zoom, sig_clean_zoom, t_aliased_zoom, sig_aliased_zoom, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, fc)
display(clean_vs_aliased_plot)
# Save the plot
savefig(clean_vs_aliased_plot, "signal_processing_project/Images/qpsk_clean_vs_aliased.png")


# ========================
# 3. PLOT DSSS (Baseband)
# ========================
# Access the intermediate DSSS signal using signals.dsss
dsss_plot = plot_clean(real.(signals.dsss), real.(signals.dsss), imag.(signals.dsss), target1.fs, 0.5)
savefig(dsss_plot, "data/dsss_qpsk_time_zoom.png")

# ===========================
# 4. PLOT FHSS (Spectrogram)
# ===========================
samples_per_hop = round(Int, target1.fs / target1.hop_rate)
win_size = nextpow(2, samples_per_hop)
p_fhss = spectrogram(real.(signals.fhss), win_size, div(win_size, 2); fs=target1.fs)

p_spectrogram = heatmap(p_fhss.time, p_fhss.freq, 10*log10.(p_fhss.power),
    title="FHSS Spectrogram", color=:viridis)
savefig(p_spectrogram, "data/fhss_qpsk_spectrogram.png")

# =====================================
# 5. PLOT FINAL CHANNEL (Faded + AWGN)
# =====================================
final_plot = plot_clean(real.(signals.final), real.(signals.final), imag.(signals.final), target1.fs, 0.5)
savefig(final_plot, "data/final_qpsk_time_zoom.png")