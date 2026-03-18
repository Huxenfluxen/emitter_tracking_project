# Plotting functions for the signal processing project
using Plots
using Measures
using Plots.Measures
using Random
using FFTW
using Statistics
using DSP


# ==========================================================================================
# Plot clean VS Aliased QPSK signal in time and frequency domain and constellation diagram
# ==========================================================================================

function zoom_time(signal, fs, zoom_limit)

    # Number of samples to show for each rate
    N_zoom = round(Int, zoom_limit * fs)
    N_zoom = min(N_zoom, length(signal)) # Ensure the zoom is not larger than the signal length

    # Slice the signals first
    sig_zoom = signal[1:N_zoom]

    # Create matching time vectors for the zoom
    t_zoom = (0:N_zoom-1) ./ fs;
    return t_zoom, sig_zoom
end

function get_spectrum(signal, fs)
    F = fft(signal) |> fftshift;
    freqs = fftfreq(length(signal), fs) |> fftshift;
    magnitude = abs.(F) ./ length(signal);
    return freqs, magnitude
end

function plot_time(t_zoom, signal_zoom)
    p = plot(t_zoom, signal_zoom,
        label="QPSK Signal",
        xlabel="Time (s)",
        ylabel="Amplitude",
        title="Time Domain: QPSK Signal",
        grid=true,
        legend=:outertop)
    return p
end

function plot_spectrum(signal, fs)
    freqs, magnitude = get_spectrum(signal, fs)
    p = plot(freqs, magnitude,
        title="Frequency Spectrum: QPSK signal",
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        label="QPSK Signal Spectrum",
        xlims=(-fs/4, fs/4),
        grid=true,
        legend=:outertop)
    return p
end

function plot_const(I_signal, Q_signal; const_lims=(-1.5, 1.5))
    p = scatter(I_signal, Q_signal,
        title="Constellation Diagram: QPSK Signal",
        xlabel="In-Phase (I)",
        ylabel="Quadrature (Q)",
        label="QPSK Signal",
        xlims=const_lims,
        ylims=const_lims,
        aspect_ratio=:equal,
        # markerstrokewidth=0,
        alpha=0.5,
        grid=true)
    return p
end

function plot_clean_vs_aliased(signal, signal_aliased, fs, fs_aliased, I_clean, Q_clean, I_aliased, Q_aliased, zoom_limit)

    t_zoom, sig_zoom = zoom_time(signal, fs, zoom_limit)
    t_aliased_zoom, sig_aliased_zoom = zoom_time(signal_aliased, fs_aliased, zoom_limit)


    # Plot the clean and aliased signals in time domain to visualise the aliasing effect
    p_time = plot_time(t_zoom, sig_zoom)
    plot!(p_time, t_aliased_zoom, sig_aliased_zoom,
        title="Clean QPSK Signal vs Aliased Version",
        label="Aliased QPSK Signal",
        color=:red, linewidth=1)

    # Add a plot in frequency domain to visualise the aliasing effect "folding"
    p_freq = plot_spectrum(signal, fs)
    freqs_aliased, magnitude_aliased = get_spectrum(signal_aliased, fs_aliased)
    plot!(p_freq, freqs_aliased, magnitude_aliased,
        title="Frequency Spectrum: Clean vs Aliased QPSK Signal",
        label="Aliased QPSK Signal Spectrum",
        color=:red, linewidth=0.5)

    # Add constellation diagrams for both signals
    p_const = plot_const(I_clean, Q_clean)
    scatter!(p_const, I_aliased, Q_aliased,
        title="Constellation Diagram: Clean vs Aliased QPSK Signal",
        label="Aliased QPSK Signal")


    # Combine time, frequency and constellation plots
    l = @layout [[a; b] c]
    final_plot = plot(p_time, p_freq, p_const,
        #title = "Clean QPSK Signal (fs = $fs Hz >> 2fc = $(2*fc) Hz) vs Aliased Version (fs = $(fs/downsample_factor) Hz << 2fc = $(2*fc) Hz)",
        layout = l, 
        size = (1300, 600), 
        margin = 5mm,           # Global margin for all sides
        bottom_margin = 15mm,   # Extra space for the cut-off frequency labels
        left_margin = 10mm)
    return final_plot
end

function plot_clean(signal, I_signal, Q_signal, fs, zoom_limit; const_lims=(-1.5, 1.5))

    t_zoom, sig_zoom = zoom_time(signal, fs, zoom_limit)

    # Plot the clean signal in time domain
    p_time = plot_time(t_zoom, sig_zoom)
    p_freq = plot_spectrum(signal, fs)
    p_const = plot_const(I_signal, Q_signal; const_lims=const_lims)
    # Combine time, frequency and constellation plots
    l = @layout [[a; b] c]
    final_plot = plot(p_time, p_freq, p_const,
        layout = l, 
        size = (1300, 600), 
        margin = 5mm,           # Global margin for all sides
        bottom_margin = 15mm,   # Extra space for the cut-off frequency labels
        left_margin = 10mm)
    return final_plot
end

# function add_lens(p, zoom_x_start, zoom_x_end, zoom_y_span)
#     lens!(p, [zoom_x_start, zoom_x_end], [zoom_y_span[1], zoom_y_span[2]], 
#           inset = (1, bbox(0.5, 0.5, 0.4, 0.4)), # Place inset in center-right
#           color =:red,
#           linewidth=2,
#           subplot=2)
# end
