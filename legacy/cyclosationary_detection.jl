# Functions for Spectral Analysis and Cyclostationary Detection
using FFTW
using Statistics
using DSP

# ==== Cycloctationary Detection Functions ====

# Spectral Correlation density (SCD) Direct method
# Direct method to compute SCD

function spectral_correlation_density(x, fs)
    N = length(x)
    Rx_matrix = zeros(ComplexF64, N, N)

    for τ in 0:(N-1)
        x_shifted = circshift(x, τ)
        Rx_matrix[:, τ+1] = x .* conj.(x_shifted)
    end

    cyclic_ACF = fft(Rx_matrix, 1)

    SCD = fft(cyclic_ACF, 2)

    SCD_shifted = fftshift(SCD)

    # Cyclic freq axis
    axis_α = fftfreq(N, fs) |> fftshift
    # Spectral freq axis
    axis_f = fftfreq(N, fs) |> fftshift

    return axis_f, axis_α, abs.(SCD_shifted)
end