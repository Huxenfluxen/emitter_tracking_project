using HDF5
using Plots
using DSP

# Include the FastICA module
include("src/receiver/blind_detection.jl")
using .BlindDetection

# ==============================================================================
# 1. LOAD THE SENSOR DATA
# ==============================================================================
dataset_path = "data/sensor_recording.h5"
println("Loading sensor data from $dataset_path ...")

h5open(dataset_path, "r") do file
    global X_noisy = read(file, "X_noisy")
    global fs = read(file, "fs")
    global num_targets = read(file, "num_targets")
end

println("Loaded Sensor Matrix X of size: ", size(X_noisy))

# ==============================================================================
# 2. BLIND SOURCE SEPARATION (FastICA)
# ==============================================================================
println("\nInitiating FastICA...")
# In a real scenario, we might use AIC/MDL to estimate the number of targets.
# Here, we pass our known num_targets to extract the principal components.
S_est = run_fastica(X_noisy, num_targets)

println("Separation complete! Extracted signals matrix size: ", size(S_est))

# ==============================================================================
# 3. VISUALIZE: CHAOS VS. ORDER
# ==============================================================================
# Let's zoom in on the first 0.1 seconds to clearly see the waveforms
zoom_limit = 0.1  
num_samples = Int(zoom_limit * fs)
t = (0:num_samples-1) ./ fs

# What the antenna sees (A jumbled mess of 5 targets and thermal noise)
p_chaos = plot(t, X_noisy[1, 1:num_samples], 
    title="What Sensor 1 Sees (Chaos)", 
    xlabel="Time (s)", ylabel="Amplitude",
    label="Mixed Passband Signal", 
    color=:purple, alpha=0.7, grid=true)

# What FastICA extracted 
p_clean1 = plot(t, S_est[1, 1:num_samples], 
    title="FastICA Extracted Target A", 
    xlabel="Time (s)", ylabel="Amplitude",
    label="Independent Component 1", 
    color=:blue, grid=true)

p_clean2 = plot(t, S_est[2, 1:num_samples], 
    title="FastICA Extracted Target B", 
    xlabel="Time (s)", ylabel="Amplitude",
    label="Independent Component 2", 
    color=:red, grid=true)

# Combine the plots to show the before and after
final_plot = plot(p_chaos, p_clean1, p_clean2, layout=(3,1), size=(1000, 800), margin=5Plots.mm)
display(final_plot)