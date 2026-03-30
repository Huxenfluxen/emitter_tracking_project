using HDF5

# 1. Load the recorded sensor data
h5open("data/sensor_recording.h5", "r") do file
    global X_noisy = read(file, "X_noisy")
    global fs = read(file, "fs")
end

println("Loaded sensor data of size: ", size(X_noisy))