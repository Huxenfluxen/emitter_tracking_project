module BlindDetection

using LinearAlgebra
using Statistics
using HDF5

export run_fastica

# 1. Contrast function (g) and its derivative (g_prime)
# log-cosh is robust for sub-Gaussian signals like QPSK/DSSS
g(u) = tanh.(u)
g_prime(u) = 1.0 .- tanh.(u).^2

function run_fastica(X::Matrix{Float64}, num_components::Int; max_iter=1000, tol=1e-6)
    M, T = size(X)
    
    # STEP 1: Centering
    println("Centering data...")
    mean_X = mean(X, dims=2)
    X_centered = X .- mean_X
    
    # STEP 2: Whitening (PCA)
    println("Whitening data...")
    covariance_matrix = (X_centered * X_centered') / T
    
    # Eigenvalue decomposition
    eig_vals, eig_vecs = eigen(covariance_matrix)
    
    # Sort eigenvalues descending to keep the most important variance (PCA)
    idx = sortperm(eig_vals, rev=true)
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    # Keep only the top 'num_components' (This acts as physical noise reduction!)
    V = eig_vecs[:, 1:num_components]
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(eig_vals[1:num_components]))
    
    # The whitening matrix
    W_whiten = D_inv_sqrt * V'
    Z = W_whiten * X_centered  # Z is the whitened data
    
    # STEP 3: FastICA Iteration (Symmetric Orthogonalization)
    println("Running FastICA Iterations...")
    C = num_components
    W = randn(C, C) # Initial random unmixing matrix
    
    # Symmetric orthogonalization for the initial guess
    W = (W * W')^(-0.5) * W 
    
    for i in 1:max_iter
        # W * Z projects the whitened data onto our current guess
        Y = W * Z
        
        # Calculate g(Y) and g'(Y)
        g_Y = g(Y)
        gp_Y = g_prime(Y)
        
        # The FastICA update rule
        # E[Z * g(W^T Z)] - E[g'(W^T Z)] * W
        term1 = (g_Y * Z') / T
        term2 = Diagonal(vec(mean(gp_Y, dims=2))) * W
        
        W_new = term1 - term2
        
        # Symmetric orthogonalization (prevents W from finding the same component)
        W_new = (W_new * W_new')^(-0.5) * W_new
        
        # Check convergence (if the vectors stopped moving)
        delta = maximum(abs.(abs.(diag(W_new * W')) .- 1.0))
        W = W_new
        
        if delta < tol
            println("FastICA converged after $i iterations.")
            break
        end
        
        if i == max_iter
            println("Warning: FastICA reached max_iter without fully converging.")
        end
    end
    
    # The final unmixing matrix applied to the raw data!
    # S_est = W * W_whiten * X_centered
    S_estimated = W * Z
    
    return S_estimated
end

end # module