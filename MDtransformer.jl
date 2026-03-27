using LinearAlgebra
using Random
using Statistics
using Flux
using Plots

# 1. Configuration
mutable struct QEDConfig
    grid_res::Int
    extent::Float64
    softening::Float64
    learning_rate::Float64
    eps::Float64
end

cfg = QEDConfig(400, 2.0, 1e-5, 1e-4, 1e-12)

# 2. Architecture - Transformer Field Block
function layer_norm(x, eps=cfg.eps)
    μ = mean(x)
    σ = std(x)
    return (x .- μ) ./ (σ .+ eps)
end

function transformer_field_block(x)
    # Self-attention-like operation
    attn = softmax(x) .* x
    x = layer_norm(x .+ attn)
    
    # Feed-forward with GELU activation
    ff = gelu(x)
    return layer_norm(x .+ ff)
end

# GELU activation if not available in Flux
function gelu(x)
    return x .* (0.5 .* (1.0 .+ tanh.(sqrt(2.0/π) .* (x .+ 0.044715 .* x.^3))))
end

# 3. Physics & Ground Truth
function get_ground_truth_psi(grid, Z=1)
    r = sqrt.(sum(grid.^2, dims=3) .+ cfg.softening)
    psi_gt = exp.(-Z .* r)
    norm_val = norm(vec(psi_gt)) + cfg.eps
    return psi_gt ./ norm_val
end

# Physical consistency loss
function physical_consistency_loss(psi_2d, V_field)
    psi_norm = psi_2d ./ (norm(vec(psi_2d)) + cfg.eps)
    
    # Compute gradients using finite differences
    grad_x = similar(psi_norm)
    grad_y = similar(psi_norm)
    
    grad_x[2:end, :] = diff(psi_norm, dims=1)
    grad_x[1, :] = grad_x[2, :]
    
    grad_y[:, 2:end] = diff(psi_norm, dims=2)
    grad_y[:, 1] = grad_y[:, 2]
    
    kinetic = mean((grad_x.^2 .+ grad_y.^2))
    potential = mean(psi_norm.^2 .* V_field)
    entropy = -0.05 * mean(log.(psi_norm.^2 .+ cfg.eps))
    
    return kinetic + potential + entropy
end

# 4. Training Step
function train_step!(state, i, grid, v_em, total_steps; mode="hybrid")
    psi, opt_m, opt_v, rng = state
    alpha = i / total_steps
    
    # Loss function with gradient computation
    function loss_fn(p)
        p_norm = p ./ (norm(p) + cfg.eps)
        p_2d = reshape(p_norm, cfg.grid_res, cfg.grid_res)
        
        l_phys = physical_consistency_loss(p_2d, v_em)
        gt = vec(get_ground_truth_psi(grid))
        l_gt = mean((p_norm .- gt).^2)
        
        return (1.0 - alpha) * l_phys + alpha * l_gt * 50.0
    end
    
    # Compute gradients using finite differences approximation
    eps_grad = 1e-5
    grad_est = similar(psi)
    for j in eachindex(psi)
        psi_plus = copy(psi)
        psi_plus[j] += eps_grad
        psi_minus = copy(psi)
        psi_minus[j] -= eps_grad
        grad_est[j] = (loss_fn(psi_plus) - loss_fn(psi_minus)) / (2 * eps_grad)
    end
    
    # Adam optimizer
    t = i + 1
    new_m = 0.9 .* opt_m .+ 0.1 .* grad_est
    new_v = 0.999 .* opt_v .+ 0.001 .* (grad_est.^2)
    
    m_hat = new_m ./ (1.0 - 0.9^t)
    v_hat = new_v ./ (1.0 - 0.999^t)
    
    psi_update = cfg.learning_rate .* (m_hat ./ (sqrt.(abs.(v_hat)) .+ cfg.eps))
    new_psi = transformer_field_block(psi .- psi_update)
    new_psi = new_psi ./ (norm(new_psi) + cfg.eps)
    
    loss_val = loss_fn(psi)
    
    return (new_psi, new_m, new_v, rng), loss_val
end

# 5. Execution
function run_simulation(nuclei_pos; steps=1000)
    res = cfg.grid_res
    lin = range(-cfg.extent, cfg.extent, length=res)
    
    # Create mesh grid
    X = repeat(lin', res, 1)
    Y = repeat(lin, 1, res)
    
    # Stack to create grid with shape (res, res, 2)
    grid = cat(X, Y, dims=3)
    
    # Compute potential from nuclei
    dist_sq = zeros(res, res)
    for k in axes(nuclei_pos, 1)
        dist_sq .+= (grid[:,:,1] .- nuclei_pos[k,1]).^2 .+ (grid[:,:,2] .- nuclei_pos[k,2]).^2
    end
    v_em = -5.0 .* exp.(-dist_sq ./ 0.01)
    
    # Initialize wavefunction
    psi_init = vec(exp.(-sum(grid.^2, dims=3)))
    psi_init = psi_init ./ norm(psi_init)
    
    # Initialize optimizer state
    state = (psi_init, zeros(length(psi_init)), zeros(length(psi_init)), Random.default_rng())
    
    println("Running Integrated Neural QED Simulation...")
    
    losses = Float64[]
    for i in 1:steps
        state, loss_val = train_step!(state, i, grid, v_em, steps)
        push!(losses, loss_val)
        
        if mod(i, 100) == 0
            println("Step $i / $steps, Loss: $loss_val")
        end
    end
    
    psi_final = state[1]
    
    # Visualization
    density = abs.(reshape(psi_final, res, res))
    density_normalized = density ./ (maximum(density) + 1e-12)
    
    plot(layout=(1,2), size=(1600, 600))
    
    # Subplot 1: Quantum Probability Density
    heatmap!(density_normalized.^0.5, title="Predicted Quantum Probability Density", 
             color=:magma, subplot=1)
    
    # Subplot 2: Energy Convergence
    plot!(losses, title="Energy Functional Convergence", color=:crimson, subplot=2)
    
    savefig("qed_simulation.png")
    display(plot!())
    
    return psi_final, losses
end

# Run the simulation
nuclei_pos = [0.3 0.0; -0.3 0.0]
psi_final, losses = run_simulation(nuclei_pos, steps=1000)