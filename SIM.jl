using Plots

include("SIMBase.jl")


# set grids
ny = 7
σ_y = 0.7
ρ_y = 0.975

y, p_ss, Pi = discretize_income(ρ_y, σ_y, ny);

na = 500
a_grid = discretize_assets(0, 10_000, na);

r = 0.01/4
# β = 1 - 0.08/4
β = 0.99
eis = 1 # eis = 1/gamma, whatever that means

Va, a, c = policy_ss(Pi, a_grid, y, r, β, eis)

# Plot the consumption function
a_up_plot = 2

pl_consumption_fun = plot(a_grid, c[1,:], xlabel = "saving", ylabel = "consumption", ylims = (0.0, 1.0), xlims = (0.0, a_up_plot), legend = :topleft, label = "y = " * string(round(y[1], digits = 2)))
for s in [2,3,4]
    plot!(a_grid, c[s,:], label = "y = " * string(round(y[s], digits = 2)))
end

pl_consumption_fun

# calculate mpc
mpcs = get_mpc(c, a, a_grid, r);

pl_mpcs = plot(a_grid, mpcs[1,:], xlabel = "saving", ylabel = "MPC", ylims = (0.0, 1.0), xlims = (0.0, a_up_plot), legend = :topleft, label = "y = " * string(round(y[1], digits = 2)))

for s in [2,3,4,7]
    plot!(a_grid, mpcs[s,:], label = "y = " * string(round(y[s], digits = 2)))
end

pl_mpcs

# calculate stationary distribution
p_stat_SIM = SIM_stationary_distribution(a, a_grid, Pi)

sum(sum(p_stat_SIM, dims = 1)[1,:] .* a_grid)
sum(p_stat_SIM .* mpcs)
