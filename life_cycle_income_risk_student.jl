using Plots

include("SIMBase.jl")

## parameters

# age
K = 65
intial_age = 25

# income process
σ_y = 0.8  # sd of log income
σ_z = 0.65 # sd of persistent income risk
σ_age = sqrt(σ_y^2 - σ_z^2) # sd of age component, set to match total income risk
ρ_z = 0.975

# life cycle income profile
# we use the bell-curve to capture the hump-shaped life cycle
peak_income_age = 47.5
income_profile_width = 15.0

# grid size
nz = 7
na = 500

# bequest parameters
κ = 5.0
γ = 0.8

# other parameters
r = 0.01
β = 0.98
eis = 1.0


## Part 1: Discretization

# Age
k_grid = collect(1:K)
age = k_grid .+ (intial_age - 1)

# Grid for persistent income risk
# TODO: Fill up the arguments
z_grid, p_stat_z, Pi_z = discretize_income(XXX, XXX, XXX)

# life cycle income risk
logy_age = @. - ((age - peak_income_age) / income_profile_width)^2 / 2

# rescale the logy_age to match σ_age
# TODO: calculate the SD of the age component of log(y)
logy_age_sd = XXX
logy_age = logy_age / logy_age_sd * σ_age

y_age = @. exp(logy_age)

# rescale the age component to ensure that the total income is one
p_stat_y = fill(1/K, K) * p_stat_z'
y_grid = y_age * z_grid'

y_grid = y_grid ./ sum(p_stat_y .* y_grid)

# asset grid
na = 100
a_grid = discretize_assets(0.0, 10_000, na)

## Part 2: Backward induction

# pre-allocate memory for value function
Va = zeros(K, nz, na)
c = zeros(K, nz, na)
a = zeros(K, nz, na)

# TODO: Calculate the marginal value for the bequest
Va_final = XXXX

for k in K:-1:1
    # get the last period value function
    if k == K
        Va_last = Va_final
    else
        Va_last = Va[k+1,:,:]
    end 

    # backward induction
    # TODO: Apply the backward_induction() function here to compute the value a period before
    Va[k,:,:], a[k,:,:], c[k,:,:] = XXXX
end

## Part 3: Stationary Distribution
Pi = zeros(K, nz, na, K, nz, na);

for k_i in 1:(K-1)
    for z_i in 1:nz
        for a_i in 1:na
            for z_i_next in 1:nz
                a_next_i, a_next_i_p = get_lottery(a[k_i, z_i, a_i], a_grid)

                Pi[k_i, z_i, a_i, k_i + 1, z_i_next, a_next_i] += Pi_z[z_i, z_i_next] * a_next_i_p
                Pi[k_i, z_i, a_i, k_i + 1, z_i_next, a_next_i + 1] += Pi_z[z_i, z_i_next] * (1 - a_next_i_p)
            end
        end
    end
end

# Agents at age K dies and newborn agents arrive with zero wealth
# We can think of it as old agents "reincarnate" as agent with age 0 and zero wealth
#
# TODO: complete the transition matrix for age K
for z_i in 1:nz
    for a_i in 1:na
        for z_i_next in 1:nz
            Pi[K, z_i, a_i, 1, z_i_next, 1] += XXX
        end
    end
end

# TODO: Reshape the Pi into a 2-dimensional object for stationary distribution
Pi_2d = reshape(Pi, XXX, XXX)
Pi_2d = sparse(Pi_2d)

# initial distribution for life-cycle model
#
# Usually, an initial distribution is not needed. But because of the life-cycle, the Pi doesn't guarantee a stationary distribution.
# The p_initial ensures that every age has the same weight.
p_initial = reshape([(1 / K) * (1 / na) * p_stat_z[z_i] for k_i in 1:K, z_i in 1:nz, a_i in 1:na], K * nz * na)

p_stat = stationary_markov(Pi_2d, 1e-14, initial_guess = p_initial)

# TODO: Reshape the p_stat into (K, nz, na) dimension
p_stat = reshape(p_stat, K, nz, na)


## Part 4: Plots and your own calculations
