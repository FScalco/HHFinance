using Interpolations, LinearAlgebra, SparseArrays


function discretize_assets(amin, amax, n_a)
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = log(1.0 + log(1.0 + amax - amin))

    # make uniform grid
    u_grid = range(0.0, ubar, length = n_a)

    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin .+ exp.(exp.(u_grid) .- 1.0) .- 1.0
end


function rouwenhorst_Pi(N, p)
    # base case Pi_2
    Pi = [p 1-p;1-p p]

    # recursion to build up from Pi_2 to Pi_N
    for n in 3:N
        Pi_old = Pi
        Pi = zeros(n,n)

        Pi[1:(n-1),1:(n-1)] += p * Pi_old
        Pi[2:n,1:(n-1)] += (1-p) * Pi_old
        Pi[1:(n-1),2:n] += (1-p) * Pi_old
        Pi[2:n,2:n] += p * Pi_old
        Pi[2:(n-1),:] /= 2
    end

    return Pi
end


function stationary_markov(Pi, tol = 1e-14; initial_guess = nothing)
    # start with uniform distribution over all states
    n = size(Pi)[1]

    if isnothing(initial_guess)
        p_stat = fill(1/n,n)
    else
        p_stat = initial_guess
    end

    # update distribution using Pi until successive iterations differ by less than tol
    for i in 1:10_100
        p_stat_new = Pi' * p_stat
        if maximum(abs.(p_stat_new - p_stat)) < tol
            return p_stat_new
        end
        p_stat = p_stat_new
    end
end


function discretize_income(ρ,σ,n_s)
    # choose inner-switching probability p to match persistence rho
    p = (1 + ρ)/2

    # start with states from 0 to n_s-1, scale by alpha to match standard deviation sigma
    s = 1:n_s
    α = 2 * σ / sqrt(n_s - 1)
    s = α * s

    # obtain Markov transition matrix Pi and its stationary distribution
    Pi = rouwenhorst_Pi(n_s, p)
    p_stat = stationary_markov(Pi)

    # s is log income, get income y and scale so that mean is 1
    y = exp.(s)
    y /= sum(p_stat .* y)

    return y, p_stat, Pi
end


function backward_iteration(Va, Pi, a_grid, y, r, β, eis)
    # step 1: discounting and expectations
    Wa = β * Pi * Va

    # step 2: solving for asset policy using the first-order condition
    c_endog = Wa .^(-eis)
    coh = [i + j for i in y, j in (1+r) * a_grid] # Why calculating it here?

    a = similar(coh) # This is the saving function
    
    for s in 1:length(y)
        itp = LinearInterpolation(c_endog[s,:] + a_grid, a_grid; extrapolation_bc = Line())
        a[s,:] = itp.(coh[s,:])
    end

    # step 3: enforcing the borrowing constraint and backing out consumption
    a = max.(a,a_grid[1])
    c = coh - a

    # step 4: using the envelope condition to recover the derivative of the value function
    Va = (1+r) * c .^ (-1/eis)

    return Va, a, c
end



function policy_ss(Pi, a_grid, y , r, β, eis, tol = 1e-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = [i + j for i in y, j in (1+r) * a_grid]
    c = 0.05 * coh # we start from this guess of consumption being 5% of cash on hand, this guess doesn't really matter, it's just a start point for the iteration
    Va = (1+r) * c .^ (-1/eis)

    a_old = similar(c)

    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in 1:10_000
        Va, a, c = backward_iteration(Va, Pi, a_grid, y, r, β, eis)

        # after iteration 0, can compare new policy function to old one
        if it > 1
            if maximum(abs.(a - a_old)) < tol
                return Va, a, c
            end
        end

        a_old .= a
    end
end



function policy_ss_unconstrained(Pi, a_grid, y , r, β, eis, tol = 1e-9)
    # initial guess for Va: assume consumption 5% of cash-on-hand, then get Va from envelope condition
    coh = [i + j for i in y, j in (1+r) * a_grid]
    c = 0.05 * [i * ((1 + r) / (r)) + j for i in y, j in (1+r) * a_grid]
    Va = (1+r) * c .^ (-1/eis)

    a_old = similar(c)

    # iterate until maximum distance between two iterations falls below tol, fail-safe max of 10,000 iterations
    for it in 1:10_000
        Va, a, c = backward_iteration_unconstrained(Va, Pi, a_grid, y, r, β, eis)

        # after iteration 0, can compare new policy function to old one
        if it > 1
            if maximum(abs.(a - a_old)) < tol
                return Va, a, c
            end
        end

        a_old .= a
    end
end

function get_mpc(c, a, a_grid, r)
    mpcs = zeros(size(c))

    for s in 1:size(c)[1]
        mpcs[s,1] = (c[s,2] - c[s,1]) / (a_grid[2] - a_grid[1]) / (1 + r)

        for i in 2:(size(c)[2]-1)
            mpcs[s,i] = (c[s,i+1] - c[s,i-1]) / (a_grid[i+1] - a_grid[i-1]) / (1 + r)
        end

        mpcs[s,end] = (c[s,end] - c[s,end-1]) / (a_grid[end] - a_grid[end-1]) / (1 + r)
    end

    # when constrainted
    mpcs[a .== a_grid[1]] .= 1.0

    return mpcs
end

function get_lottery(a,a_grid)
    # step 1: find the i such that a' lies between gridpoints a_i and a_(i+1)
    a_i = max.(min.(searchsortedfirst.(Ref(a_grid), a),length(a_grid)),2) .- 1

    # step 2: implement (8) to obtain lottery probabilities pi
    a_pi = (a_grid[a_i .+ 1] - a) ./ (a_grid[a_i .+ 1] - a_grid[a_i])

    return a_i, a_pi
end

function SIM_transitional_matrix(a, a_grid, Pi_y)
    # get size
    ns, na = size(Pi_y)[1], length(a_grid)

    Pi = zeros(ns, na, ns, na)

    # get lottery
    a_next, a_next_p = get_lottery(a, a_grid)

    for s in 1:ns
        for s_next in 1:ns
            for a_i in 1:na
                a_next_i, a_next_i_p = a_next[s, a_i], a_next_p[s, a_i]

                Pi[s, a_i, s_next, a_next_i] += Pi_y[s, s_next] * a_next_i_p
                Pi[s, a_i, s_next, a_next_i + 1] += Pi_y[s, s_next] * (1 - a_next_i_p)
            end
        end
    end

    return Pi
end

function SIM_stationary_distribution(a, a_grid, Pi_y; tol = 1e-10)
    # get size
    ns, na = size(Pi_y)[1], length(a_grid)

    # build the transition matrix
    Pi = SIM_transitional_matrix(a, a_grid, Pi_y)

    # reshape to 2D
    Pi_2d = reshape(Pi, ns*na, ns*na)

    # get stationary distribution
    p_stat_vec = stationary_markov(sparse(Pi_2d), tol)

    # reshape back to 2D
    return reshape(p_stat_vec, ns, na)
end