import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# ----------------------
# Model & fixed params
# ----------------------
# Genes: S, B, M (state vector x = [S, B, M])
# Signals: F, W
alpha = {"S": 2.0, "B": 2.0, "M": 2.0}   # production max
delta = {"S": 2.0, "B": 2.0, "M": 2.0}   # degradation
Kpol = {"S": 1.0, "B": 1.0, "M": 1.0}
xPol = 1.0
n_sites = 2  # number of binding sites (applied as exponent)
# Cooperativity factors
c_act = 10.0
c_rep = 0.1

# Fixed signal binding affinities (Wnt and FGF)
K_F_on_B = 10.0
K_W_on_B = 10.0
K_F_on_S = 10.0

# time vector for evaluation (hours)
t_eval_points = np.linspace(0, 48, 241)  # dense for plotting if wanted

# experimental target data 
targets = {
    "neural": {
        0.0: np.array([0.8, 0.0, 0.0]),   # S,B,M
        24.0: np.array([0.5, 1.0, 0.0]),
        48.0: np.array([1.0, 0.0, 0.0])
    },
    "mesoderm": {
        0.0: np.array([0.8, 0.0, 0.0]),
        24.0: np.array([0.5, 1.0, 0.0]),
        48.0: np.array([0.0, 0.3, 1.0])
    }
}

# initial concentrations at t=0 (given)
def initial_state():
    return np.array([0.8, 0.001, 0.001])  # S, B, M (user wanted B,M,F,W = 0.001 but F,W are external)

# external baseline signals (base concentrations)
base_F = 0.001
base_W = 0.001

# time-varying external conditions (piecewise) for each experiment
def external_signals(t, condition):
    # returns effective (F, W) concentrations at time t for given condition
    if condition == "neural":
        # 0-24: x_ext_F =1, x_ext_W =0 ; 24-48: both 0
        if t <= 24.0:
            return base_F + 1.0, base_W + 0.0
        else:
            return base_F + 0.0, base_W + 0.0
    elif condition == "mesoderm":
        # 0-24: x_ext_F =1, x_ext_W =0 ; 24-48: x_ext_F =0, x_ext_W =1
        if t <= 24.0:
            return base_F + 1.0, base_W + 0.0
        else:
            return base_F + 0.0, base_W + 1.0
    else:
        raise ValueError("Unknown condition")

# time-varying effective multiplier for S binding affinity (applied to both K_SB and K_SM)
def S_binding_multiplier(t, condition):
    # user-specified multipliers:
    # neural: 0-24 -> 2.75*h ; 24-48 -> 4*h
    # mesoderm: 0-24 -> 2.75*h ; 24-48 -> 1*h
    if t <= 24.0:
        return 2.75
    else:
        if condition == "neural":
            return 4.0
        else:
            return 1.0

# ----------------------
# Topology: activators / repressors for each gene
# ----------------------
# Using the topology you gave:
# S: activated by F ; repressed by B and M
# B: activated by F and W ; repressed by S and M
# M: activated by B ; repressed by S

# We'll map binding constants in a structured way. The optimiser will return:
# K_MB, K_SB, K_BS, K_MS, K_BM, K_SM  (six values)
# We'll interpret them as:
#  - K_MB : M -> B  (M acting on B)
#  - K_SB : S -> B
#  - K_BS : B -> S
#  - K_MS : M -> S
#  - K_BM : B -> M
#  - K_SM : S -> M

# Build helper to fetch K_T_on_X with S multiplier applied when T == 'S'
def build_K_dict(params, condition):
    K_MB, K_SB, K_BS, K_MS, K_BM, K_SM = params
    # baseline mapping
    K = {
        ('M','B'): K_MB,
        ('S','B'): K_SB,
        ('B','S'): K_BS,
        ('M','S'): K_MS,
        ('B','M'): K_BM,
        ('S','M'): K_SM
    }
    # when using T=='S', apply time-dependent multiplier inside ODE (we will pass multiplier in ODE)
    return K

# ----------------------
# Thermodynamic rule: compute p_bound for each gene
# ----------------------
def p_bound_gene(gene, x_state, K_map, t, condition):
    # x_state is vector [S,B,M]; external signals F,W depend on t and condition
    S_val, B_val, M_val = x_state
    F_val, W_val = external_signals(t, condition)

    # helper to get concentration of input T
    conc = {
        'S': S_val,
        'B': B_val,
        'M': M_val,
        'F': F_val,
        'W': W_val
    }

    # set of activators & repressors for each gene (topology)
    if gene == 'S':
        activators = ['F']
        repressors = ['B','M']
    elif gene == 'B':
        activators = ['F','W']
        repressors = ['S','M']
    elif gene == 'M':
        activators = ['B']
        repressors = ['S']
    else:
        raise ValueError("Unknown gene")

    # For each input T we need K_T_on_gene. For signals and fixed inputs:
    # - K_F_on_B, K_W_on_B, K_F_on_S are fixed (10). We will treat K_F_on_S = 10 if needed.
    # Build Z_bound and Z_unbound factorised forms from your equations:
    # Z_bound = Kpol * xPol * product_{T in Act} (1 + K_TX * c_T * x_T)^{n_sites}
    # Z_unbound = product_{T in Act ∪ Rep} (1 + K_TX * x_T)^{n_sites}
    # Note: if a regulator is missing from K_map and is an external signal, use the fixed K values.

    # Helper to get binding K for input T acting on gene
    def get_K(T, gene):
        if T == 'F':
            if gene == 'B':
                return K_F_on_B
            elif gene == 'S':
                return K_F_on_S
            else:
                return 0.0
        if T == 'W':
            if gene == 'B':
                return K_W_on_B
            else:
                return 0.0
        # for protein regulators use K_map
        return K_map.get((T, gene), 0.0)

    # Build Z_bound (active states = those containing Pol and no repressors)
    # According to the factorised form: Z_bound = Kpol * xPol * product_{T in Act} (1 + K*c*x)^{n_sites}
    prod_bound = 1.0
    for T in activators:
        Kt = get_K(T, gene)
        # apply special multiplier if T == 'S'
        if T == 'S':
            Kt = Kt * S_binding_multiplier(t, condition)
        prod_bound *= (1.0 + Kt * c_act * conc[T]) ** n_sites

    Z_bound = Kpol[gene] * xPol * prod_bound

    # Build Z_unbound = product_{T in Act ∪ Rep} (1 + K_TX * x_T)^{n_sites}
    prod_unbound = 1.0
    for T in activators + repressors:
        Kt = get_K(T, gene)
        if T == 'S':
            Kt = Kt * S_binding_multiplier(t, condition)
        prod_unbound *= (1.0 + Kt * conc[T]) ** n_sites

    Z_unbound = prod_unbound

    p = Z_bound / (Z_bound + Z_unbound)
    return float(p)

# ----------------------
# ODE system
# ----------------------
def ode_system(t, y, K_map, condition):
    # y = [S, B, M]
    S_val, B_val, M_val = y
    x_state = np.array([S_val, B_val, M_val], dtype=float)

    pS = p_bound_gene('S', x_state, K_map, t, condition)
    pB = p_bound_gene('B', x_state, K_map, t, condition)
    pM = p_bound_gene('M', x_state, K_map, t, condition)

    dS = alpha['S'] * pS - delta['S'] * S_val
    dB = alpha['B'] * pB - delta['B'] * B_val
    dM = alpha['M'] * pM - delta['M'] * M_val

    return [dS, dB, dM]

# ----------------------
# Simulator util: returns concentrations at requested times (including 0,24,48)
# ----------------------
def simulate(params, condition, t_points=(0.0, 24.0, 48.0)):
    # params: vector of the six baseline K's
    K_map = build_K_dict(params, condition)
    y0 = initial_state()

    sol = solve_ivp(fun=lambda t, y: ode_system(t, y, K_map, condition),
                    t_span=(0.0, 48.0), y0=y0, t_eval=sorted(set(t_eval_points).union(t_points)),
                    method='RK45', atol=1e-8, rtol=1e-6)
    # extract values at desired t_points
    out = {}
    for tp in t_points:
        idx = np.argmin(np.abs(sol.t - tp))
        out[tp] = sol.y[:, idx].copy()  # S,B,M
    return out, sol

# ----------------------
# Distance function used for optimisation
# ----------------------
def score_params(params, verbose=False):
    # params is array-like of six K's in the order:
    # [K_MB, K_SB, K_BS, K_MS, K_BM, K_SM]
    # enforce parameter bounds softly (they should be within DE bounds anyway)
    params = np.asarray(params)
    if np.any(params <= 0):
        return 1e6 + np.sum(np.abs(params))  # large penalty

    total_ssq = 0.0

    # Simulate neural condition
    sim_neural, _ = simulate(params, 'neural', t_points=(0.0,24.0,48.0))
    # Simulate mesoderm condition
    sim_meso, _ = simulate(params, 'mesoderm', t_points=(0.0,24.0,48.0))

    # sum squared differences at each timepoint and each gene
    for cond_name, sim in (('neural', sim_neural), ('mesoderm', sim_meso)):
        for tp, target in targets[cond_name].items():
            pred = sim[tp]
            # normalise each gene difference by 1.0 (values already in [0,1]); you can change weights here
            ssq = np.sum((pred - target) ** 2)
            total_ssq += ssq

    # small regularisation term preventing extreme K values (optional)
    # penalise params near edges of allowed range [0.01, 1000]
    lower, upper = 0.01, 1000.0
    reg = 0.0
    for p in params:
        if p < lower:
            reg += (lower - p) ** 2 * 1e3
        elif p > upper:
            reg += (p - upper) ** 2 * 1e3
    total_ssq += reg * 1e-3

    if verbose:
        print("params:", params, "score:", total_ssq)
    return float(total_ssq)

# ----------------------
# Optimisation call
# ----------------------
def fit_parameters(maxiter=60, popsize=15, seed=1):
    # bounds for the six K parameters (per your rule)
    bounds = [(0.01, 1000.0)] * 6

    result = differential_evolution(score_params, bounds,
                                    strategy='best1bin',
                                    maxiter=maxiter,
                                    popsize=popsize,
                                    tol=1e-6,
                                    polish=True,
                                    seed=seed,
                                    updating='deferred',
                                    workers=1)  # set workers>1 if you want parallel evaluation
    return result

# ----------------------
# Run optimisation and plot result
# ----------------------
if __name__ == "__main__":
    print("Starting optimisation (this can take minutes depending on popsize/maxiter)...")
    res = fit_parameters(maxiter=40, popsize=12, seed=42)
    print("Optimization finished.")
    print("Best score:", res.fun)
    print("Best params (K_MB, K_SB, K_BS, K_MS, K_BM, K_SM):")
    print(res.x)

    # simulate best solution for both conditions and plot
    best_params = res.x
    sim_neural, sol_neural = simulate(best_params, 'neural', t_points=(0.0,24.0,48.0))
    sim_meso, sol_meso = simulate(best_params, 'mesoderm', t_points=(0.0,24.0,48.0))

    # Plot trajectories for visual inspection
    fig, axs = plt.subplots(1,2, figsize=(12,4), sharey=True)
    for sol, cond_name, ax in [(sol_neural,'neural',axs[0]), (sol_meso,'mesoderm',axs[1])]:
        ax.plot(sol.t, sol.y[0,:], label='S')
        ax.plot(sol.t, sol.y[1,:], label='B')
        ax.plot(sol.t, sol.y[2,:], label='M')
        # overlay target points
        for tp, val in targets[cond_name].items():
            ax.scatter([tp],[val[0]], marker='o', color='C0')
            ax.scatter([tp],[val[1]], marker='s', color='C1')
            ax.scatter([tp],[val[2]], marker='^', color='C2')
        ax.set_title(cond_name)
        ax.set_xlabel('time (h)')
        ax.set_ylim(-0.05,1.05)
        ax.legend()
    axs[0].set_ylabel('relative conc')
    plt.tight_layout()
    plt.show()
