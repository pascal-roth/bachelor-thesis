### Implementation constant pressure, fixed mass reactor ###

# Information source
# Cantera Website
# - https://cantera.org/science/reactors.html
# - https://cantera.github.io/docs/sphinx/html/cython/thermo.html
# - https://cantera.org/documentation/docs-2.4/sphinx/html/cython/zerodim.html
# - https://cantera.org/examples/python/reactors/reactor2.py.html (Example of two reactors with a piston and with heat
# loss to the environment

# Source for ignition delay code
# https://cantera.org/examples/jupyter/reactors/batch_reactor_ignition_delay_NTC.ipynb.html

# Import Packages
import cantera as ct
import pandas as pd
import time

# %% Surrounding gas/ air for POMEn mixture
gas = ct.Solution('gri30.xml')
air = gas

reactorTemperature = [1800, 1600, 1400, 1200, 1000, 950, 925, 900, 850, 825, 800,
     750, 700, 675, 650, 625, 600, 550, 500]
reactorPressure = ct.one_atm

estimatedIgnitionDelayTime = 1.0
ignitionDelays = pd.DataFrame(data={'T': reactorTemperature})

def ignitionDelay(df, species):
    return df[species].idxmax()

for i in range(0, len(reactorTemperature), 1):
    gas.TP = reactorTemperature[i], reactorPressure

    # equivalent ratio can be set, stoichiometry ratio is equal to 1.0
    gas.set_equivalence_ratio(0.75, 'CH4', 'O2:0.21, N2:0.79')

    env = ct.Reservoir(contents=air)

    r1 = ct.ConstPressureReactor(contents=gas, energy='on', name='homogeneous_reactor')

    A_w1 = 1.0              # Area of piston
    K_w1 = 0.5e-4           # Wall expansion rate parameter [m/s/Pa]
    U_w1 = 500.0            # Overall heat transfer coefficient [W/m^2]

    w1 = ct.Wall(r1, env, A=A_w1, K=K_w1, U=U_w1)

    sim = ct.ReactorNet([r1])

    timeHistory = pd.DataFrame(columns=[r1.component_name(item) for item in range(r1.n_vars)])

    time0 = time.time()
    t = 0
    counter = 0
    while t < estimatedIgnitionDelayTime:
        t = sim.step()
        if not counter % 20:
            timeHistory.loc[t] = r1.get_state()
        counter += 1

    tau = ignitionDelay(timeHistory, 'CH4')
    time1 = time.time()

    print('Computed Ignition Delay: {:.3e} seconds for T={}K. Took {:3.2f}s to compute'.format(tau, reactorTemperature[i], time1 - time0))

    ignitionDelays.at[i, 'ignDelay'] = tau

# %% Plot results
plot_ign = True

if plot_ign is True:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(1000 / ignitionDelays['T'], ignitionDelays['ignDelay'], 'o-')
    ax.set_ylabel('Ignition Delay (s)')
    ax.set_xlabel(r'$\frac{1000}{T (K)}$', fontsize=18)

    # Add a second axis on top to plot the temperature for better readability
    ax2 = ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xticklabels((1000 / ticks).round(1))
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'Temperature: $T(K)$');

    plt.show()