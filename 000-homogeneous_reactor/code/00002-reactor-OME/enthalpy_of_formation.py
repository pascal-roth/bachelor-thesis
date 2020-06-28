#######################################################################################################################
# Calculate the standard enthalpies of formation for the species of the PODE mechanism
#######################################################################################################################

#import packages
import cantera as ct
import numpy as np
import pandas as pd
from pathlib import Path

ct.suppress_thermo_warnings()

pode = ct.Solution('cai_ome14_2019.xml')
species = pode.species_names
enthalpies = np.zeros((len(species), 2))

for i in range(len(species)):
    pode.TPY = 298.15, ct.one_atm, '{}:1.0'.format(species[i])
    enthalpies[i, 0] = pode.enthalpy_mole
    enthalpies[i, 1] = pode.enthalpy_mass


path = Path(__file__).resolve()
path_h0 = path.parents[2] / 'data/00002-reactor-OME/enthalpies_of_formation.csv'

enthalpies = pd.DataFrame(enthalpies)
enthalpies.columns = ['h0_mole', 'h0_mass']
enthalpies.to_csv(path_h0)