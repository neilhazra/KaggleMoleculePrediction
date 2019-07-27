import pandas as pd
import pickle as pk

data_path_prefix = "../data/"
molecule_prop = {}

dp_moments = pd.read_csv(data_path_prefix + "dipole_moments.csv")
potential_energy = pd.read_csv(data_path_prefix + "potential_energy.csv")
df_molecule_prop = dp_moments.merge(potential_energy)
print(dp_moments.head())
s = set(dp_moments['molecule_name'].unique())
for molecule in s:
    molecule_prop[molecule] = df_molecule_prop.loc[df_molecule_prop['molecule_name']==molecule]
with open(data_path_prefix + "molecule_prop_set.pkl",'wb') as f:
    pk.dump(s,f,pk.HIGHEST_PROTOCOL)
with open(data_path_prefix + "molecule_prop.pkl",'wb') as f:
    pk.dump(molecule_prop,f,pk.HIGHEST_PROTOCOL)

