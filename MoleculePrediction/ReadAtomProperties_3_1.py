import pandas as pd
import pickle as pk
from mendeleev import get_table
from rdkit import Chem
pt = Chem.GetPeriodicTable()
ptable = get_table('elements')
import os
ptable_access = {}
for _,row in ptable.iterrows():
    ptable_access[row["symbol"]] = row["atomic_number"]

data_path_prefix = "../data_3_0/"
path_to_structures = data_path_prefix + "/structures/"

molecule_structures = {}

print("Processing Structures and atom properties")
for i,molecule in enumerate(os.listdir(path_to_structures)):
    df = pd.read_csv(path_to_structures + molecule, sep = ' ', skiprows= [0],names = ["atom","x","y","z"])
    for a_index in range(len(df)):
        atom = df["atom"].iloc[a_index]
        df.at[a_index, "electron_affinity"] = ptable.at[ptable_access[atom]-1,"electron_affinity"]
        df.at[a_index, "electronegativity"] = ptable.at[ptable_access[atom]-1,"en_pauling"]
        df.at[a_index, "num_protons"] = ptable_access[atom]
        df.at[a_index, "dipole_polarizability"] = ptable.at[ptable_access[atom]-1,"dipole_polarizability"]
        df.at[a_index, "covalent_radius"] = pt.GetRcovalent(ptable_access[atom])
        df.at[a_index, "c6"] = ptable.at[ptable_access[atom] - 1, "c6"]
        df.at[a_index, "c6_gb"] = ptable.at[ptable_access[atom] - 1, "c6_gb"]
        df.at[a_index, "atomic_weight"] = ptable.at[ptable_access[atom] - 1, "atomic_weight"]

    molecule_structures[molecule[:-4]] = df
    if i%100 == 0:
        print(i)

print("Pickling")
with open(data_path_prefix + "molecule_structures.pkl",'wb') as f:
    pk.dump(molecule_structures,f,pk.HIGHEST_PROTOCOL)

with open(data_path_prefix + "molecule_composition_data.pkl",'wb') as f:
    pk.dump(molecule_structures,f,pk.HIGHEST_PROTOCOL)
print("DONE")

