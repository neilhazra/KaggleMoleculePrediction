import pandas as pd
import pickle as pk
from mendeleev import get_table
ptable = get_table('elements')
import os

set_of_molecules = set()
set_of_molecules_with_mulliken = set()

ptable_access = {}
for _,row in ptable.iterrows():
    ptable_access[row["symbol"]] = row["atomic_number"]

data_path_prefix = "../data/"
path_to_structures = data_path_prefix + "/structures/"

molecule_structures = {}

#with open(data_path_prefix + "molecule_structures.pkl",'rb') as f:
#    molecule_structures = pk.load(f)

print("Processing Structures and atom properties")
for i,molecule in enumerate(os.listdir(path_to_structures)):
    df = pd.read_csv(path_to_structures + molecule, sep = ' ', skiprows= [0],names = ["atom","x","y","z"])
    for a_index in range(len(df)):
        atom = df["atom"].iloc[a_index]
        df.at[a_index, "electron_affinity"] = ptable.at[ptable_access[atom]-1,"electron_affinity"]
        df.at[a_index, "electronegativity"] = ptable.at[ptable_access[atom]-1,"en_pauling"]
        df.at[a_index, "num_protons"] = ptable_access[atom]
        df.at[a_index, "dipole_polarizability"] = ptable.at[ptable_access[atom]-1,"dipole_polarizability"]

    molecule_structures[molecule[:-4]] = df
    set_of_molecules.add(molecule[:-4])
    if i%100 == 0:
        print(i)

print("Pickling")
with open(data_path_prefix + "molecule_structures.pkl",'wb') as f:
    pk.dump(molecule_structures,f,pk.HIGHEST_PROTOCOL)

mc = pd.read_csv(data_path_prefix + "mulliken_charges.csv")
molecule = ""
print("Processing mulliken_charges")
m_df = None
for i, row in mc.iterrows():
    temp = row["molecule_name"]
    if molecule != temp:
        molecule = temp
        m_df = molecule_structures[molecule]
        set_of_molecules_with_mulliken.add(molecule)
    if i%10 == 0:
        print(i)
    a_index = row["atom_index"]
    m_df.at[a_index,"mulliken"] = row["mulliken_charge"]


print("Pickling")
with open(data_path_prefix + "molecule_composition_data.pkl",'wb') as f:
    pk.dump(molecule_structures,f,pk.HIGHEST_PROTOCOL)
print("DONE")
with open(data_path_prefix + "set_of_molecules_with_mulliken.pkl",'wb') as f:
    pk.dump(set_of_molecules_with_mulliken,f,pk.HIGHEST_PROTOCOL)
print("DONE")
with open(data_path_prefix + "set_of_molecules.pkl",'wb') as f:
    pk.dump(set_of_molecules,f,pk.HIGHEST_PROTOCOL)
print("DONE")
