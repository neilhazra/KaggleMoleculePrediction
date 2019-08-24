import tensorflow as tf
import pandas as pd
import math
import numpy as np
import pickle as pk
from subprocess import call

from sklearn.model_selection import train_test_split
data_path_prefix = "../data/"

class molecule_prediction_data_wrapper:
    def __init__(self):
        self.molecule_structures = {}
        self.molecule_prop = {}
        self.columb_matrices = {}
        self.molecule_set = None
        self.molecule_mulliken_set = None
        self.molecule_properties_set = None
        self.molecules_train = []
        self.molecules_test = []
        self.valid_molecule_set = None

        try:
            print("Reading molecule structures")
            with open(data_path_prefix + "molecule_composition_data.pkl", 'rb') as f:
                self.molecule_structures = pk.load(f)
            print("Done Reading molecule structures")
        except:
            print("error atom")
            print("Running ReadAtomProperties this will take a while")
            call(["python", "ReadAtomProperties.py"])
            with open(data_path_prefix + "molecule_composition_data.pkl", 'rb') as f:
                self.molecule_structures = pk.load(f)
        try:
            print("Reading molecule prop")
            with open(data_path_prefix + "molecule_prop.pkl", 'rb') as f:
                self.molecule_prop = pk.load(f)
            print("Done Reading molecule prop")
        except:
            print("error prop")
            print("Running ReadMoleculeProperties this will take a while")
            call(["python", "ReadMoleculeProperties.py"])
            with open(data_path_prefix + "molecule_prop.pkl", 'rb') as f:
                self.molecule_prop = pk.load(f)

        with open(data_path_prefix + "set_of_molecules.pkl", 'rb') as f:
            self.molecule_set = pk.load(f)
        with open(data_path_prefix + "molecule_prop_set.pkl", 'rb') as f:
            self.molecule_properties_set = pk.load(f)
        with open(data_path_prefix + "set_of_molecules_with_mulliken.pkl", 'rb') as f:
            self.molecule_mulliken_set = pk.load(f)

        try:
            print("Reading Coulomb_mat")
            with open(data_path_prefix + "coulomb_mat.pkl", 'rb') as f:
                self.columb_matrices = pk.load(f)
            print("done reading coulomb mat")
        except:
            print("error coulomb, regenerating matrices")
            self.genColumbMatrix()
            with open(data_path_prefix + "coulomb_mat.pkl", 'rb') as f:
                self.columb_matrices = pk.load(f)

        self.valid_molecule_set = self.molecule_set.intersection \
            (self.molecule_properties_set, self.molecule_mulliken_set)
        try:
            with open(data_path_prefix + "test_molecules.pkl", 'rb') as f:
                self.molecules_test = pk.load(f)
            with open(data_path_prefix + "train_molecules.pkl", 'rb') as f:
                self.molecules_train = pk.load(f)
        except:
            molecule_list = np.array(list(self.valid_molecule_set))
            self.molecules_train, self.molecules_test = train_test_split \
                (molecule_list, test_size=0.2)
            with open(data_path_prefix + "test_molecules.pkl", 'wb') as f:
                pk.dump(self.molecules_test, f, pk.HIGHEST_PROTOCOL)
            with open(data_path_prefix + "train_molecules.pkl", 'wb') as f:
                pk.dump(self.molecules_train, f, pk.HIGHEST_PROTOCOL)
        self.batch_gen = BatchGenerator(self)
        #created batchgen
    def genColumbMatrix(self):
        columb_matrices = {}
        if len(columb_matrices) < 130_000:
            for n, molecule in enumerate(self.molecule_set):
                if n % 1000 == 0:
                    print(n)
                df = self.molecule_structures[molecule]
                num_atoms = len(df)
                matrix = np.empty((num_atoms, num_atoms))
                for i in range(num_atoms):
                    for j in range(i + 1):
                        if i == j:
                            matrix[i][j] = 0.5 * df.loc[i, "num_protons"] ** 2.4
                            continue
                        z_x_z = df.loc[i, "num_protons"] * df.loc[j, "num_protons"]
                        r_2 = (df.loc[i, "x"] - df.loc[j, "x"]) ** 2 + \
                              (df.loc[i, "y"] - df.loc[j, "y"]) ** 2 + \
                              (df.loc[i, "z"] - df.loc[j, "z"]) ** 2
                        m = z_x_z / math.sqrt(r_2)
                        matrix[i][j] = m
                        matrix[j][i] = m
                columb_matrices[molecule] = matrix
            with open(data_path_prefix + "coulomb_mat.pkl", 'wb') as f:
                pk.dump(columb_matrices, f, pk.HIGHEST_PROTOCOL)
    def get_input_vector(self, molecule):
        df = self.molecule_structures[molecule]
        if molecule in self.molecule_mulliken_set:
            df = df.drop(columns=['atom', 'x', 'y', 'z', 'mulliken'])
        else:
            df = df.drop(columns=['atom', 'x', 'y', 'z'])
        atom_array = np.pad(df.values, [(0, 30 - df.shape[0]), (0, 0)], 'constant', constant_values=(0.0, 0.0))
        padded_cm = pad(self.columb_matrices[molecule], (30, 30), (0, 0))
        input_mat = np.concatenate((atom_array, padded_cm), axis=1)
        return np.ravel(input_mat)
    def get_output_vector(self,molecule):
        df = self.molecule_prop[molecule]
        return np.ravel(df.drop(columns=['molecule_name', 'potential_energy']).values)
def pad(array, reference_shape, offsets):
    result = np.zeros(reference_shape)
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
    result[insertHere] = array
    return result
class BatchGenerator:
    def __init__(self, wrap,batch_size=8):
        self.wrap = wrap
        self.molecules = np.array(wrap.molecules_train)
        self.samples = len(self.molecules)
        self.buffer = []
    def generate(self):
        for i, molecule in enumerate(self.molecules):
            if len(self.buffer) > i:
                yield self.buffer[i]
            else:
                input_vector = self.wrap.get_input_vector(molecule)
                output_vector = self.wrap.get_output_vector(molecule)
                self.buffer.append((input_vector, output_vector))
                yield input_vector, output_vector















