"""PDB parsing utilities for ProteinMPNN."""

from typing import List, Dict, Optional, Tuple
import numpy as np


# Amino acid mappings
ALPHA_1 = list("ARNDCQEGHILKMFPSTWYV-")
ALPHA_3 = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP'
]
AA_1_N = {a: n for n, a in enumerate(ALPHA_1)}
AA_3_N = {a: n for n, a in enumerate(ALPHA_3)}
AA_N_1 = {n: a for n, a in enumerate(ALPHA_1)}
AA_3_1 = {b: a for a, b in zip(ALPHA_1, ALPHA_3)}


def _N_to_AA(x: np.ndarray) -> List[str]:
    """Convert numeric sequence to amino acid string."""
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([AA_N_1.get(a, "-") for a in y]) for y in x]


def parse_PDB_biounits(
    filename: str,
    atoms: List[str] = None,
    chain: str = None
) -> Tuple:
    """
    Parse a PDB file and extract coordinates.

    Args:
        filename: Path to PDB file
        atoms: List of atom types to extract (default: ['N', 'CA', 'C'])
        chain: Specific chain to extract (default: all chains)

    Returns:
        Tuple of (coordinates array, sequence list)
    """
    if atoms is None:
        atoms = ['N', 'CA', 'C']

    xyz, seq = {}, {}
    min_resn, max_resn = 1e6, -1e6

    with open(filename, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()

            # Handle selenomethionine
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ")
                line = line.replace("MSE", "MET")

            if line[:4] == "ATOM":
                ch = line[21:22]
                if ch == chain or chain is None:
                    atom = line[12:16].strip()
                    resi = line[17:20]
                    resn = line[22:27].strip()

                    try:
                        x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]
                    except ValueError:
                        continue

                    if resn[-1].isalpha():
                        resa, resn = resn[-1], int(resn[:-1]) - 1
                    else:
                        resa, resn = "", int(resn) - 1

                    if resn < min_resn:
                        min_resn = resn
                    if resn > max_resn:
                        max_resn = resn
                    if resn not in xyz:
                        xyz[resn] = {}
                    if resa not in xyz[resn]:
                        xyz[resn][resa] = {}
                    if resn not in seq:
                        seq[resn] = {}
                    if resa not in seq[resn]:
                        seq[resn][resa] = resi

                    if atom not in xyz[resn][resa]:
                        xyz[resn][resa][atom] = np.array([x, y, z])

    # Convert to numpy arrays
    seq_, xyz_ = [], []
    try:
        for resn in range(int(min_resn), int(max_resn) + 1):
            if resn in seq:
                for k in sorted(seq[resn]):
                    seq_.append(AA_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms:
                    xyz_.append(np.full(3, np.nan))

        return np.array(xyz_).reshape(-1, len(atoms), 3), _N_to_AA(np.array(seq_))
    except (TypeError, ValueError):
        return 'no_chain', 'no_chain'


def parse_PDB(
    path_to_pdb: str,
    input_chain_list: List[str] = None,
    ca_only: bool = False
) -> List[Dict]:
    """
    Parse a PDB file and return structured data.

    Args:
        path_to_pdb: Path to PDB file
        input_chain_list: List of chains to parse (default: all)
        ca_only: Whether to extract CA atoms only

    Returns:
        List of dictionaries with parsed structure data
    """
    pdb_dict_list = []

    # Default chain alphabet
    init_alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    extra_alphabet = [str(item) for item in range(300)]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    my_dict = {}
    s = 0
    concat_seq = ''

    for letter in chain_alphabet:
        if ca_only:
            sidechain_atoms = ['CA']
        else:
            sidechain_atoms = ['N', 'CA', 'C', 'O']

        xyz, seq = parse_PDB_biounits(path_to_pdb, atoms=sidechain_atoms, chain=letter)

        if type(xyz) != str:
            concat_seq += seq[0]
            my_dict['seq_chain_' + letter] = seq[0]
            coords_dict_chain = {}

            if ca_only:
                coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
            else:
                coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()

            my_dict['coords_chain_' + letter] = coords_dict_chain
            s += 1

    fi = path_to_pdb.rfind("/")
    my_dict['name'] = path_to_pdb[(fi + 1):-4]
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq

    if s <= len(chain_alphabet):
        pdb_dict_list.append(my_dict)

    return pdb_dict_list
