"""Dataset classes for ProteinMPNN."""

import json
import time
from typing import List, Dict, Optional, Callable
import numpy as np


class StructureDataset:
    """Dataset for loading pre-parsed protein structures from JSONL files."""

    def __init__(
        self,
        jsonl_file: str,
        verbose: bool = True,
        truncate: Optional[int] = None,
        max_length: int = 100,
        alphabet: str = 'ACDEFGHIKLMNPQRSTVWYX-'
    ):
        """
        Initialize structure dataset.

        Args:
            jsonl_file: Path to JSONL file with parsed structures
            verbose: Whether to print loading progress
            truncate: Maximum number of structures to load
            max_length: Maximum sequence length
            alphabet: Valid amino acid alphabet
        """
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []
            lines = f.readlines()
            start = time.time()

            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                bad_chars = set([s for s in seq]).difference(alphabet_set)
                if len(bad_chars) == 0:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    if verbose:
                        print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                if truncate is not None and len(self.data) == truncate:
                    break

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print(f'{len(self.data)} entries ({i+1} loaded) in {elapsed:.1f} s')

            if verbose:
                print('discarded', discard_count)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class StructureDatasetPDB:
    """Dataset for protein structures parsed directly from PDB."""

    def __init__(
        self,
        pdb_dict_list: List[Dict],
        verbose: bool = True,
        truncate: Optional[int] = None,
        max_length: int = 100,
        alphabet: str = 'ACDEFGHIKLMNPQRSTVWYX-'
    ):
        """
        Initialize PDB structure dataset.

        Args:
            pdb_dict_list: List of parsed PDB dictionaries
            verbose: Whether to print loading progress
            truncate: Maximum number of structures to load
            max_length: Maximum sequence length
            alphabet: Valid amino acid alphabet
        """
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []
        start = time.time()

        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            if truncate is not None and len(self.data) == truncate:
                break

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


class StructureLoader:
    """Batch loader that clusters proteins by similar lengths."""

    def __init__(
        self,
        dataset,
        batch_size: int = 100,
        shuffle: bool = True,
        collate_fn: Callable = lambda x: x,
        drop_last: bool = False
    ):
        """
        Initialize structure loader.

        Args:
            dataset: Dataset to load from
            batch_size: Maximum tokens per batch
            shuffle: Whether to shuffle batches
            collate_fn: Collation function
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self) -> int:
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
