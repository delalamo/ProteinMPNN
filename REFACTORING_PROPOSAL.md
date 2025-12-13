# ProteinMPNN Refactoring Proposal

## Executive Summary

This document analyzes information flow patterns in the ProteinMPNN codebase and proposes several approaches to improve readability and maintainability. The key issues are:

1. Functions with excessive return values (up to 20 items)
2. Functions with many keyword arguments (18+)
3. Monolithic main functions (400+ lines)
4. Duplicated code between inference and training modules
5. Configuration data scattered across multiple dictionaries

---

## Current Information Flow Analysis

### 1. Key Problematic Functions

#### `tied_featurize()` - Returns 20 Values
**Location**: `protein_mpnn_utils.py:191-436`

```python
def tied_featurize(batch, device, chain_dict, fixed_position_dict=None,
                   omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None,
                   bias_by_res_dict=None, ca_only=False):
    ...
    return X_out, S, mask, lengths, chain_M, chain_encoding_all, \
           letter_list_list, visible_list_list, masked_list_list, \
           masked_chain_length_list_list, chain_M_pos, omit_AA_mask, \
           residue_idx, dihedral_mask, tied_pos_list_of_lists_list, \
           pssm_coef_all, pssm_bias_all, pssm_log_odds_all, bias_by_res_all, tied_beta
```

**Issues**:
- Returns 20 separate tensors/lists
- Caller must unpack in exact order (error-prone)
- Adding/removing outputs requires changes everywhere

#### `model.sample()` - 18+ Keyword Arguments
**Location**: `protein_mpnn_utils.py:1104-1188`

```python
def sample(self, X, randn, S_true, chain_mask, chain_encoding_all, residue_idx,
           mask=None, temperature=1.0, omit_AAs_np=None, bias_AAs_np=None,
           chain_M_pos=None, omit_AA_mask=None, pssm_coef=None, pssm_bias=None,
           pssm_multi=None, pssm_log_odds_flag=None, pssm_log_odds_mask=None,
           pssm_bias_flag=None, bias_by_res=None):
```

**Issues**:
- Hard to read and maintain
- Many arguments are related (PSSM arguments, bias arguments)
- Easy to pass arguments in wrong positions

#### `model.tied_sample()` - Similar Issues
**Location**: `protein_mpnn_utils.py:1191-1289`

Has the same problems plus two additional arguments (`tied_pos`, `tied_beta`).

### 2. Monolithic Main Functions

#### `main()` in `protein_mpnn_run.py` (469 lines)
Responsibilities mixed together:
- Argument processing
- Model loading
- Configuration dictionary loading (7 different dictionaries)
- Output directory creation
- Score-only mode
- Conditional probability mode
- Unconditional probability mode
- Sequence generation mode
- Result formatting and saving

### 3. Duplicated Code

| Component | `protein_mpnn_utils.py` | `training/model_utils.py` |
|-----------|------------------------|---------------------------|
| `ProteinMPNN` class | Lines 1019-1382 | Lines 396-470 |
| `ProteinFeatures` class | Lines 920-1015 | Lines 297-392 |
| `EncLayer` class | Lines 623-668 | Lines 182-227 |
| `DecLayer` class | Lines 671-709 | Lines 231-269 |
| `gather_*` functions | Lines 595-620 | Lines 154-179 |
| `loss_*` functions | Lines 440-460 | Lines 128-150 |

### 4. Configuration Data Flow

```
main() loads 7 dictionaries:
├── chain_id_dict      → tied_featurize() → model.forward()
├── fixed_positions_dict → tied_featurize()
├── omit_AA_dict       → tied_featurize()
├── tied_positions_dict → tied_featurize() → model.tied_sample()
├── pssm_dict          → tied_featurize() → model.sample()
├── bias_AA_dict       → main() processes → model.sample()
└── bias_by_res_dict   → tied_featurize() → model.sample()
```

---

## Proposed Refactoring Approaches

### Approach 1: Data Classes (Recommended)

**Philosophy**: Group related data into typed containers.

#### A. Create `ProteinBatch` dataclass for featurize output

```python
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class ProteinBatch:
    """Container for featurized protein batch data."""
    # Core tensors
    X: torch.Tensor              # Coordinates [B, L, 4, 3] or [B, L, 3]
    S: torch.Tensor              # Sequence indices [B, L]
    mask: torch.Tensor           # Valid position mask [B, L]
    lengths: np.ndarray          # Sequence lengths [B]
    residue_idx: torch.Tensor    # Residue indices [B, L]

    # Chain information
    chain_M: torch.Tensor              # Design mask [B, L]
    chain_M_pos: torch.Tensor          # Position design mask [B, L]
    chain_encoding_all: torch.Tensor   # Chain IDs [B, L]

    # Chain lists (for output formatting)
    chain_lists: 'ChainLists'

    # Optional constraint tensors
    constraints: Optional['ConstraintData'] = None
    pssm: Optional['PSSMData'] = None

@dataclass
class ChainLists:
    """Chain letter information for output formatting."""
    letter_list: List[List[str]]
    visible_list: List[List[str]]
    masked_list: List[List[str]]
    masked_chain_lengths: List[List[int]]
    tied_positions: List[List[List[int]]]

@dataclass
class ConstraintData:
    """Constraint tensors for sequence design."""
    omit_AA_mask: torch.Tensor    # [B, L, 21]
    dihedral_mask: torch.Tensor   # [B, L, 3]
    tied_beta: torch.Tensor       # [L]

@dataclass
class PSSMData:
    """PSSM-related tensors."""
    coef: torch.Tensor       # [B, L]
    bias: torch.Tensor       # [B, L, 21]
    log_odds: torch.Tensor   # [B, L, 21]
    bias_by_res: torch.Tensor # [B, L, 21]
```

#### B. Create `SamplingConfig` dataclass for sample() arguments

```python
@dataclass
class SamplingConfig:
    """Configuration for sequence sampling."""
    temperature: float = 1.0

    # Global AA constraints
    omit_AAs_np: Optional[np.ndarray] = None
    bias_AAs_np: Optional[np.ndarray] = None

    # Per-position constraints
    omit_AA_mask: Optional[torch.Tensor] = None
    bias_by_res: Optional[torch.Tensor] = None

    # PSSM settings
    pssm_coef: Optional[torch.Tensor] = None
    pssm_bias: Optional[torch.Tensor] = None
    pssm_multi: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_log_odds_mask: Optional[torch.Tensor] = None
    pssm_bias_flag: bool = False

    # Tied positions (for symmetry)
    tied_pos: Optional[List[List[int]]] = None
    tied_beta: Optional[torch.Tensor] = None
```

#### C. Refactored function signatures

```python
# Before (20 return values)
def tied_featurize(batch, device, chain_dict, ...):
    return X, S, mask, lengths, chain_M, chain_encoding_all, ...

# After (1 return value)
def featurize_batch(batch, device, config: DesignConfig) -> ProteinBatch:
    return ProteinBatch(...)

# Before (18+ arguments)
def sample(self, X, randn, S_true, chain_mask, ..., pssm_bias_flag=None):
    ...

# After (4 arguments)
def sample(self, batch: ProteinBatch, randn: torch.Tensor,
           config: SamplingConfig) -> SampleOutput:
    ...
```

**Benefits**:
- Type hints provide documentation
- IDE auto-completion works
- Easy to add/remove fields
- Clear grouping of related data
- Single return value

---

### Approach 2: Builder Pattern for Configuration

**Philosophy**: Use a builder to construct complex configurations step-by-step.

```python
class DesignJobBuilder:
    """Builder for constructing protein design jobs."""

    def __init__(self, pdb_path_or_jsonl: str):
        self._structure_source = pdb_path_or_jsonl
        self._fixed_positions = {}
        self._tied_positions = {}
        self._pssm = {}
        self._bias = {}
        self._omit_aa = {}
        self._chains_to_design = None

    def fix_positions(self, chain: str, positions: List[int]) -> 'DesignJobBuilder':
        """Specify positions to keep fixed."""
        self._fixed_positions.setdefault(chain, []).extend(positions)
        return self

    def tie_positions(self, *position_groups: Tuple[str, int]) -> 'DesignJobBuilder':
        """Tie positions together for symmetry."""
        # position_groups like [('A', 1), ('B', 1)] ties A:1 to B:1
        ...
        return self

    def add_pssm(self, chain: str, pssm_data: np.ndarray) -> 'DesignJobBuilder':
        """Add PSSM constraint for a chain."""
        ...
        return self

    def design_chains(self, *chains: str) -> 'DesignJobBuilder':
        """Specify which chains to design (others are fixed)."""
        self._chains_to_design = list(chains)
        return self

    def omit_amino_acids(self, aas: str,
                         positions: Optional[Dict[str, List[int]]] = None) -> 'DesignJobBuilder':
        """Omit specific amino acids globally or at positions."""
        ...
        return self

    def build(self) -> 'DesignJob':
        """Construct the final DesignJob."""
        return DesignJob(
            structure_source=self._structure_source,
            fixed_positions=self._fixed_positions,
            tied_positions=self._tied_positions,
            pssm=self._pssm,
            bias=self._bias,
            omit_aa=self._omit_aa,
            chains_to_design=self._chains_to_design
        )

# Usage
job = (DesignJobBuilder("1abc.pdb")
       .design_chains('A', 'B')
       .fix_positions('A', [1, 5, 10])
       .tie_positions(('A', 1), ('B', 1))
       .omit_amino_acids('C')  # no cysteines
       .build())
```

**Benefits**:
- Fluent interface is readable
- Each method documents its purpose
- Complex configurations are built incrementally
- Easy to extend with new options

---

### Approach 3: Namespace Objects (Simpler Alternative)

**Philosophy**: Use simple namespace objects to group related data without strict typing.

```python
from types import SimpleNamespace

def tied_featurize(batch, device, chain_dict, ...):
    ...

    # Group into namespaces
    tensors = SimpleNamespace(
        X=X_out, S=S, mask=mask, residue_idx=residue_idx,
        chain_M=chain_M, chain_M_pos=chain_M_pos,
        chain_encoding_all=chain_encoding_all,
        omit_AA_mask=omit_AA_mask, dihedral_mask=dihedral_mask
    )

    chain_info = SimpleNamespace(
        letter_list=letter_list_list,
        visible_list=visible_list_list,
        masked_list=masked_list_list,
        masked_chain_lengths=masked_chain_length_list_list
    )

    pssm = SimpleNamespace(
        coef=pssm_coef_all, bias=pssm_bias_all,
        log_odds=pssm_log_odds_all, bias_by_res=bias_by_res_all
    )

    tied = SimpleNamespace(
        positions=tied_pos_list_of_lists_list,
        beta=tied_beta
    )

    return SimpleNamespace(
        tensors=tensors, chain_info=chain_info,
        pssm=pssm, tied=tied, lengths=lengths
    )

# Usage
batch_data = tied_featurize(batch, device, chain_dict, ...)
X = batch_data.tensors.X
pssm_coef = batch_data.pssm.coef
```

**Benefits**:
- Minimal code changes
- No new class definitions needed
- Dot notation is cleaner than tuple unpacking
- Easy to add new fields

---

### Approach 4: Modular Architecture Refactor

**Philosophy**: Split the codebase into focused modules with clear responsibilities.

```
proteinmpnn/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── model.py              # ProteinMPNN model only
│   ├── features.py           # ProteinFeatures, CA_ProteinFeatures
│   ├── layers.py             # EncLayer, DecLayer, PositionWiseFeedForward
│   └── embeddings.py         # PositionalEncodings
├── data/
│   ├── __init__.py
│   ├── types.py              # ProteinBatch, SamplingConfig dataclasses
│   ├── parsers.py            # parse_PDB, parse_fasta, parse_PDB_biounits
│   ├── featurize.py          # tied_featurize, featurize
│   └── datasets.py           # StructureDataset, StructureDatasetPDB, StructureLoader
├── config/
│   ├── __init__.py
│   ├── loaders.py            # load_chain_dict, load_fixed_positions, etc.
│   └── types.py              # DesignConfig, TrainingConfig
├── inference/
│   ├── __init__.py
│   ├── sampler.py            # SequenceSampler class
│   ├── scorer.py             # SequenceScorer class
│   └── runner.py             # Main inference orchestration
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Training loop
│   └── losses.py             # loss_nll, loss_smoothed
└── utils/
    ├── __init__.py
    ├── tensor_ops.py         # gather_edges, gather_nodes, cat_neighbors_nodes
    └── helpers.py            # _scores, _S_to_seq
```

#### Key Changes:

**1. Unified `model.py` (replaces duplication)**
```python
# proteinmpnn/core/model.py
class ProteinMPNN(nn.Module):
    """Unified ProteinMPNN model for training and inference."""

    def __init__(self, config: ModelConfig):
        ...

    def forward(self, batch: ProteinBatch, decoding_order: Optional[torch.Tensor] = None):
        """Forward pass for training or scoring."""
        ...

    def sample(self, batch: ProteinBatch, config: SamplingConfig) -> SampleOutput:
        """Generate sequences."""
        ...

    def encode(self, batch: ProteinBatch) -> EncoderOutput:
        """Run encoder only (useful for analysis)."""
        ...
```

**2. `SequenceSampler` class (encapsulates sampling logic)**
```python
# proteinmpnn/inference/sampler.py
class SequenceSampler:
    """High-level interface for sequence sampling."""

    def __init__(self, model: ProteinMPNN, device: torch.device):
        self.model = model
        self.device = device

    def sample_sequences(
        self,
        structure: ProteinBatch,
        num_sequences: int = 1,
        temperature: float = 0.1,
        config: Optional[SamplingConfig] = None
    ) -> List[SampleResult]:
        """Generate sequences for a protein structure."""
        ...

    def sample_with_tied_positions(
        self,
        structure: ProteinBatch,
        tied_groups: List[List[Tuple[str, int]]],
        **kwargs
    ) -> List[SampleResult]:
        """Generate sequences with tied position constraints."""
        ...
```

**3. Config loaders module**
```python
# proteinmpnn/config/loaders.py
@dataclass
class DesignConfig:
    """All configuration for a design job."""
    chain_id_dict: Optional[Dict] = None
    fixed_positions: Optional[Dict] = None
    omit_AA: Optional[Dict] = None
    tied_positions: Optional[Dict] = None
    pssm: Optional[Dict] = None
    bias_AA: Optional[Dict] = None
    bias_by_res: Optional[Dict] = None

def load_design_config(
    chain_id_jsonl: str = '',
    fixed_positions_jsonl: str = '',
    omit_AA_jsonl: str = '',
    tied_positions_jsonl: str = '',
    pssm_jsonl: str = '',
    bias_AA_jsonl: str = '',
    bias_by_res_jsonl: str = ''
) -> DesignConfig:
    """Load all configuration files into a single config object."""
    return DesignConfig(
        chain_id_dict=_load_jsonl(chain_id_jsonl),
        fixed_positions=_load_jsonl(fixed_positions_jsonl),
        ...
    )
```

---

### Approach 5: Functional Composition with Context Objects

**Philosophy**: Pass a context object that accumulates state through a pipeline.

```python
class DesignContext:
    """Mutable context object passed through the design pipeline."""

    def __init__(self, structure_path: str):
        self.structure_path = structure_path
        self.device = None
        self.batch = None
        self.config = DesignConfig()
        self.results = []

    def with_device(self, device: torch.device) -> 'DesignContext':
        self.device = device
        return self

    def with_config(self, config: DesignConfig) -> 'DesignContext':
        self.config = config
        return self

# Pipeline functions operate on context
def load_structure(ctx: DesignContext) -> DesignContext:
    """Load and parse protein structure."""
    pdb_dict = parse_PDB(ctx.structure_path)
    ctx.pdb_dict = pdb_dict
    return ctx

def featurize(ctx: DesignContext) -> DesignContext:
    """Convert structure to tensor batch."""
    ctx.batch = tied_featurize_to_batch(ctx.pdb_dict, ctx.device, ctx.config)
    return ctx

def generate_sequences(ctx: DesignContext, model: ProteinMPNN,
                       num_seqs: int, temperature: float) -> DesignContext:
    """Generate sequences and store results."""
    results = model.sample(ctx.batch, SamplingConfig(temperature=temperature))
    ctx.results.extend(results)
    return ctx

# Usage with composition
ctx = (DesignContext("1abc.pdb")
       .with_device(torch.device("cuda"))
       .with_config(load_design_config(...)))

ctx = load_structure(ctx)
ctx = featurize(ctx)
ctx = generate_sequences(ctx, model, num_seqs=10, temperature=0.1)
```

---

## Comparison of Approaches

| Criterion | Dataclasses | Builder | Namespace | Modular | Context |
|-----------|-------------|---------|-----------|---------|---------|
| Code changes required | Medium | Medium | Low | High | Medium |
| Type safety | High | Medium | Low | High | Medium |
| Readability | High | High | Medium | High | High |
| Learning curve | Low | Medium | Low | Medium | Medium |
| Extensibility | High | High | Medium | High | High |
| Backwards compat | Medium | High | High | Low | Medium |

---

## Recommended Implementation Strategy

### Phase 1: Quick Wins (Low Risk)
1. **Create data classes** for `tied_featurize` output and `sample` config
2. **Extract config loading** into a dedicated function
3. Keep existing function signatures as aliases

### Phase 2: Consolidation (Medium Risk)
1. **Merge duplicate code** between inference and training modules
2. **Split `main()` function** into focused helpers
3. Create high-level `SequenceSampler` and `SequenceScorer` classes

### Phase 3: Full Refactor (Higher Risk)
1. Implement modular architecture
2. Deprecate old function signatures
3. Update documentation and examples

---

## Specific Refactoring Recommendations

### Immediate Changes

#### 1. `tied_featurize` → Returns `ProteinBatch`
```python
# Add at top of protein_mpnn_utils.py
@dataclass
class ProteinBatch:
    X: torch.Tensor
    S: torch.Tensor
    mask: torch.Tensor
    lengths: np.ndarray
    chain_M: torch.Tensor
    chain_encoding_all: torch.Tensor
    chain_M_pos: torch.Tensor
    omit_AA_mask: torch.Tensor
    residue_idx: torch.Tensor
    dihedral_mask: torch.Tensor
    pssm_coef: torch.Tensor
    pssm_bias: torch.Tensor
    pssm_log_odds: torch.Tensor
    bias_by_res: torch.Tensor
    tied_beta: torch.Tensor
    # Metadata
    letter_list: List[List[str]]
    visible_list: List[List[str]]
    masked_list: List[List[str]]
    masked_chain_lengths: List[List[int]]
    tied_positions: List[List[List[int]]]

# Keep old signature for backwards compatibility
def tied_featurize(batch, device, chain_dict, ...):
    result = tied_featurize_v2(batch, device, chain_dict, ...)
    # Return tuple for backwards compat
    return (result.X, result.S, result.mask, ...)

def tied_featurize_v2(batch, device, chain_dict, ...) -> ProteinBatch:
    # New implementation returning dataclass
    ...
```

#### 2. `sample()` → Takes `SamplingConfig`
```python
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    omit_AAs_np: Optional[np.ndarray] = None
    bias_AAs_np: Optional[np.ndarray] = None
    chain_M_pos: Optional[torch.Tensor] = None
    omit_AA_mask: Optional[torch.Tensor] = None
    pssm_coef: Optional[torch.Tensor] = None
    pssm_bias: Optional[torch.Tensor] = None
    pssm_multi: float = 0.0
    pssm_log_odds_flag: bool = False
    pssm_log_odds_mask: Optional[torch.Tensor] = None
    pssm_bias_flag: bool = False
    bias_by_res: Optional[torch.Tensor] = None

class ProteinMPNN(nn.Module):
    # Keep old signature
    def sample(self, X, randn, S_true, chain_mask, chain_encoding_all,
               residue_idx, mask=None, temperature=1.0, ...):
        config = SamplingConfig(temperature=temperature, ...)
        return self.sample_v2(X, randn, S_true, chain_mask,
                              chain_encoding_all, residue_idx, mask, config)

    # New signature
    def sample_v2(self, X, randn, S_true, chain_mask, chain_encoding_all,
                  residue_idx, mask, config: SamplingConfig):
        ...
```

#### 3. Extract config loading from `main()`
```python
def load_all_configs(args) -> Dict[str, Any]:
    """Load all configuration dictionaries."""
    configs = {}

    config_files = [
        ('chain_id_dict', args.chain_id_jsonl),
        ('fixed_positions_dict', args.fixed_positions_jsonl),
        ('pssm_dict', args.pssm_jsonl),
        ('omit_AA_dict', args.omit_AA_jsonl),
        ('bias_AA_dict', args.bias_AA_jsonl),
        ('tied_positions_dict', args.tied_positions_jsonl),
        ('bias_by_res_dict', args.bias_by_res_jsonl),
    ]

    for name, path in config_files:
        configs[name] = load_jsonl_config(path) if os.path.isfile(path) else None

    return configs

def load_jsonl_config(path: str) -> Dict:
    """Load a JSONL configuration file."""
    with open(path, 'r') as f:
        result = {}
        for line in f:
            result.update(json.loads(line))
    return result
```

---

## Conclusion

The recommended approach is to implement **Approach 1 (Data Classes)** combined with **Approach 4 (Modular Architecture)** in phases:

1. Start with dataclasses for immediate type safety and reduced function signatures
2. Gradually refactor into modular architecture
3. Maintain backwards compatibility through wrapper functions

This will result in:
- Functions with 3-5 arguments instead of 18+
- Single return values (dataclass objects) instead of 20-tuples
- Clear separation of concerns
- Type hints for better tooling support
- Easier testing and maintenance
