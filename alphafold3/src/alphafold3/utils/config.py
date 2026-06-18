import os
import shutil
from pathlib import Path
from typing import Final, List

import numpy as np


USE_MSA: Final[bool] = False
USE_TEMPLATE: Final[bool] = False
USE_MOCK_TEMPLATE: Final[bool] = False

TEMPLATE_FILE_SUFFIX: Final[str] = "cif"

# input config
MODEL_SEEDS: Final[int] = 42

DIALECT: Final[str] = "alphafold3"

VERSION: Final[int] = 2


# model config
DEFAULT_MODEL_DIR: Final[Path] = Path("/data/soft/AF3-Model")
MODEL_DIR: Final[str] = DEFAULT_MODEL_DIR.as_posix()

# Control which stages to run.
RUN_DATA_PIPELINE: Final[bool] = True


# Binary paths.
JACKHMMER_BINARY_PATH: Final[str] = shutil.which("jackhmmer")
NHMMER_BINARY_PATH: Final[str] = shutil.which("nhmmer")
HMMALIGN_BINARY_PATH: Final[str] = shutil.which("hmmalign")
HMMSEARCH_BINARY_PATH: Final[str] = shutil.which("hmmsearch")
HMMBUILD_BINARY_PATH: Final[str] = shutil.which("hmmbuild")

# Database paths.
# DB dir
DEFAULT_DB_DIR: Final[Path] = Path("/data/soft/AF3-Database")
DB_DIR: Final[str] = DEFAULT_DB_DIR.as_posix()
SMALL_BFD_DATABASE_PATH: Final[str] = (
    f"{DB_DIR}/bfd-first_non_consensus_sequences.fasta"
)
MGNIFY_DATABASE_PATH: Final[str] = f"{DB_DIR}/mgy_clusters_2022_05.fa"
UNIPROT_CLUSTER_ANNOT_DATABASE_PATH: Final[str] = f"{DB_DIR}/uniprot_all_2021_04.fa"
UNIREF90_DATABASE_PATH: Final[str] = f"{DB_DIR}/uniprot_all_2021_04.fa"
NTRNA_DATABASE_PATH: Final[str] = (
    f"{DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
)
RFAM_DATABASE_PATH: Final[str] = (
    f"{DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
)
RNA_CENTRAL_DATABASE_PATH: Final[str] = (
    f"{DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta"
)
PDB_DATABASE_PATH: Final[str] = f"{DB_DIR}/mmcif_files"
SEQRES_DATABASE_PATH: Final[str] = f"{DB_DIR}/pdb_seqres_2022_09_28.fasta"

# Number of CPUs to use for MSA tools.
JACKHMMER_N_CPU: Final[int] = min(os.cpu_count(), 96)
NHMMER_N_CPU: Final[int] = min(os.cpu_count(), 96)

# Template search configuration.
MAX_TEMPLATE_DATE: Final[str] = "2026-01-30"

CONFORMER_MAX_ITERATIONS = None

# JAX inference performance tuning.
JAX_COMPILATION_CACHE_DIR = None

GPU_DEVICE: Final[int] = 0
BUCKETS: Final[List[int]] = [
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
]
BUCKETS_INT = tuple(int(bucket) for bucket in BUCKETS)

# enum_values=["triton", "cudnn", "xla"]
FLASH_ATTENTION_IMPLEMENTATION_FOR_INFER: Final[str] = "triton"
NUM_RECYCLES_FOR_INFER: Final[int] = 10
NUM_DIFFUSION_SAMPLES_FOR_INFER: Final[int] = 5
NUM_SEEDS: Final[int] = 1

# Output controls.
SAVE_EMBEDDINGS: Final[bool] = False

UPPER_LETTERS_START: Final[int] = 65

VALID_DTYPES: Final[List[type]] = [
    np.float32,
    np.float64,
    np.int8,
    np.int32,
    np.int64,
    bool,
]
