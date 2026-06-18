# data process utils. file opts and simple calc.
import gzip
import lzma
import datetime
from enum import Enum
from pathlib import Path
from typing import cast, TypeAlias
import zstandard as zstd
from alphafold3.common.folding_input import (
    DnaChain,
    Ligand,
    ProteinChain,
    RnaChain,
)
from alphafold3.data.pipeline import DataPipelineConfig
from alphafold3.model.pipeline.pipeline import WholePdbPipeline

from alphafold3.utils.config import (
    BUCKETS,
    CONFORMER_MAX_ITERATIONS,
    HMMALIGN_BINARY_PATH,
    HMMBUILD_BINARY_PATH,
    HMMSEARCH_BINARY_PATH,
    JACKHMMER_BINARY_PATH,
    JACKHMMER_N_CPU,
    MAX_TEMPLATE_DATE,
    MGNIFY_DATABASE_PATH,
    NHMMER_BINARY_PATH,
    NHMMER_N_CPU,
    NTRNA_DATABASE_PATH,
    PDB_DATABASE_PATH,
    RFAM_DATABASE_PATH,
    RNA_CENTRAL_DATABASE_PATH,
    SEQRES_DATABASE_PATH,
    SMALL_BFD_DATABASE_PATH,
    UNIPROT_CLUSTER_ANNOT_DATABASE_PATH,
    UNIREF90_DATABASE_PATH,
    UPPER_LETTERS_START,
)


ACID2RES_DICT = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


class ChainType(Enum):
    Protein = 1
    Ligand = 2
    RNA = 3
    DNA = 4


# sequence_id mod_index ccd_code one_letter_code
ModifiedResidueId: TypeAlias = tuple[int, str, str]

# id chain_type sequence
ChainData: TypeAlias = tuple[str, ChainType, str]


def read_file(path: Path) -> str:

    with open(path, "rb") as f:
        first_six_bytes = f.read(6)
        f.seek(0)

        # Detect the compression type using the magic number in the header.
        if first_six_bytes[:2] == b"\x1f\x8b":
            with gzip.open(f, "rt") as gzip_f:
                return cast(str, gzip_f.read())
        elif first_six_bytes == b"\xfd\x37\x7a\x58\x5a\x00":
            with lzma.open(f, "rt") as xz_f:
                return cast(str, xz_f.read())
        elif first_six_bytes[:4] == b"\x28\xb5\x2f\xfd":
            with zstd.open(f, "rt") as zstd_f:
                return cast(str, zstd_f.read())
        else:
            return f.read().decode("utf-8")


def mock_sequence_id(index: int) -> str:
    return chr(index + UPPER_LETTERS_START)


def mock_chain(
    sequence: str,
    sequence_id: str,
    chain_type: ChainType,
    smiles: str | None = None,
    ptms: list[tuple[str, int]] | None = [],
) -> ProteinChain:
    if chain_type == ChainType.Ligand:
        if isinstance(sequence, list):
            sequence = tuple(sequence)
        if isinstance(sequence, str):
            sequence = (sequence,)
        return Ligand(id=sequence_id, ccd_ids=sequence, smiles=smiles)

    if chain_type == ChainType.RNA:
        return RnaChain(id=sequence_id, sequence=sequence, modifications=ptms)

    if chain_type == ChainType.DNA:
        return DnaChain(id=sequence_id, sequence=sequence, modifications=ptms)

    return ProteinChain(
        id=sequence_id,
        sequence=sequence,
        ptms=ptms,
        paired_msa=None,
        unpaired_msa=None,
        templates=None,
    )


def mock_data_pipeline_config() -> DataPipelineConfig:
    max_template_date = datetime.date.fromisoformat(MAX_TEMPLATE_DATE)
    data_pipeline_config = DataPipelineConfig(
        jackhmmer_binary_path=JACKHMMER_BINARY_PATH,
        nhmmer_binary_path=NHMMER_BINARY_PATH,
        hmmalign_binary_path=HMMALIGN_BINARY_PATH,
        hmmsearch_binary_path=HMMSEARCH_BINARY_PATH,
        hmmbuild_binary_path=HMMBUILD_BINARY_PATH,
        small_bfd_database_path=SMALL_BFD_DATABASE_PATH,
        mgnify_database_path=MGNIFY_DATABASE_PATH,
        uniprot_cluster_annot_database_path=UNIPROT_CLUSTER_ANNOT_DATABASE_PATH,
        uniref90_database_path=UNIREF90_DATABASE_PATH,
        ntrna_database_path=NTRNA_DATABASE_PATH,
        rfam_database_path=RFAM_DATABASE_PATH,
        rna_central_database_path=RNA_CENTRAL_DATABASE_PATH,
        pdb_database_path=PDB_DATABASE_PATH,
        seqres_database_path=SEQRES_DATABASE_PATH,
        jackhmmer_n_cpu=JACKHMMER_N_CPU,
        nhmmer_n_cpu=NHMMER_N_CPU,
        max_template_date=max_template_date,
    )
    return data_pipeline_config


def mock_model_pipeline_config() -> WholePdbPipeline.Config:
    max_template_date = datetime.date.fromisoformat(MAX_TEMPLATE_DATE)
    return WholePdbPipeline.Config(
        buckets=tuple(int(bucket) for bucket in BUCKETS),
        max_template_date=max_template_date,
        conformer_max_iterations=CONFORMER_MAX_ITERATIONS,
    )


if __name__ == "__main__":
    a = "SMTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETSLLDILDTAGQEEYSAMRDQYMRTGEGFLLVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKSDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEK"
    for s in a:
        if s not in ACID2RES_DICT:
            print(s)
