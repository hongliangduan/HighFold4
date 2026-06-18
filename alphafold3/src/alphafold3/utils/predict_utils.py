import copy
import os
from json import JSONDecodeError
from pathlib import Path
from typing import cast
import jax
import tokamax
from alphafold3.utils.config import (
    BUCKETS,
    FLASH_ATTENTION_IMPLEMENTATION_FOR_INFER,
    NUM_DIFFUSION_SAMPLES_FOR_INFER,
    NUM_RECYCLES_FOR_INFER,
    MODEL_DIR,
)
from alphafold3.utils.data_utils import (
    ChainType,
    ModifiedResidueId,
    mock_data_pipeline_config,
    mock_chain,
    mock_sequence_id,
)
from alphafold3.utils.infer_utils import process_fold_input, write_fold_input_json
from alphafold3.utils.input_utils import mock_peptiede_af3_input
from alphafold3.utils.model_utils import ModelRunner, make_model_config
from alphafold3.common.folding_input import Input, Ligand
from alphafold3.data.pipeline import DataPipeline
from Bio.PDB import PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Structure import Structure
from loguru import logger


def make_model_runner(
    device_index: int = 0,
    model_dir: str = MODEL_DIR,
    num_samples: int = NUM_DIFFUSION_SAMPLES_FOR_INFER,
    use_bonds: bool = True,
    use_offsets: bool = False,
) -> ModelRunner:

    devices = jax.local_devices(backend="gpu")
    device = devices[device_index]

    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=cast(
                tokamax.DotProductAttentionImplementation,
                FLASH_ATTENTION_IMPLEMENTATION_FOR_INFER,
            ),
            num_diffusion_samples=num_samples,
            num_recycles=NUM_RECYCLES_FOR_INFER,
            return_embeddings=False,
            return_distogram=False,
            use_bonds=use_bonds,
            use_offsets=use_offsets,
        ),
        device=device,
        model_dir=Path(model_dir),
    )

    return model_runner


def load_base_input(json_path: Path) -> Input | None:
    if json_path.exists():
        with open(json_path, "r") as f:
            json_str = f.read()
        try:
            fold_input = Input.from_json(json_str, json_path)
            return fold_input
        except JSONDecodeError as e:
            logger.warning(f"bad json {json_path}")
    return None


def make_base_input(
    task_id: str,
    sequences: list[str] = [],
    types: list[int] = [],
    modified_info: list[ModifiedResidueId] = {},
    bonds: list[tuple[tuple[int, str], tuple[int, str]]] = [],
    save_data: bool = True,
    save_dir: Path | None = None,
    user_ccd: str | None = None,
    only_use_cached_msa_templates: bool = False,
    cached_receptor_dir: Path | None = None,
) -> Input:
    """
    Create base input for inference.
    """
    data_pipeline_config = mock_data_pipeline_config()
    data_pipeline = DataPipeline(data_pipeline_config=data_pipeline_config)
    chains = []
    if len(types) == 0:
        types = [1] * len(sequences)
    else:
        assert len(sequences) == len(
            types
        ), "Sequences and types must have the same length."
    for i, seq in enumerate(sequences):
        chain_type = ChainType(types[i])
        if chain_type == ChainType.Ligand:
            chains.append((mock_sequence_id(i), chain_type, seq))
            continue

        chains.append((mock_sequence_id(i), chain_type, seq))
    receptor_input = mock_peptiede_af3_input(task_id, chains, modified_info, bonds)

    if only_use_cached_msa_templates and cached_receptor_dir is not None:
        logger.debug(f"Only using cached MSA and templates from {cached_receptor_dir}")
        cached_receptor = load_base_input(Path(cached_receptor_dir))
        for chain in receptor_input.protein_chains:
            for cached_chain in cached_receptor.protein_chains:
                if chain.sequence == cached_chain.sequence:
                    chain._paired_msa = cached_chain._paired_msa
                    chain._unpaired_msa = cached_chain._unpaired_msa
                    chain._templates = cached_chain._templates

    processed_receptor_input = data_pipeline.process(receptor_input)
    if isinstance(user_ccd, dict):
        processed_receptor_input.user_ccd = user_ccd
    if isinstance(user_ccd, Path) and user_ccd.exists():
        processed_receptor_input.user_ccd = user_ccd
    if save_data and save_dir is not None:
        write_fold_input_json(processed_receptor_input, save_dir)

    return processed_receptor_input


# # old version without chain id sort
# def _appand_peptide_to_receptor(
#     receptor_input: Input,
#     peptide_sequence: str,
#     bonds: list[tuple[tuple[int, str], tuple[int, str]]] = [],
#     modifications: list[ModifiedResidueId] = [],
#     ligand_bonds: list[tuple[tuple[str, int, str], tuple[str, int, str]]] = [],
# ) -> tuple[Input, str]:
#     """
#     Appends a peptide to a receptor input.
#     """
#     chains_num = len(receptor_input.chains)
#     chains = list(receptor_input.chains)
#     new_receptor = copy.deepcopy(receptor_input)
#     ptms = []
#     for ptm in modifications:
#         ptms.append((ptm[1], int(ptm[0])))
#     intact_bonds = []
#     peptide_chain_id = mock_sequence_id(chains_num)
#     for bond in bonds:
#         intact_bonds.append(
#             (
#                 (peptide_chain_id, bond[0][0] + 1, bond[0][1]),
#                 (peptide_chain_id, bond[1][0] + 1, bond[1][1]),
#             )
#         )
#     peptide_chain = mock_chain(
#         sequence=peptide_sequence,
#         sequence_id=peptide_chain_id,
#         chain_type=ChainType.Protein,
#         smiles=None,
#         ptms=ptms,
#     )
#     peptide_chain._paired_msa = f">query\n{peptide_sequence}\n"
#     peptide_chain._unpaired_msa = f">query\n{peptide_sequence}\n"
#     peptide_chain._templates = tuple([])
#     chains.append(peptide_chain)
#     new_receptor.chains = tuple(chains)
#     if new_receptor.bonded_atom_pairs is None:
#         new_receptor.bonded_atom_pairs = intact_bonds + ligand_bonds
#     else:
#         new_receptor.bonded_atom_pairs = tuple(
#             list(new_receptor.bonded_atom_pairs) + intact_bonds + ligand_bonds
#         )
#     return new_receptor, peptide_chain_id


def appand_peptide_to_receptor(
    receptor_input: Input,
    peptide_sequence: str,
    bonds: list[tuple[tuple[int, str], tuple[int, str]]] = [],
    modifications: list[ModifiedResidueId] = [],
    ligand_bonds: list[tuple[tuple[int, str], tuple[int, str]]] = [],
    id_sort_dict: dict[str, str] = {},
) -> tuple[Input, str]:
    """
    Appends a peptide to a receptor input.
    """
    chains_num = len(receptor_input.chains)

    new_receptor = copy.deepcopy(receptor_input)
    chains = list(new_receptor.chains)
    ptms = []
    for ptm in modifications:
        ptms.append((ptm[1], int(ptm[0])))
    intact_bonds = []
    peptide_chain_id = mock_sequence_id(chains_num)
    for bond in bonds:
        intact_bonds.append(
            (
                (peptide_chain_id, bond[0][0] + 1, bond[0][1]),
                (peptide_chain_id, bond[1][0] + 1, bond[1][1]),
            )
        )
    peptide_chain = mock_chain(
        sequence=peptide_sequence,
        sequence_id=peptide_chain_id,
        chain_type=ChainType.Protein,
        smiles=None,
        ptms=ptms,
    )
    peptide_chain._paired_msa = f">query\n{peptide_sequence}\n"
    peptide_chain._unpaired_msa = f">query\n{peptide_sequence}\n"
    peptide_chain._templates = tuple([])
    chains.append(peptide_chain)
    chain_ids = [chain.id for chain in chains]
    resort_chains = []
    if len(id_sort_dict) > 0:
        for new_id, old_id in id_sort_dict.items():
            chain_idx = chain_ids.index(old_id)
            chain = chains[chain_idx]
            if isinstance(chain, Ligand):
                chain.id = new_id
            else:
                chain._id = new_id
            resort_chains.append(chain)

        new_receptor.chains = tuple(resort_chains)
    else:
        new_receptor.chains = tuple(chains)

    bonded_atom_pairs = ()
    if new_receptor.bonded_atom_pairs is None:
        bonded_atom_pairs = intact_bonds + ligand_bonds
    else:
        bonded_atom_pairs = tuple(
            list(new_receptor.bonded_atom_pairs) + intact_bonds + ligand_bonds
        )
    if len(id_sort_dict) > 0:
        revert_sort_ids = {v: k for k, v in id_sort_dict.items()}
        resort_bonds = []
        for bond in bonded_atom_pairs:
            atom1, atom2 = bond
            atom1 = (revert_sort_ids[atom1[0]], atom1[1], atom1[2])
            atom2 = (revert_sort_ids[atom2[0]], atom2[1], atom2[2])
            resort_bonds.append((atom1, atom2))
        bonded_atom_pairs = tuple(resort_bonds)
    new_receptor.bonded_atom_pairs = bonded_atom_pairs

    return new_receptor, peptide_chain_id


def get_af3_prediction(
    receptor_input: Input,
    peptide_sequence: str,
    model_runner: ModelRunner,
    output_dir: Path,
    bonds: list[tuple[tuple[int, str], tuple[int, str]]] = [],
    modifications: list[ModifiedResidueId] = [],
    ligand_bonds: list[tuple[tuple[str, int, str], tuple[str, int, str]]] = [],
    return_ids: bool = False,
    save_data: bool = False,
    id_sort_dict: dict[str, str] = {},
) -> str:
    """
    Get prediction from AlphaFold3 model.
    """
    logger.disable("af_finetune.utils.infer_utils")
    logger.disable("af_finetune.utils.data_utils")
    logger.disable("alphafold3.model.pipeline")
    logger.disable("alphafold3.model.features")

    new_receptor_input, peptide_chain_id = appand_peptide_to_receptor(
        receptor_input,
        peptide_sequence,
        bonds,
        modifications,
        ligand_bonds=ligand_bonds,
        id_sort_dict=id_sort_dict,
    )
    process_fold_input(
        new_receptor_input,
        data_pipeline_config=None,
        model_runner=model_runner,
        output_dir=output_dir,
        buckets=BUCKETS,
        save_data=save_data,
    )
    if return_ids:
        return peptide_chain_id, [chain.id for chain in new_receptor_input.chains]
    return peptide_chain_id


def get_af3_complex_prediction(
    receptor_input: Input,
    model_runner: ModelRunner,
    output_dir: Path,
    save_data: bool = False,
    id_sort_dict: dict[str, str] = {},
):
    chains = list(receptor_input.chains)
    chain_ids = [chain.id for chain in chains]
    resort_chains = []
    if len(id_sort_dict) > 0:
        for new_id, old_id in id_sort_dict.items():
            chain_idx = chain_ids.index(old_id)
            chain = chains[chain_idx]
            if isinstance(chain, Ligand):
                chain.id = new_id
            else:
                chain._id = new_id
            resort_chains.append(chain)
        receptor_input.chains = tuple(resort_chains)
    else:
        receptor_input.chains = tuple(chains)

    bonded_atom_pairs = receptor_input.bonded_atom_pairs
    if len(id_sort_dict) > 0:
        revert_sort_ids = {v: k for k, v in id_sort_dict.items()}
        resort_bonds = []
        for bond in bonded_atom_pairs:
            atom1, atom2 = bond
            atom1 = (revert_sort_ids[atom1[0]], atom1[1], atom1[2])
            atom2 = (revert_sort_ids[atom2[0]], atom2[1], atom2[2])
            resort_bonds.append((atom1, atom2))
        bonded_atom_pairs = tuple(resort_bonds)
    receptor_input.bonded_atom_pairs = bonded_atom_pairs
    process_fold_input(
        receptor_input,
        data_pipeline_config=None,
        model_runner=model_runner,
        output_dir=output_dir,
        buckets=BUCKETS,
        save_data=save_data,
    )


def convert_cif_to_pdb(cif_file: str) -> Structure | None:
    """
    Convert a CIF file to PDB format using Biopython.
    """
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("structure_cif", cif_file)
    io = PDBIO()
    io.set_structure(s)
    pdb_path = Path(cif_file).with_suffix(".pdb")
    io.save(str(pdb_path))
    if pdb_path.exists():
        return s[0]
    return None


if __name__ == "__main__":
    target_path = Path("/home/fuxin/lab/wwt/repos/alphafold3/tests/batch_test")
    task_tag = "epha5"

    receptor_data_path = Path(
        "/home/fuxin/lab/wwt/repos/alphafold3/tests/batch_test/epha5_data.json"
    )
    if receptor_data_path.exists():
        receptor_input = load_base_input(receptor_data_path)
    else:
        receptor_seq = [
            "IIGGEFTTIENQPWFAAIYRRHRGGSVTYVCGGSLISPCWVISATHCFIDYPKKEDYIVYLGRSRLNSNTQGEMKFEVENLILHKDYSADTLAHHNDIALLKIRSKEGRCAQPSRTIQTIALPSMYNDPQFGTSCEITGFGKEQSTDYLYPEQLKMTVVKLISHRECQQPHYYGSEVTTKMLCAADPQWKTDSCQGDSGGPLVCSLQGRMTLTGIVSWGRGCALKDKPGVYTRVSHFLPWIRSHT"
        ]
        receptor_input = make_base_input(
            task_tag, receptor_seq, [1], True, receptor_data_path.parent
        )

    model_runner = make_model_runner(
        device_index=0,
        model_dir="/data/soft/AF3-Model/",
        num_samples=5,
        use_bonds=True,
        use_offsets=False,
    )

    predict_sequences = [
        "CDDVIYIPEVGC",
        "CGEVDPETGEVC",
        "CDPLLEERIPGC",
        "CSYKGVEVLPGC",
        "CKDPLLAAVDPC",
    ]
    ids = [
        "rank_1",
        "rank_2",
        "rank_3",
        "rank_4",
        "rank_5",
    ]
    bonds = [((0, "SG"), (11, "SG"))]
    for i, (sequence, id) in enumerate(zip(predict_sequences, ids)):
        receptor_input.name = id
        cif_file = f"{str(target_path)}/{id}/{id.lower()}_model.cif"
        if not Path(cif_file).exists():

            peptide_chain_id = get_af3_prediction(
                receptor_input,
                sequence,
                model_runner,
                target_path / f"{id}",
                bonds=bonds,
                modifications=[],
                ligand_bonds=[],
            )

        predict_structure = convert_cif_to_pdb(cif_file)
