import json
from pathlib import Path
from typing import List
from alphafold3.common.folding_input import (
    BondAtomId,
    Input,
    ProteinChain,
    Template,
)
from loguru import logger

from alphafold3.utils.config import MODEL_SEEDS


from alphafold3.utils.data_utils import (
    ChainData,
    ChainType,
    ModifiedResidueId,
    mock_chain,
    read_file,
)
from alphafold3.utils.ccd_utils import (
    build_ligand_ccd_from_smiles,
    build_ligand_ccd_from_cdmol,
    build_ccd_from_csv,
    build_residue_ccd_from_cdmol,
    build_residue_ccd_from_smiles,
)


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except json.JSONDecodeError:
        print("错误: 无法解析 JSON 数据!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None


def parse_json_file(json_path: Path) -> Input | None:
    if json_path.exists():
        with open(json_path, "r") as f:
            json_str = f.read()
        fold_input = Input.from_json(json_str, json_path)
        return fold_input
    return None


def convert_cif_to_template(cif_path: Path, query_length: int):
    if not cif_path.exists():
        logger.warning(f"{cif_path} does not exist")
        return ()
    mmcif = read_file(cif_path)
    query_to_template_map = dict(zip(range(query_length), range(query_length)))

    return (Template(mmcif=mmcif, query_to_template_map=query_to_template_map),)


def handle_msa_and_template(
    af3_input: Input,
    use_msa: bool,
    use_template: bool,
    use_mock_template: bool,
    template_file: Path,
) -> Input:
    chains = []
    for chain in list(af3_input.chains):
        if not use_msa:
            if isinstance(chain, ProteinChain):
                chain._paired_msa = f">query\n{chain.sequence}\n"
            chain._unpaired_msa = f">query\n{chain.sequence}\n"
        if not use_template and isinstance(chain, ProteinChain):
            chain._templates = ()
        if (
            use_mock_template
            and template_file.exists()
            and isinstance(chain, ProteinChain)
        ):
            chain._templates = convert_cif_to_template(
                template_file, len(chain.sequence)
            )
        chains.append(chain)
    af3_input.chains = tuple(chains)
    return af3_input


def handle_ptms_with_custom_ccd(ptms=[]):
    modifitions = {}
    custom_ccd_dict = {}
    for ptm in ptms:
        chain_index, res_index, ptm_info, ptm_name = ptm
        chain_index_key = int(chain_index) - 1
        res_index_int = int(res_index)
        if chain_index_key not in modifitions.keys():
            modifitions[chain_index_key] = []
        if len(ptm_info) == 3 and ptm_info == ptm_name:

            modifitions[chain_index_key].append((res_index_int, ptm_name))
        else:
            ptm_like_path = Path(ptm_info)
            if ptm_like_path.exists():
                if ptm_like_path.suffix == ".csv":
                    custom_ccd_dict[ptm_name] = build_ccd_from_csv(
                        ptm_like_path, "PEPTIDE LINKING"
                    )
                if ptm_like_path.suffix == ".cdxml":
                    custom_ccd_dict[ptm_name] = build_residue_ccd_from_cdmol(
                        ptm_name, ptm_like_path
                    )
                modifitions[chain_index_key].append((res_index_int, ptm_name))
            else:
                custom_ccd_dict[ptm_name] = build_residue_ccd_from_smiles(
                    ptm_name, ptm_info
                )
                modifitions[chain_index_key].append((res_index_int, ptm_name))
    return modifitions, custom_ccd_dict


def handle_ligands_with_custom_ccd(ligand_chains=[]):
    custom_ccd_dict = {}
    sequences = []
    types = []
    for ligand_chain in ligand_chains:
        ligand_name = ligand_chain["name"]
        smiles = ligand_chain.get("smiles", None)
        cdxml = ligand_chain.get("cdxml", None)
        ccd_code = ligand_chain.get("ccd", None)
        csv_file = ligand_chain.get("csv", None)
        if smiles is not None:
            custom_ccd_dict[ligand_name] = build_ligand_ccd_from_smiles(
                ligand_name, smiles
            )
            sequences.append([ligand_name])
            types.append(ChainType.Ligand.value)

        if cdxml is not None:
            custom_ccd_dict[ligand_name] = build_ligand_ccd_from_cdmol(
                ligand_name, cdxml
            )
            sequences.append([ligand_name])
            types.append(ChainType.Ligand.value)

        if csv_file is not None:
            custom_ccd_dict[ligand_name] = build_ccd_from_csv(csv_file, "NON-POLYMER")
            sequences.append([ligand_name])
            types.append(ChainType.Ligand.value)

        if ccd_code is not None:
            sequences.append([ligand_name])
            types.append(ChainType.Ligand.value)

    return sequences, types, custom_ccd_dict


def mock_peptiede_af3_input(
    task_id: str,
    sequences: list[ChainData],
    modified_info: dict[int, List[ModifiedResidueId]],
    bond_pairs_info: List[tuple[BondAtomId, BondAtomId]] = [],
) -> Input:

    chains = []
    for i, chain in enumerate(sequences):
        chain_id, chain_type, sequence = chain
        ptms = []
        if i in modified_info.keys():
            for ptm in modified_info[i]:
                ptms.append((ptm[1], int(ptm[0])))
        chains.append(
            mock_chain(
                sequence=sequence,
                sequence_id=chain_id,
                chain_type=chain_type,
                smiles=None,
                ptms=ptms,
            )
        )

    return Input(
        name=task_id,
        chains=chains,
        rng_seeds=[MODEL_SEEDS],
        bonded_atom_pairs=bond_pairs_info,
        user_ccd=None,
    )
