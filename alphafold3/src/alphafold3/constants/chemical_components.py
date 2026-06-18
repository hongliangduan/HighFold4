# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Chemical Components found in PDB (CCD) constants."""

from collections.abc import ItemsView, Iterator, KeysView, Mapping, Sequence, ValuesView
import dataclasses
import functools
import os
import csv
from pathlib import Path

from alphafold3.constants import residue_names
from alphafold3.common import resources
from alphafold3.common import safe_pickle
from alphafold3.cpp import cif_dict
from alphafold3.utils.ccd_utils import (
    build_ligand_ccd_from_smiles,
    build_ligand_ccd_from_cdmol,
    build_ccd_from_csv,
    build_residue_ccd_from_cdmol,
    build_residue_ccd_from_smiles,
)

_CCD_PICKLE_FILE = resources.filename(
    resources.ROOT / "constants/converters/ccd.pickle"
)


def read_csv_file(file_path):
    data = []
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter=",")
            headers = next(tsv_reader, None)
            if headers:
                data.append(headers)
                for row in tsv_reader:
                    data.append(row)
        return data

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
    return None


def get_ccd_from_user_file(file_path: Path):
    ccd_data = []
    if file_path.suffix == ".csv":
        ccd_data = read_csv_file(file_path)
    else:
        print(f"错误: 不支持的文件类型 {file_path.suffix}")
        return None

    ccd_dict = {}
    for key, val in zip(ccd_data[0], ccd_data[1]):
        ccd_dict[key] = tuple(val.strip().split(", "))
    ccd_name = ccd_dict.pop("Key")[0]
    # ccd_dict["_chem_comp_atom.atom_id"] = ccd_dict[
    #     "_chem_comp_atom.pdbx_component_atom_id"
    # ]
    ccd_dict["_chem_comp.type"] = ("PEPTIDE LINKING",)

    return ccd_name, ccd_dict


@functools.cache
def _load_ccd_pickle_cached(
    path: os.PathLike[str],
) -> dict[str, Mapping[str, Sequence[str]]]:
    """Loads the CCD pickle file and caches it so that it is only loaded once."""
    with open(path, "rb") as f:
        return safe_pickle.load(f)


class Ccd(Mapping[str, Mapping[str, Sequence[str]]]):
    """Chemical Components found in PDB (CCD) constants.

    See https://academic.oup.com/bioinformatics/article/31/8/1274/212200 for CCD
    CIF format documentation.

    Wraps the dict to prevent accidental mutation.
    """

    __slots__ = ("_dict", "_ccd_pickle_path")

    def __init__(
        self,
        ccd_pickle_path: os.PathLike[str] | None = None,
        user_ccd: str | None = None,
    ):
        """Initialises the chemical components dictionary.

        Args:
          ccd_pickle_path: Path to the CCD pickle file. If None, uses the default
            CCD pickle file included in the source code.
          user_ccd: A string containing the user-provided CCD. This has to conform
            to the same format as the CCD, see https://www.wwpdb.org/data/ccd. If
            provided, takes precedence over the CCD for the the same key. This can
            be used to override specific entries in the CCD if desired.
        """
        self._ccd_pickle_path = ccd_pickle_path or _CCD_PICKLE_FILE
        self._dict = _load_ccd_pickle_cached(self._ccd_pickle_path)

        if user_ccd is not None:
            appix_dict = {}
            if isinstance(user_ccd, dict):

                appix_dict.update(self._extend_user_ccd_dict(user_ccd))
            if isinstance(user_ccd, str):
                user_ccd_path = Path(user_ccd)
                if user_ccd_path.is_dir():
                    user_ccd_files = user_ccd_path.glob("*")
                    for user_ccd_file in user_ccd_files:
                        if not user_ccd_file.is_file():
                            continue
                        appix_dict.update(self._extend_user_ccd_files(user_ccd_file))
                if user_ccd_path.is_file():
                    appix_dict.update(self._extend_user_ccd_files(user_ccd_path))

            self._dict.update(appix_dict)

            for key, value in appix_dict.items():
                if value["_chem_comp.type"][0] == "NON-POLYMER":
                    continue
                if "_chem_comp.mon_nstd_parent_comp_id" in value.keys():
                    parent_comp_id = value["_chem_comp.mon_nstd_parent_comp_id"][0]
                    if key in residue_names.CCD_NAME_TO_ONE_LETTER:
                        continue
                    if parent_comp_id == "?":
                        residue_names.CCD_NAME_TO_ONE_LETTER[key] = "X"
                    if parent_comp_id in residue_names.CCD_NAME_TO_ONE_LETTER:
                        residue_names.CCD_NAME_TO_ONE_LETTER[key] = (
                            residue_names.CCD_NAME_TO_ONE_LETTER[parent_comp_id]
                        )

    def _extend_user_ccd_dict(
        self, extra_dict: dict[str, Mapping[str, Sequence[str]]]
    ) -> dict[str, Mapping[str, Sequence[str]]]:
        "_chem_comp.id"
        appix_dict = {}
        for key, value in extra_dict.items():
            if "_chem_comp.id" in value.keys():
                appix_dict[key] = value
            if "smiles" in value.keys():
                mol_type = value["type"]
                if mol_type == "ligand":
                    appix_dict[key] = build_ligand_ccd_from_smiles(key, value["smiles"])
                if mol_type == "ptm":
                    appix_dict[key] = build_residue_ccd_from_smiles(
                        key, value["smiles"]
                    )
            if "cdxml" in value.keys():
                mol_type = value["type"]
                if mol_type == "ligand":
                    appix_dict[key] = build_ligand_ccd_from_cdmol(key, value["cdxml"])
                if mol_type == "ptm":
                    appix_dict[key] = build_residue_ccd_from_cdmol(key, value["cdxml"])
            if "csv" in value.keys():
                mol_type = value["type"]
                ccd_type = "PEPTIDE LINKING" if mol_type == "ptm" else "NON-POLYMER"
                appix_dict[key] = build_ccd_from_csv(value["csv"], ccd_type)

        return appix_dict

    def _extend_user_ccd_files(self, user_ccd_file: Path) -> None:
        """Extends the user CCD files with the given user CCD.

        Args:
          user_ccd_file: A string containing the user-provided CCD. This has to conform
            to the same format as the CCD, see https://www.wwpdb.org/data/ccd. If
            provided, takes precedence over the CCD for the the same key. This can
            be used to override specific entries in the CCD if desired.
        """
        if not user_ccd_file:
            raise ValueError("User CCD cannot be an empty string.")

        appix_dict = {}

        if user_ccd_file.suffix == ".cif":
            appix_dict = {
                key: {k: tuple(v) for k, v in value.items()}
                for key, value in cif_dict.parse_multi_data_cif(
                    user_ccd_file.read_text()
                ).items()
            }
        if user_ccd_file.suffix == ".csv":
            user_ccd_name, user_ccd_data = get_ccd_from_user_file(user_ccd_file)
            appix_dict[user_ccd_name] = user_ccd_data

        return appix_dict

    def __getitem__(self, key: str) -> Mapping[str, Sequence[str]]:
        return self._dict[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dict

    def __iter__(self) -> Iterator[str]:
        return self._dict.__iter__()

    def __len__(self) -> int:
        return len(self._dict)

    def __hash__(self) -> int:
        return id(self)  # Ok since this is immutable.

    def get(
        self, key: str, default: None | Mapping[str, Sequence[str]] = None
    ) -> Mapping[str, Sequence[str]] | None:
        return self._dict.get(key, default)

    def items(self) -> ItemsView[str, Mapping[str, Sequence[str]]]:
        return self._dict.items()

    def values(self) -> ValuesView[Mapping[str, Sequence[str]]]:
        return self._dict.values()

    def keys(self) -> KeysView[str]:
        return self._dict.keys()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ComponentInfo:
    name: str
    type: str
    pdbx_synonyms: str
    formula: str
    formula_weight: str
    mon_nstd_parent_comp_id: str
    mon_nstd_flag: str
    pdbx_smiles: str


def mmcif_to_info(mmcif: Mapping[str, Sequence[str]]) -> ComponentInfo:
    """Converts CCD mmCIFs to component info. Missing fields are left empty."""
    names = mmcif["_chem_comp.name"]
    types = mmcif["_chem_comp.type"]
    mon_nstd_parent_comp_ids = mmcif["_chem_comp.mon_nstd_parent_comp_id"]
    pdbx_synonyms = mmcif["_chem_comp.pdbx_synonyms"]
    formulas = mmcif["_chem_comp.formula"]
    formula_weights = mmcif["_chem_comp.formula_weight"]

    def front_or_empty(values: Sequence[str]) -> str:
        return values[0] if values else ""

    type_ = front_or_empty(types)
    mon_nstd_parent_comp_id = front_or_empty(mon_nstd_parent_comp_ids)
    if type_.lower() == "non-polymer":
        # Unset for non-polymers, e.g. water or ions.
        mon_nstd_flag = "."
    elif mon_nstd_parent_comp_id == "?":
        # A standard component - it doesn't have a standard parent, e.g. MET.
        mon_nstd_flag = "y"
    else:
        # A non-standard component, e.g. MSE.
        mon_nstd_flag = "n"

    # Default SMILES is the canonical SMILES, but we fall back to the SMILES if a
    # canonical SMILES is not available. Of canonical SMILES, we prefer ones from
    # the OpenEye OEToolkits program.
    canonical_pdbx_smiles = ""
    fallback_pdbx_smiles = ""
    descriptor_types = mmcif.get("_pdbx_chem_comp_descriptor.type", [])
    descriptors = mmcif.get("_pdbx_chem_comp_descriptor.descriptor", [])
    programs = mmcif.get("_pdbx_chem_comp_descriptor.program", [])
    for descriptor_type, descriptor, program in zip(
        descriptor_types, descriptors, programs
    ):
        if descriptor_type == "SMILES_CANONICAL":
            if (not canonical_pdbx_smiles) or program == "OpenEye OEToolkits":
                canonical_pdbx_smiles = descriptor
        if not fallback_pdbx_smiles and descriptor_type == "SMILES":
            fallback_pdbx_smiles = descriptor
    pdbx_smiles = canonical_pdbx_smiles or fallback_pdbx_smiles

    return ComponentInfo(
        name=front_or_empty(names),
        type=type_,
        pdbx_synonyms=front_or_empty(pdbx_synonyms),
        formula=front_or_empty(formulas),
        formula_weight=front_or_empty(formula_weights),
        mon_nstd_parent_comp_id=mon_nstd_parent_comp_id,
        mon_nstd_flag=mon_nstd_flag,
        pdbx_smiles=pdbx_smiles,
    )


@functools.lru_cache(maxsize=128)
def component_name_to_info(ccd: Ccd, res_name: str) -> ComponentInfo | None:
    component = ccd.get(res_name)
    if component is None:
        return None
    return mmcif_to_info(component)


def type_symbol(ccd: Ccd, res_name: str, atom_name: str) -> str:
    """Returns the element type for the given component name and atom name.

    Args:
      ccd: The chemical components dictionary.
      res_name: The component name, e.g. ARG.
      atom_name: The atom name, e.g. CB, OXT, or NH1.

    Returns:
      Element type, e.g. C for (ARG, CB), O for (ARG, OXT), N for (ARG, NH1).
    """
    res = ccd.get(res_name)
    if res is None:
        return "?"
    try:
        return res["_chem_comp_atom.type_symbol"][
            res["_chem_comp_atom.atom_id"].index(atom_name)
        ]
    except (ValueError, IndexError, KeyError):
        return "?"
