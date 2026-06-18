import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from itertools import combinations
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from typing import Dict, Tuple, List, Optional, Union, Set
from loguru import logger

# --- Constants Definition ---
# Standard amino acid set (3-letter codes)
PROTEIN_RESIDUES: Set[str] = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

# Nucleic acid residue set (Includes DNA and RNA)
NUCLEIC_ACIDS: Set[str] = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}

# Combined set for token identification
VALID_RESIDUES: Set[str] = PROTEIN_RESIDUES | NUCLEIC_ACIDS


def write_scores_to_pdb_comment(
    pdb_input_path: str, pdb_output_path: str, scores_dict: dict
) -> None:
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("score_structure", pdb_input_path)

        if "raw_header" in structure.header and structure.header["raw_header"]:
            raw_header = structure.header["raw_header"]
        else:
            with open(pdb_input_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            header_lines = []
            for line in all_lines:
                if line.strip().startswith(("ATOM", "HETATM")):
                    break
                header_lines.append(line.rstrip("\n"))
            raw_header = "\n".join(header_lines)
        # ==========================================================

        raw_header_lines = raw_header.split("\n")
        cleaned_header_lines = []
        for line in raw_header_lines:
            if not (
                line.strip().startswith("COMMENT CUSTOM_SCORES_START")
                or line.strip().startswith("COMMENT CUSTOM_SCORES_END")
                or (
                    line.strip().startswith("COMMENT ")
                    and any(key in line for key in scores_dict.keys())
                )
            ):
                cleaned_header_lines.append(line)

        comment_lines = [
            "COMMENT CUSTOM_SCORES_START",
            "COMMENT ==============================",
        ]
        for key, value in scores_dict.items():
            score_line = f"COMMENT {key}: {value:.6f}"
            if len(score_line) > 80:
                score_line = score_line[:80]
            comment_lines.append(score_line)
        comment_lines.append("COMMENT ==============================")
        comment_lines.append("COMMENT CUSTOM_SCORES_END")

        insert_pos = len(cleaned_header_lines)
        for i, line in enumerate(cleaned_header_lines):
            if line.strip().startswith(("ATOM", "HETATM")):
                insert_pos = i
                break

        final_header_lines = (
            cleaned_header_lines[:insert_pos]
            + comment_lines
            + cleaned_header_lines[insert_pos:]
        )

        with open(pdb_input_path, "r", encoding="utf-8") as f:
            all_pdb_lines = f.readlines()

        atom_start_idx = 0
        for i, line in enumerate(all_pdb_lines):
            if line.strip().startswith(("ATOM", "HETATM")):
                atom_start_idx = i
                break

        new_header = "\n".join(final_header_lines) + "\n"
        new_pdb_content = new_header + "".join(all_pdb_lines[atom_start_idx:])

        with open(pdb_output_path, "w", encoding="utf-8") as f:
            f.write(new_pdb_content)
    except Exception as e:
        logger.error(f"Error occurred while writing PDB file: {e}")
        return


def read_scores_from_pdb_comment(pdb_file_path: str) -> dict:

    try:
        with open(pdb_file_path, "r", encoding="utf-8") as f:
            raw_header = ""
            for line in f:
                if line.strip().startswith(("ATOM", "HETATM")):
                    break
                raw_header += line
        # ==========================================================

        scores_dict = {}
        score_pattern = re.compile(r"COMMENT (\w+):\s*([-+]?\d+\.?\d*)")

        in_score_block = False
        for line in raw_header.split("\n"):
            line = line.strip()

            if line == "COMMENT CUSTOM_SCORES_START":
                in_score_block = True
                continue
            if line == "COMMENT CUSTOM_SCORES_END":
                in_score_block = False
                break

            if in_score_block and line.startswith("COMMENT ") and ":" in line:
                match = score_pattern.search(line)
                if match:
                    key = match.group(1)
                    value = float(match.group(2))
                    scores_dict[key] = value

        return scores_dict
    except Exception as e:
        logger.error(f"Error parsing scores from pdb file: {e}")
        return {}


def convert_cit_to_pdb(cif_file: str, pdb_file: str) -> None:
    """ """
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("structure_cif", cif_file)
    io = PDBIO()
    io.set_structure(s)
    io.save(str(pdb_file))


def parse_pdb_atom_line(line: str) -> Optional[Dict[str, Union[int, str]]]:
    """
    Parses a single ATOM or HETATM line from a PDB file.
    Returns a dictionary of parsed values or None if the line format is invalid.
    """
    if len(line) < 54:
        return None

    try:
        # PDB format uses fixed-column widths
        return {
            "atom_num": int(line[6:11]),
            "atom_name": line[12:16].strip(),
            "residue_name": line[17:20].strip(),
            "chain_id": line[21].strip(),
            "residue_seq_num": int(line[22:26]),
        }
    except ValueError:
        return None


def load_af3_pae_and_chains(
    json_path: Union[str, Path], pdb_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts PAE matrix, chain IDs, and residue types from AF3 output files.

    Logic:
    1. Parse PDB to identify residues corresponding to AF3 Tokens (typically CA or C1').
    2. Read JSON to retrieve the raw PAE matrix.
    3. Use the mask generated from PDB to slice the PAE matrix to valid residues.
    """
    json_path = Path(json_path)
    pdb_path = Path(pdb_path)

    # 1. Parse PDB to construct the token mask
    token_mask = []
    chains = []
    residue_types = []

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = parse_pdb_atom_line(line)
            if atom is None:
                continue

            atom_name = atom["atom_name"]
            res_name = atom["residue_name"]

            # Token Logic:
            # 1. Protein Alpha Carbons (CA) or Nucleic Acid C1' atoms -> Token=1 (Keep)
            if atom_name == "CA" or (res_name in NUCLEIC_ACIDS and "C1" in atom_name):
                token_mask.append(1)
                chains.append(atom["chain_id"])
                residue_types.append(res_name)

            # 2. Non-backbone atoms and non-standard residues (e.g., ligands/modifications)
            # would be marked as Token=0 if we needed to track them in the full PAE.
            # Standard residue side-chain atoms are ignored as they don't represent a Token.

    token_array = np.array(token_mask, dtype=bool)  # Convert to boolean for indexing
    chain_ids = np.array(chains)
    res_types = np.array(residue_types)

    # 2. Read JSON configuration
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Compatibility check for different AF3 JSON key naming conventions
    if "pae" in data:
        raw_pae = np.array(data["pae"])
    elif "predicted_aligned_error" in data:
        raw_pae = np.array(data["predicted_aligned_error"])
    else:
        # Handle cases where AF3 output might be a list containing the data
        if isinstance(data, list) and len(data) > 0 and "pae" in data[0]:
            raw_pae = np.array(data[0]["pae"])
        else:
            raise ValueError(f"Could not find 'pae' data in JSON file: {json_path}")

    # 3. Validation and Slicing
    n_tokens = len(token_mask)
    n_pae = raw_pae.shape[0]

    if n_tokens != n_pae:
        print(
            f"[Warning] Token count from PDB ({n_tokens}) does not match PAE dimensions ({n_pae})."
        )
        print(
            "This may cause slicing errors. Ensure the PDB contains all atoms and matches the model."
        )

        # Fallback: crop to the smallest common dimension to prevent hard crashes
        min_dim = min(n_tokens, n_pae)
        token_array = token_array[:min_dim]
        raw_pae = raw_pae[:min_dim, :min_dim]

    # Perform dual-axis slicing using boolean indexing to keep only identified residues
    filtered_pae = raw_pae[np.ix_(token_array, token_array)]

    return filtered_pae, chain_ids, res_types


def _calc_d0_array(L_array: np.ndarray, pair_type: str = "protein") -> np.ndarray:
    """Calculates the d0 normalization factor (vectorized)."""
    # L is clamped at a minimum of 27.0
    L = np.maximum(27.0, L_array.astype(float))

    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0

    # Formula: d0 = 1.24 * (L-15)^(1/3) - 1.8
    d0 = 1.24 * np.cbrt(L - 15.0) - 1.8
    return np.maximum(min_value, d0)


def _classify_chain_type(residue_types_subset: np.ndarray) -> str:
    """Classifies chain type: if it contains any nucleic acid residue, it's a nucleic_acid chain."""
    if np.isin(residue_types_subset, list(NUCLEIC_ACIDS)).any():
        return "nucleic_acid"
    return "protein"


def calculate_ipsae(
    pae_matrix: np.ndarray,
    chain_ids: np.ndarray,
    residue_types: Optional[np.ndarray] = None,
    pae_cutoff: float = 10.0,
) -> Dict[str, float]:
    """
    Calculates the ipSAE score.

    Optimization: Fully vectorized Mean PTM calculation, removing Python loops for better performance.
    """
    unique_chains = np.unique(chain_ids)
    scores = {}

    # Pre-determine chain types
    chain_type_map = {}
    if residue_types is not None:
        for chain in unique_chains:
            mask = chain_ids == chain
            chain_type_map[chain] = _classify_chain_type(residue_types[mask])
    else:
        for chain in unique_chains:
            chain_type_map[chain] = "protein"

    # Iterate through all chain pairs
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            # Determine interaction type for d0 calculation
            c1_type = chain_type_map[chain1]
            c2_type = chain_type_map[chain2]
            pair_type = (
                "nucleic_acid" if "nucleic_acid" in (c1_type, c2_type) else "protein"
            )

            # Extract sub-matrix for the pair
            mask_c1 = chain_ids == chain1
            mask_c2 = chain_ids == chain2

            # sub_pae shape: (N_residues_c1, N_residues_c2)
            sub_pae = pae_matrix[np.ix_(mask_c1, mask_c2)]

            if sub_pae.size == 0:
                scores[f"{chain1}_{chain2}"] = 0.0
                continue

            # 1. Identify valid interactions (contacts within cutoff)
            valid_mask = sub_pae < pae_cutoff  # Boolean matrix

            # 2. Calculate n0res (effective contact count per residue in Chain1)
            n0res_per_residue = np.sum(valid_mask, axis=1)

            # 3. Calculate d0 per residue
            d0_per_residue = _calc_d0_array(n0res_per_residue, pair_type)

            # 4. Calculate PTM matrix
            # Use broadcasting: (N, 1) against (N, M)
            ptm_matrix = 1.0 / (1.0 + (sub_pae / d0_per_residue[:, np.newaxis]) ** 2.0)

            # 5. Calculate ipSAE (Vectorized average)
            # We only average PTM values where valid_mask is True

            # Set invalid positions to 0 for the summation
            masked_ptm_sum = np.sum(ptm_matrix * valid_mask, axis=1)

            # Prevent division by zero for residues with no valid contacts
            with np.errstate(divide="ignore", invalid="ignore"):
                ipsae_per_residue = masked_ptm_sum / n0res_per_residue

            # Replace NaN (0/0 cases) with 0.0
            ipsae_per_residue = np.nan_to_num(ipsae_per_residue, nan=0.0)

            # 6. Take the maximum value as the final directional score
            final_score = (
                np.max(ipsae_per_residue) if ipsae_per_residue.size > 0 else 0.0
            )

            scores[f"{chain1}_{chain2}"] = float(final_score)

    return scores


if __name__ == "__main__":
    # Example paths
    pdb_file = Path(
        "/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b.pdb"
    )
    json_file = Path(
        "/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b/seed-10_sample-0/confidences.json"
    )

    if pdb_file.exists() and json_file.exists():
        try:
            print(f"Processing: {pdb_file} ...")
            pae, chains, res_types = load_af3_pae_and_chains(json_file, pdb_file)

            # Debugging output
            print(f"PAE shape: {pae.shape}, Chains shape: {chains.shape}")

            results = calculate_ipsae(pae, chains, res_types, pae_cutoff=10)

            print("\nipSAE Scores (Directional Chain A -> Chain B):")
            for pair_id, score in results.items():
                # Display results in "Chain1 -> Chain2: Score" format
                c1, c2 = pair_id.split("_")
                print(f"  {c1} -> {c2}: {score:.4f}")

        except Exception as e:
            print(f"Error during processing: {e}")
    else:
        print("Example files not found. Please check your file paths.")


def get_chains_from_pdb(pdb_path):
    """
    Extracts all unique chain IDs from a PDB file.

    Args:
        pdb_path (str): Path to the input PDB file.
    Returns:
        list: Sorted list of unique chain identifiers.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]  # Defaulting to the first model in the structure
    chains = [chain.id for chain in model.get_chains()]
    return sorted(set(chains))


def get_interface_res_from_pdb(pdb_file, chain1="A", chain2="B", dist_cutoff=10):
    """
    Identifies interface residues between two chains based on CA atom distances.

    Args:
        pdb_file (str): Path to the PDB file.
        chain1, chain2 (str): Chain IDs to compare.
        dist_cutoff (int): Distance threshold in Angstroms.
    Returns:
        tuple: (list of residues in chain1 interface, list of residues in chain2 interface)
    """
    chain_coords = defaultdict(dict)

    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                chain_id = line[21].strip()
                residue_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if atom_name == "CA":
                    chain_coords[chain_id][residue_id] = np.array([x, y, z])

    # Extract coordinates for the specified chains
    chain_1_res = sorted(chain_coords[chain1].keys())
    chain_2_res = sorted(chain_coords[chain2].keys())

    chain_1_coords = np.array([chain_coords[chain1][res] for res in chain_1_res])
    chain_2_coords = np.array([chain_coords[chain2][res] for res in chain_2_res])

    # Calculate pairwise Euclidean distance matrix
    # Using broadcasting for efficiency: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    dist = np.sqrt(
        np.sum(
            (chain_1_coords[:, None, :] - chain_2_coords[None, :, :]) ** 2,
            axis=2,
        )
    )
    interface_residues = np.where(dist < dist_cutoff)

    interface_1 = sorted(set(chain_1_res[i] for i in interface_residues[0]))
    interface_2 = sorted(set(chain_2_res[i] for i in interface_residues[1]))

    return interface_1, interface_2


def extract_token_chain_and_res_ids(pdb_file):
    """
    Extracts token-level chain IDs and residue IDs from a PDB file.
    Each token corresponds to one residue containing a CA atom.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    token_chain_ids = []
    token_res_ids = []

    for chain in model:
        for residue in chain:
            if "CA" in residue:  # Only count residues with a protein backbone CA atom
                token_chain_ids.append(chain.id)
                token_res_ids.append(
                    residue.id[1]
                )  # residue.id is (hetfield, resseq, icode)

    return token_chain_ids, token_res_ids


def parse_confidences_json(conf_path, pdb_path):
    """
    Parses AlphaFold3 confidence files and calculates chain-wise and interface PAE.

    Args:
        conf_path (str): Path to confidences.json.
        pdb_path (str): Path to the predicted structure.
    Returns:
        tuple: (intra-chain PAE, interface-residue PAE, inter-chain PAE)
    """
    with open(conf_path) as f:
        conf = json.load(f)

    chains = get_chains_from_pdb(pdb_path)
    pae = np.array(conf["pae"])
    token_chain_ids, token_res_ids = extract_token_chain_and_res_ids(pdb_path)

    # Map chain IDs to their respective indices in the PAE matrix
    chain_indices = {chain: [] for chain in chains}
    for i, chain in enumerate(token_chain_ids):
        chain_indices[chain].append(i)

    # Calculate average intra-chain PAE
    chain_pae = {
        chain: float(np.mean(pae[np.ix_(idxs, idxs)]))
        for chain, idxs in chain_indices.items()
    }

    ipae = {}
    pae_interaction = {}

    # Process pairwise interface PAE and inter-chain PAE
    for ch1, ch2 in combinations(chains, 2):
        try:
            # 1. Interface-specific PAE (based on distance cutoff)
            idx1_res, idx2_res = get_interface_res_from_pdb(
                pdb_path, chain1=ch1, chain2=ch2
            )
            idx1 = [
                i
                for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                if chain == ch1 and res_id in idx1_res
            ]
            idx2 = [
                i
                for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                if chain == ch2 and res_id in idx2_res
            ]

            pair_key = f"{ch1}_{ch2}"

            if idx1 and idx2:
                # Average PAE of residues at the structural interface
                ipae[pair_key] = np.mean(
                    [
                        np.mean(pae[np.ix_(idx1, idx2)]),
                        np.mean(pae[np.ix_(idx2, idx1)]),
                    ]
                )

            # 2. General Inter-chain PAE (all residues between two chains)
            chain_1_indices = [
                i for i, chain in enumerate(token_chain_ids) if chain == ch1
            ]
            chain_2_indices = [
                i for i, chain in enumerate(token_chain_ids) if chain == ch2
            ]

            pae_interaction[pair_key] = np.mean(
                [
                    np.mean(pae[np.ix_(chain_1_indices, chain_2_indices)]),
                    np.mean(pae[np.ix_(chain_2_indices, chain_1_indices)]),
                ]
            )

        except Exception as e:
            print(f"[Warning] Failed to process pair ({ch1}, {ch2}): {e}")

    return chain_pae, ipae, pae_interaction


def process_single_description(conf_path, pdb_path, summary_path):
    """
    Worker function to process all metrics for a single prediction directory.
    """
    try:
        # Construct directory and file paths

        # Validate existence of required files
        if not summary_path.exists():
            return None, " missing summary file"
        if not pdb_path.exists():
            return None, " missing pdb file"
        if not conf_path.exists():
            return None, " missing conf file"

        # Calculate ipSAE (interface Predicted Structural Alignment Error)
        ipsae_metrics = {}
        pae_matrix, chain_ids, residue_types = load_af3_pae_and_chains(
            conf_path, pdb_path
        )
        ipsae_dict = calculate_ipsae(
            pae_matrix, chain_ids, residue_types, pae_cutoff=10
        )
        for k, v in ipsae_dict.items():
            ipsae_metrics[f"ipsae_{k}"] = v

        # Load AlphaFold3 confidence data
        summary = json.loads(summary_path.read_text())
        conf = json.loads(conf_path.read_text())
        chains = get_chains_from_pdb(pdb_path)

        # Map chain-level ipTM and PTM scores
        iptm = dict(zip(chains, summary.get("chain_iptm", [])))
        ptm = dict(zip(chains, summary.get("chain_ptm", [])))

        # Process inter-chain pair ipTM matrix
        iptm_matrix = summary["chain_pair_iptm"]
        interchain_iptm_dict = {}
        num_chains = len(chains)
        for i in range(num_chains):
            for j in range(i + 1, num_chains):
                interchain_iptm_dict[f"iptm_{chains[i]}_{chains[j]}"] = iptm_matrix[i][
                    j
                ]

        # Calculate pLDDT scores
        atom_plddts = conf["atom_plddts"]
        atom_chain_ids = conf["atom_chain_ids"]
        # Per-chain average pLDDT
        chain_plddt = {
            ch: float(
                np.mean(
                    [pl for pl, cid in zip(atom_plddts, atom_chain_ids) if cid == ch]
                )
            )
            for ch in chains
        }
        # Overall complex pLDDT
        complex_plddt = float(np.mean(list(chain_plddt.values())))

        # Extract PAE-related metrics
        chain_pae, ipae, inter_pae = parse_confidences_json(conf_path, str(pdb_path))

        result = {"plddt": complex_plddt}

        for pair_key, value in ipae.items():
            result[f"ipae_{pair_key}"] = float(value)

        for pair_key, value in inter_pae.items():
            result[f"inter_pae_{pair_key}"] = float(value)

        for ch in chains:
            result[f"chain_{ch}_plddt"] = chain_plddt.get(ch, np.nan)
            result[f"chain_{ch}_pae"] = chain_pae.get(ch, np.nan)
            result[f"chain_{ch}_ptm"] = ptm.get(ch, np.nan)
            result[f"chain_{ch}_iptm"] = iptm.get(ch, np.nan)

        result.update(ipsae_metrics)
        result.update(interchain_iptm_dict)

        return result

    except Exception as e:
        logger.error(f"Error in parse_pdb_metrics: {e}")
        return {}
