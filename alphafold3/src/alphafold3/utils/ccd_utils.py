import re
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Deque
from collections import deque, defaultdict
from rdkit.Chem.rdmolfiles import SDMolSupplier  # removed PDBWriter
from rdkit.Chem.rdmolfiles import SDWriter
from rdkit.Geometry import Point3D
from alphafold3.data.tools import rdkit_utils
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdMD
from loguru import logger

# SMILES for 20 standard amino acids (side chains in neutral form)
AMINO_ACIDS = {
    "ALA": "N[C@@H](C)C(=O)O",
    "ARG": "N[C@@H](CCCNC(N)=N)C(=O)O",
    "ASN": "N[C@@H](CC(=O)N)C(=O)O",
    "ASP": "N[C@@H](CC(=O)O)C(=O)O",
    "CYS": "N[C@@H](CS)C(=O)O",
    "GLN": "N[C@@H](CCC(=O)N)C(=O)O",
    "GLU": "N[C@@H](CCC(=O)O)C(=O)O",
    "GLY": "NCC(=O)O",
    "HIS": "N[C@@H](CC1=CN=CN1)C(=O)O",
    "ILE": "N[C@@H](C(C)CC)C(=O)O",
    "LEU": "N[C@@H](CC(C)C)C(=O)O",
    "LYS": "N[C@@H](CCCCN)C(=O)O",
    "MET": "N[C@@H](CCSC)C(=O)O",
    "PHE": "N[C@@H](CC1=CC=CC=C1)C(=O)O",
    "PRO": "N1CCC[C@H]1C(=O)O",
    "SER": "N[C@@H](CO)C(=O)O",
    "THR": "N[C@@H](C(O)C)C(=O)O",
    "TRP": "N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)O",
    "TYR": "N[C@@H](CC1=CC=C(O)C=C1)C(=O)O",
    "VAL": "N[C@@H](C(C)C)C(=O)O",
}


GREEK_LEVEL = ["B", "G", "D", "E", "Z", "H", "I", "K", "L", "M"]

# AlphaFold3 ATOM37 compatibility support
ATOM37_NAMES = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

ATOM37_ORDER = {name: i for i, name in enumerate(ATOM37_NAMES)}

# Standard amino-acid atom templates
STANDARD_RESIDUE_ATOMS = {
    "ALA": ("C", "CA", "CB", "N", "O"),
    "ARG": ("C", "CA", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "N", "O"),
    "ASN": ("C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"),
    "ASP": ("C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"),
    "CYS": ("C", "CA", "CB", "N", "O", "SG"),
    "GLN": ("C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"),
    "GLU": ("C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"),
    "GLY": ("C", "CA", "N", "O"),
    "HIS": ("C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"),
    "ILE": ("C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"),
    "LEU": ("C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"),
    "LYS": ("C", "CA", "CB", "CG", "CD", "CE", "NZ", "N", "O"),
    "MET": ("C", "CA", "CB", "CG", "CE", "N", "O", "SD"),
    "PHE": ("C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"),
    "PRO": ("C", "CA", "CB", "CG", "CD", "N", "O"),
    "SER": ("C", "CA", "CB", "N", "O", "OG"),
    "THR": ("C", "CA", "CB", "CG2", "N", "O", "OG1"),
    "TRP": (
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ),
    "TYR": ("C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "N", "O"),
    "VAL": ("C", "CA", "CB", "CG1", "CG2", "N", "O"),
    # More amino-acid templates can be added...
}


def sanitize_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(s).strip()) or "NONAME"


def build_3d(smi: str, seed: int, do_opt: str | None):
    """Return a 3D molecule (with explicit hydrogens). do_opt: None / 'uff' / 'mmff'"""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, "parse_failed"
    m = Chem.AddHs(m)  # 3D: keep explicit hydrogens
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = False
    params.enforceChirality = True  # preserve chirality in SMILES
    if AllChem.EmbedMolecule(m, params) != 0:
        return None, "embed_failed"

    if do_opt == "uff":
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
    elif do_opt == "mmff":
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass

    return m, ""


def _is_carbonyl_double_oxygen(b: Chem.Bond, nbr: Chem.Atom) -> bool:
    return nbr.GetAtomicNum() == 8 and b.GetBondType() == Chem.BondType.DOUBLE


def _is_carboxylate_oxygen(b: Chem.Bond, nbr: Chem.Atom) -> bool:
    return (
        nbr.GetAtomicNum() == 8
        and b.GetBondType() == Chem.BondType.SINGLE
        and nbr.GetFormalCharge() <= 0
    )


def find_backbone_atoms(
    mol: Chem.Mol,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    """Return (idx_N, idx_CA, idx_C, idx_O, idx_OXT).

    Improvements:
    - C is still the “carboxyl-like carbon”;
    - CA is no longer required to be a direct carbon neighbor of C,
      but is searched via BFS starting from C along “C–C bonds” within at most 3 steps, looking for a “carbon attached to N”:
        · distance 1 → α-amino acid
        · distance 2 → β-amino acid
        · distance 3 → γ-amino acid
    """
    backbone_candidates = []

    logger.debug("=== Analyze molecular bond connectivity ===")
    for i, atom in enumerate(mol.GetAtoms()):
        bonds_info = []
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            bonds_info.append(f"{neighbor.GetIdx()}({neighbor.GetSymbol()})")
        logger.debug(f"Atom {i} ({atom.GetSymbol()}): connected to {bonds_info}")

    for c_atom in mol.GetAtoms():
        if c_atom.GetAtomicNum() != 6:
            continue

        # === 1. Identify carboxyl-like C ===
        o_double = []
        o_single = []
        has_o = False
        for bond in c_atom.GetBonds():
            a = bond.GetOtherAtom(c_atom)
            if a.GetAtomicNum() != 8:
                continue
            has_o = True
            if _is_carbonyl_double_oxygen(bond, a):
                o_double.append(a.GetIdx())
            elif _is_carboxylate_oxygen(bond, a):
                o_single.append(a.GetIdx())
        if not has_o:
            continue

        carboxyl_like = (len(o_double) == 1) or (
            len(o_double) == 0 and len(o_single) >= 2
        )
        if not carboxyl_like:
            continue

        c_idx = c_atom.GetIdx()
        logger.debug(
            f"Found carboxyl-like carbon: atom {c_idx}, double-bond O: {o_double}, single-bond O: {o_single}"
        )

        # O / OXT decision (same as before)
        idx_O = None
        idx_OXT = None
        if len(o_double) >= 1:
            idx_O = o_double[0]
            if len(o_single) >= 1:
                idx_OXT = o_single[0]
        elif len(o_single) >= 2:
            idx_O, idx_OXT = o_single[0], o_single[1]
        elif len(o_single) == 1:
            idx_O = o_single[0]

        # BFS from carboxyl C along C–C bonds, up to 3 steps, to find a carbon attached to N as CA
        max_dist = 3  # α/β/γ
        visited: Set[int] = {c_idx}
        q: Deque[Tuple[int, int]] = deque()

        # Start from direct carbon neighbors
        for bond in c_atom.GetBonds():
            nb = bond.GetOtherAtom(c_atom)
            if nb.GetAtomicNum() != 6:
                continue
            nb_idx = nb.GetIdx()
            visited.add(nb_idx)
            q.append((nb_idx, 1))

        while q:
            ca_idx, dist = q.popleft()
            ca = mol.GetAtomWithIdx(ca_idx)

            logger.debug(
                f"  [Backbone C {c_idx}] BFS to carbon {ca_idx}, distance {dist}"
            )

            # 2.1 Is this carbon attached to N?
            n_neighs = []
            for bond in ca.GetBonds():
                n_atom = bond.GetOtherAtom(ca)
                if n_atom.GetAtomicNum() == 7:
                    n_neighs.append(n_atom)

            if n_neighs:
                logger.debug(
                    f"    As CA candidate: {ca_idx}, attached N atoms {[n.GetIdx() for n in n_neighs]}, dist={dist}"
                )

                # Score N: still prefer NH2-like (two hydrogens)
                best_n_atom = None
                best_n_score = float("inf")
                for n_atom in n_neighs:
                    n_idx = n_atom.GetIdx()
                    h_count = 0
                    heavy_count = 0
                    for n_bond in n_atom.GetBonds():
                        n_nb = n_bond.GetOtherAtom(n_atom)
                        if n_nb.GetAtomicNum() == 1:
                            h_count += 1
                        elif n_nb.GetAtomicNum() > 1:
                            heavy_count += 1
                    h_score = abs(h_count - 2)
                    heavy_score = heavy_count * 0.1
                    n_score = h_score + heavy_score
                    logger.debug(
                        f"      N atom {n_idx}: H={h_count}, heavy={heavy_count}, score={n_score:.2f}"
                    )
                    if n_score < best_n_score:
                        best_n_score = n_score
                        best_n_atom = n_atom

                if best_n_atom is None:
                    logger.debug("No qualified N atom; skip this CA candidate")
                else:
                    # CA hybridization preference: sp3 looks more backbone-like
                    hyb_penalty = (
                        0.0
                        if ca.GetHybridization() == Chem.HybridizationType.SP3
                        else 0.2
                    )
                    # Distance penalty: α(1) best, then β(2), then γ(3)
                    dist_penalty = 0.3 * (dist - 1)
                    score = best_n_score + hyb_penalty + dist_penalty

                    logger.debug(
                        f"Record backbone candidate: N={best_n_atom.GetIdx()}, CA={ca_idx}, C={c_idx}, "
                        f"O={idx_O}, OXT={idx_OXT}, dist={dist}, total score={score:.2f}"
                    )

                    backbone_candidates.append(
                        (best_n_atom.GetIdx(), ca_idx, c_idx, idx_O, idx_OXT, score)
                    )

            # 2.2 Expand further along carbon chain (only C–C, up to 3 steps)
            if dist >= max_dist:
                continue

            for bond in ca.GetBonds():
                nb = bond.GetOtherAtom(ca)
                nb_idx = nb.GetIdx()
                if nb_idx in visited:
                    continue
                if nb_idx == c_idx:
                    continue
                if nb.GetAtomicNum() != 6:
                    continue
                visited.add(nb_idx)
                q.append((nb_idx, dist + 1))

    # 3. Pick the lowest-score candidate as backbone
    if not backbone_candidates:
        logger.warning("No backbone candidates found (may not be an α/β/γ amino acid)")
        return None, None, None, None, None

    logger.debug("\n=== Backbone candidate evaluation ===")
    for i, (n_idx, ca_idx, c_idx, o_idx, oxt_idx, score) in enumerate(
        backbone_candidates
    ):
        logger.debug(
            f"Candidate {i+1}: N={n_idx}, CA={ca_idx}, C={c_idx}, O={o_idx}, OXT={oxt_idx}, score={score:.2f}"
        )

    idx_N, idx_CA, idx_C, idx_O, idx_OXT, _ = min(
        backbone_candidates, key=lambda x: x[-1]
    )
    logger.debug(
        f"Selected backbone: N={idx_N}, CA={idx_CA}, C={idx_C}, O={idx_O}, OXT={idx_OXT}"
    )
    return idx_N, idx_CA, idx_C, idx_O, idx_OXT


def choose_CB(mol: Chem.Mol, idx_CA: int, main_set: Set[int]) -> Optional[int]:
    """Choose CB from CA's directly connected carbon neighbors: exclude backbone atoms; if multiple, prefer the larger carbon branch."""
    cand: List[int] = []
    ca = mol.GetAtomWithIdx(idx_CA)

    logger.debug(f"\n=== Choose CB: from neighbors of CA({idx_CA}) ===")
    # Traverse all atoms directly connected to CA
    for bond in ca.GetBonds():
        neighbor = bond.GetOtherAtom(ca)
        neighbor_idx = neighbor.GetIdx()
        logger.debug(f"  CA connected to: {neighbor_idx} ({neighbor.GetSymbol()})")

        if neighbor.GetAtomicNum() != 6:  # only carbon atoms
            logger.debug(f"    Skip non-carbon: {neighbor_idx}")
            continue
        if neighbor_idx in main_set:  # exclude backbone atoms
            logger.debug(f"    Skip backbone atom: {neighbor_idx}")
            continue
        cand.append(neighbor_idx)
        logger.debug(f"    Candidate CB: {neighbor_idx}")

    if not cand:
        logger.debug("  No CB candidates found")
        return None

    # Sort by "reachable carbon count": larger is better
    def carbon_reach(start: int) -> int:
        seen: Set[int] = set(main_set) | {idx_CA}
        q: Deque[int] = deque([start])
        cnt = 0
        while q:
            u = q.popleft()
            if u in seen:
                continue
            seen.add(u)
            atom = mol.GetAtomWithIdx(u)
            if atom.GetAtomicNum() == 6:
                cnt += 1
            # Use bond connectivity info instead of distance
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetIdx() not in seen:
                    q.append(neighbor.GetIdx())
        return cnt

    # Compute weights for CB candidates
    cb_weights = []
    for cb_idx in cand:
        weight = carbon_reach(cb_idx)
        cb_weights.append((cb_idx, weight))
        logger.debug(f"    Carbon-chain weight for CB candidate {cb_idx}: {weight}")

    # Choose the highest-weight one as CB
    cb_weights.sort(
        key=lambda x: (x[1], mol.GetAtomWithIdx(x[0]).GetDegree()), reverse=True
    )
    selected_cb = cb_weights[0][0]
    logger.debug(f"  Selected CB: {selected_cb}")
    return selected_cb


# Remove reliance on _get_conf and distance calculations
def _get_conf(mol: Chem.Mol):
    """Only generate a conformer when coordinates are needed for output."""
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol.GetConformer()


def assign_names(mol: Chem.Mol) -> Dict[int, str]:
    idx_N, idx_CA, idx_C, idx_O, idx_OXT = find_backbone_atoms(mol)
    name_map: Dict[int, str] = {}
    main_set: Set[int] = set()

    logger.debug("\n=== Backbone atom identification ===")
    if idx_N is not None:
        name_map[idx_N] = "N"
        main_set.add(idx_N)
        logger.debug(f"Backbone atom: index {idx_N} -> N")
    if idx_CA is not None:
        name_map[idx_CA] = "CA"
        main_set.add(idx_CA)
        logger.debug(f"Backbone atom: index {idx_CA} -> CA")
    if idx_C is not None:
        name_map[idx_C] = "C"
        main_set.add(idx_C)
        logger.debug(f"Backbone atom: index {idx_C} -> C")
    if idx_O is not None:
        name_map[idx_O] = "O"
        main_set.add(idx_O)
        logger.debug(f"Backbone atom: index {idx_O} -> O")
    if idx_OXT is not None:
        name_map[idx_OXT] = "OXT"
        main_set.add(idx_OXT)
        logger.debug(f"Backbone atom: index {idx_OXT} -> OXT")

    # Choose CB
    idx_CB = None
    if idx_CA is not None:
        idx_CB = choose_CB(mol, idx_CA, main_set)
        if idx_CB is not None:
            name_map[idx_CB] = "CB"
            main_set.add(idx_CB)
            logger.debug(f"Side-chain start: index {idx_CB} -> CB")

    logger.debug("\n=== Side-chain BFS traversal (purely bond-based) ===")
    # Verify CB connectivity
    if idx_CB is not None:
        cb_atom = mol.GetAtomWithIdx(idx_CB)
        cb_neighbors = []
        for bond in cb_atom.GetBonds():
            neighbor = bond.GetOtherAtom(cb_atom)
            cb_neighbors.append(f"{neighbor.GetIdx()}({neighbor.GetSymbol()})")
        logger.debug(f"Neighbors of CB({idx_CB}): {cb_neighbors}")

    # Side-chain naming: BFS outward from CB; level letters sequence B(already used), G, D, E, Z, H, I, ...
    if idx_CB is not None:
        visited: Set[int] = set(main_set)
        q: Deque[Tuple[int, int, Optional[str]]] = deque()
        # (current atom, level_from_CA, parent_code)
        q.append((idx_CB, 1, "CB"))

        while q:
            u, level, parent_code = q.popleft()
            atom_u = mol.GetAtomWithIdx(u)
            logger.debug(f"\n--- Level {level}, start from atom {u}({parent_code}) ---")
            logger.debug(
                f"  Bond connectivity for current atom {u} ({atom_u.GetSymbol()}):"
            )

            # Print all bonds of current atom
            connected_atoms = []
            for bond in atom_u.GetBonds():
                neighbor = bond.GetOtherAtom(atom_u)
                bond_type = bond.GetBondType()
                connected_atoms.append(f"{neighbor.GetIdx()}({neighbor.GetSymbol()})")
                logger.debug(
                    f"    Bond: {u} - {neighbor.GetIdx()} ({neighbor.GetSymbol()}), type: {bond_type}"
                )

            # 1) Collect candidates based on direct bonds
            cand = []
            for bond in atom_u.GetBonds():
                neighbor = bond.GetOtherAtom(atom_u)
                vi = neighbor.GetIdx()

                if vi in visited:
                    logger.debug(
                        f"    Skip visited atom: {vi} ({neighbor.GetSymbol()})"
                    )
                    continue
                # Do not enter backbone N/C/O/OXT/CA
                if vi in main_set and vi != u:
                    logger.debug(
                        f"    Skip backbone atom: {vi} ({neighbor.GetSymbol()})"
                    )
                    continue
                # Only handle heavy atoms
                if neighbor.GetAtomicNum() <= 1:
                    logger.debug(
                        f"    Skip hydrogen atom: {vi} ({neighbor.GetSymbol()})"
                    )
                    continue

                next_level = level + 1  # +1 per bond
                greek_idx = min(max(next_level - 1, 0), len(GREEK_LEVEL) - 1)
                greek_letter = GREEK_LEVEL[greek_idx]
                base = (
                    "C"
                    if neighbor.GetAtomicNum() == 6
                    else neighbor.GetSymbol().upper()
                ) + greek_letter
                cand.append((vi, neighbor, base, next_level))
                logger.debug(
                    f"  Found candidate atom: index {vi}, element {neighbor.GetSymbol()}, proposed name {base}, bond type {bond.GetBondType()}"
                )

            if not cand:
                logger.debug(f"  No further candidate atoms found")
                continue

            # 2) Group by base under the same parent to decide numbering 1..n
            groups: Dict[str, List[Tuple[int, Chem.Atom, str, int]]] = defaultdict(list)
            for it in cand:
                groups[it[2]].append(it)

            # Subtree weight: larger branch gets earlier numbering (to distinguish e.g. ILE 1/2)
            def branch_weight(vi: int) -> int:
                seen: Set[int] = set(visited)
                seen.add(u)  # avoid going back to parent
                cnt = 0
                dq: Deque[int] = deque([vi])
                while dq:
                    x = dq.popleft()
                    if x in seen:
                        continue
                    seen.add(x)
                    ax = mol.GetAtomWithIdx(x)
                    if ax.GetAtomicNum() > 1:
                        cnt += 1
                    # Traverse subtree by bond connectivity
                    for bond in ax.GetBonds():
                        neighbor = bond.GetOtherAtom(ax)
                        nb_i = neighbor.GetIdx()
                        if nb_i not in seen:
                            dq.append(nb_i)
                return cnt

            # 3) Assign names per base group and continue BFS
            for base, items in groups.items():
                logger.debug(f"  Processing group {base}, contains {len(items)} atoms:")

                # Only need numbering if multiple atoms in the same level
                need_numbering = len(items) > 1

                if need_numbering:
                    logger.debug(f"    Numbering needed (multiple atoms at same level)")
                else:
                    logger.debug(
                        f"    Numbering not needed (single atom at this level)"
                    )

                # Stable sort: subtree heavy-atom count desc, then atom index asc
                items_sorted = sorted(items, key=lambda t: (-branch_weight(t[0]), t[0]))

                for idx, (vi, a, base_name, nlevel) in enumerate(items_sorted, start=1):
                    weight = branch_weight(vi)
                    code = f"{base_name}{idx}" if need_numbering else base_name
                    if vi not in name_map:
                        name_map[vi] = code
                        logger.debug(
                            f"    Assign name: index {vi} ({a.GetSymbol()}) -> {code} (weight: {weight})"
                        )
                    visited.add(vi)

                    # Check if there are unvisited heavy-atom neighbors (bond-based)
                    has_heavy_child = False
                    heavy_neighbors = []
                    for bond in a.GetBonds():
                        neighbor = bond.GetOtherAtom(a)
                        if (
                            neighbor.GetAtomicNum() > 1
                            and neighbor.GetIdx() not in visited
                            and neighbor.GetIdx() not in main_set
                        ):
                            has_heavy_child = True
                            heavy_neighbors.append(
                                f"{neighbor.GetIdx()}({neighbor.GetSymbol()})"
                            )

                    if has_heavy_child:
                        q.append((vi, nlevel, code))
                        logger.debug(
                            f"    Atom {vi}({code}) has unvisited heavy neighbors: {heavy_neighbors}; enqueue for further search"
                        )
                    else:
                        logger.debug(
                            f"    Atom {vi}({code}) has no unvisited heavy neighbors"
                        )

    logger.debug("\n=== Hydrogen atom naming ===")
    # Hydrogen atoms are numbered starting from H1
    h_counter = 1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 1:  # hydrogen
            name_map[i] = f"H{h_counter}"
            logger.debug(f"Hydrogen atom: index {i} -> H{h_counter}")
            h_counter += 1

    logger.debug("\n=== Unnamed heavy-atom handling ===")
    # Assign names to all unnamed heavy atoms
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() > 1 and i not in name_map:  # heavy atom and unnamed
            element_symbol = atom.GetSymbol().upper()
            name_map[i] = element_symbol
            logger.debug(
                f"Unnamed heavy atom: index {i} ({element_symbol}) -> {element_symbol}"
            )

    # AlphaFold3 compatibility post-processing
    name_map = _apply_alphafold3_fixes(mol, name_map)

    logger.debug("\n=== Resolve duplicate-name conflicts ===")
    # !! Add numeric suffix to all atoms sharing the same name globally (but skip backbone indices) !!
    name_map = _resolve_duplicate_names(name_map, main_set)

    logger.debug("\n=== Final atom naming results ===")
    for i, atom in enumerate(mol.GetAtoms()):
        final_name = name_map.get(i, atom.GetSymbol())
        atom.SetProp("atom_name", final_name)
        # logger.debug(f"Atom {i}: {atom.GetSymbol()} -> {final_name}")

    return name_map


def _apply_alphafold3_fixes(mol: Chem.Mol, name_map: Dict[int, str]) -> Dict[int, str]:
    """Apply AlphaFold3-specific atom-name fixes: only terminal carbons with exactly two hydrogens are labeled as CH2."""
    fixed_map = name_map.copy()

    # Find all backbone atoms
    main_atoms = set()
    for idx, name in fixed_map.items():
        if name in {"N", "CA", "C", "O", "OXT"}:
            main_atoms.add(idx)

    # Check each carbon atom to see whether it should be named CH2
    for idx, name in fixed_map.items():
        atom = mol.GetAtomWithIdx(idx)

        # Only process carbon atoms and not backbone atoms
        if atom.GetAtomicNum() != 6 or idx in main_atoms:
            continue

        # Count attached hydrogens and heavy neighbors
        h_count = 0
        heavy_neighbors = []

        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            if neighbor.GetAtomicNum() == 1:  # hydrogen
                h_count += 1
            elif neighbor.GetAtomicNum() > 1:  # heavy atom
                heavy_neighbors.append(neighbor.GetIdx())

        # Terminal check: only if connected to exactly 1 heavy atom
        is_terminal = len(heavy_neighbors) == 1

        # Only if carbon has exactly 2 hydrogens and is terminal, label as CH2
        if h_count == 2 and is_terminal:
            fixed_map[idx] = "CH2"
            logger.debug(
                f"Fix terminal carbon naming: index {idx} {name} -> CH2 (H count: {h_count}, heavy count: {len(heavy_neighbors)})"
            )
        else:
            # Keep original naming logic
            logger.debug(
                f"Keep original name: index {idx} {name} (H count: {h_count}, heavy count: {len(heavy_neighbors)})"
            )

    return fixed_map


def _resolve_duplicate_names(
    name_map: Dict[int, str], backbone_indices: Optional[Set[int]] = None
) -> Dict[int, str]:
    """Global uniquification: even if a name already has a numeric suffix, ensure uniqueness within the molecule.
    Only keep the original name for true backbone indices (backbone_indices); all other duplicates are renamed
    by appending extra digits at the end.
    """
    from collections import defaultdict

    keep_idxs = set(backbone_indices or [])

    # Build reverse map
    name_to_idxs: Dict[str, List[int]] = defaultdict(list)
    for idx, nm in name_map.items():
        if nm:
            name_to_idxs[nm].append(idx)

    new_map = dict(name_map)
    # Track all occupied names to avoid conflicts
    occupied = set(new_map.values())

    for base_name, idxs in name_to_idxs.items():
        if len(idxs) <= 1:
            continue
        # Choose the index to keep original name (prefer backbone), rename others
        keep = None
        for i in idxs:
            if i in keep_idxs:
                keep = i
                break
        if keep is None:
            keep = idxs[0]

        counter = 2  # start from 2, e.g., X -> X2; if already X1 -> X12
        for i in idxs:
            if i == keep:
                continue
            cand = f"{base_name}{counter}"
            while cand in occupied:
                counter += 1
                cand = f"{base_name}{counter}"
            new_map[i] = cand
            occupied.add(cand)
            counter += 1

    return new_map


def validate_alphafold3_compatibility(name_map: Dict[int, str]) -> Dict[str, any]:
    """Validate atom naming compatibility with AlphaFold3 ATOM37."""
    report = {
        "total_atoms": len(name_map),
        "atom37_compatible": 0,
        "incompatible_atoms": [],
        "warnings": [],
    }

    for idx, atom_name in name_map.items():
        if atom_name in ATOM37_ORDER:
            report["atom37_compatible"] += 1
        else:
            report["incompatible_atoms"].append(atom_name)

    compatibility_rate = (
        (report["atom37_compatible"] / report["total_atoms"]) * 100
        if report["total_atoms"] > 0
        else 0
    )

    if report["incompatible_atoms"]:
        report["warnings"].append(
            f"The following atoms are not in AlphaFold3 ATOM37: {report['incompatible_atoms']}"
        )

    report["compatibility_rate"] = compatibility_rate
    report["is_fully_compatible"] = len(report["incompatible_atoms"]) == 0

    return report


def set_pdb_info(
    mol: Chem.Mol,
    name_map: Dict[int, str],
    resname: Optional[str] = None,
    resid: int = 1,
) -> None:
    resname_eff = (resname or "UNK")[:3] if resname else "UNK"
    for i, atom in enumerate(mol.GetAtoms()):
        elem = atom.GetSymbol().upper()
        new_name = name_map.get(i, elem)
        pdb_name = new_name.rjust(4)[:4]
        info = Chem.AtomPDBResidueInfo()
        info.SetName(pdb_name)
        info.SetResidueName(resname_eff)
        info.SetResidueNumber(resid)
        info.SetIsHeteroAtom(False)
        if hasattr(info, "SetAtomSerialNumber"):
            info.SetAtomSerialNumber(i + 1)
        elif hasattr(info, "SetSerialNumber"):
            info.SetSerialNumber(i + 1)
        atom.SetMonomerInfo(info)


def _idx_to_serial(mol: Chem.Mol) -> Dict[int, int]:
    idx2serial: Dict[int, int] = {}
    for i, atom in enumerate(mol.GetAtoms()):
        serial = i + 1
        info = atom.GetMonomerInfo()
        if info is not None:
            if hasattr(info, "GetAtomSerialNumber"):
                serial = info.GetAtomSerialNumber()
            elif hasattr(info, "GetSerialNumber"):
                serial = info.GetSerialNumber()
        idx2serial[i] = int(serial)
    return idx2serial


def _build_conect_lines(mol: Chem.Mol) -> List[str]:
    from collections import defaultdict as _dd

    idx2serial = _idx_to_serial(mol)
    nbrs: Dict[int, Set[int]] = _dd(set)
    for bond in mol.GetBonds():
        a = idx2serial[bond.GetBeginAtomIdx()]
        b = idx2serial[bond.GetEndAtomIdx()]
        if a == b:
            continue
        nbrs[a].add(b)
        nbrs[b].add(a)
    lines: List[str] = []
    for sa in sorted(nbrs.keys()):
        neighs = sorted(nbrs[sa])
        for i in range(0, len(neighs), 4):
            chunk = neighs[i : i + 4]
            lines.append("CONECT" + f"{sa:>5}" + "".join(f"{sb:>5}" for sb in chunk))
    return lines


def write_pdb(mol: Chem.Mol, pdb_path: str) -> None:
    pdb_block = Chem.MolToPDBBlock(mol)
    lines = pdb_block.rstrip().splitlines()
    if lines and lines[-1].strip() == "END":
        lines = lines[:-1]
    lines.extend(_build_conect_lines(mol))
    lines.append("END")
    with open(pdb_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_sdf(mol: Chem.Mol, sdf_path: str, name_map: Dict[int, str]) -> None:
    # Write the renamed atom-name order into a molecule property for tracing
    atom_names = [
        name_map.get(i, mol.GetAtomWithIdx(i).GetSymbol().upper())
        for i in range(mol.GetNumAtoms())
    ]
    mol.SetProp("atom_names", ",".join(atom_names))
    w = SDWriter(sdf_path)
    try:
        # If you need to force V3000 (very large molecules/many properties), you can: w.SetForceV3000(True)
        w.write(mol)
    finally:
        w.close()


def aromatic_flag(atom: Chem.Atom) -> str:
    return "Y" if atom.GetIsAromatic() else "N"


def bond_order_str(b: Chem.Bond) -> str:
    """Use RDKit bond-type to decide; aromatic bonds are converted to a Kekulé-like order."""
    bond_type = b.GetBondType()

    if bond_type == Chem.BondType.SINGLE:
        return "SING"
    elif bond_type == Chem.BondType.DOUBLE:
        return "DOUB"
    elif bond_type == Chem.BondType.TRIPLE:
        return "TRIP"
    elif bond_type == Chem.BondType.AROMATIC:
        try:
            kekulized_order = b.GetBondTypeAsDouble()
            if kekulized_order == 2.0:
                return "DOUB"
            else:
                return "SING"
        except:
            return "SING"
    else:
        return "SING"


def most_similar_amino_acid(query_mol: Chem.Mol) -> str:
    """Compute similarity to standard amino acids and return the most similar amino-acid name."""
    from rdkit.Chem import DataStructs

    if query_mol is None:
        return "Invalid"
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    best_name, best_score = None, -1
    for name, smiles in AMINO_ACIDS.items():
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        score = DataStructs.TanimotoSimilarity(query_fp, fp)
        if score > best_score:
            best_name, best_score = name, score
    return best_name


# “Backbone-first” ordering for CCD multivalue atom fields !!!
def _ordered_atom_indices_for_ccd(mol: Chem.Mol, name_map: Dict[int, str]) -> List[int]:
    """
    Only used for display order of _chem_comp_atom.* multivalue fields in export_ccd_row.
    Rule (minimal perturbation):
    - Prioritize atoms in protein-like order: N, CA, C, O, OXT, CB
    - Other atoms keep original index order
    """
    priority = ["N", "CA", "C", "O", "OXT", "CB"]
    pr_rank = {nm: i for i, nm in enumerate(priority)}

    def atom_name(i: int) -> str:
        a = mol.GetAtomWithIdx(i)
        return name_map.get(i, a.GetSymbol().upper())

    idxs = list(range(mol.GetNumAtoms()))
    idxs.sort(
        key=lambda i: (0, pr_rank[atom_name(i)]) if atom_name(i) in pr_rank else (1, i)
    )
    return idxs


# “Backbone-bond-first” ordering for CCD bond multivalue fields
def _sorted_bonds_for_ccd(
    mol: Chem.Mol, name_map: Dict[int, str], atom_order_map: Dict[int, int]
) -> List[Chem.Bond]:
    """
    Only used for display order of _chem_comp_bond.* multivalue fields in export_ccd_row.
    Rule:
    - Prefer key backbone/connection bonds:
        N-CA, CA-C, C-O, C-OXT, CA-CB
    - Other bonds are stably sorted by atom order
    """

    def nm(i: int) -> str:
        a = mol.GetAtomWithIdx(i)
        return name_map.get(i, a.GetSymbol().upper())

    def norm_pair(a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    preferred_pairs = [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("C", "OXT"),
        ("CA", "CB"),
    ]
    pair_rank = {norm_pair(a, b): r for r, (a, b) in enumerate(preferred_pairs)}

    bonds = list(mol.GetBonds())

    def key(b: Chem.Bond):
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        n1 = nm(a1)
        n2 = nm(a2)
        pr = pair_rank.get(norm_pair(n1, n2), 100)

        o1 = atom_order_map.get(a1, a1)
        o2 = atom_order_map.get(a2, a2)
        lo, hi = (o1, o2) if o1 <= o2 else (o2, o1)

        return (pr, lo, hi, b.GetIdx())

    bonds.sort(key=key)
    return bonds


def dump_mapping_csv(mol: Chem.Mol, name_map: Dict[int, str], csv_path: str) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["atom_index", "element", "new_name"])
        for i, atom in enumerate(mol.GetAtoms()):
            elem = atom.GetSymbol()
            new_name = name_map.get(i, elem)
            writer.writerow([i, elem, new_name])


def export_ccd_dict(
    mol: Chem.Mol,
    name_map: Dict[int, str],
    comp_id: str = "ZCA",
    comp_name: Optional[str] = None,
    comp_type: str = "ATOMP",
) -> str:
    """Export the molecule as a CCD/mmCIF-style single TSV row and add a res column."""

    conf = _get_conf(mol)

    # Create a copy and kekulize for bond-order determination
    mol_kekulized = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol_kekulized, clearAromaticFlags=False)
        logger.debug("Successfully kekulized molecule for bond-order determination")
    except:
        logger.warning(
            "Warning: cannot fully kekulize molecule; will use default bond-order determination"
        )
        mol_kekulized = mol  # fall back to original molecule

    # Empirical formula and molecular weight
    formula_raw = rdMD.CalcMolFormula(mol)

    # Convert e.g. C24H37N3O6S to "C24 H37 N3 O6 S"
    def spaced_formula(f: str) -> str:
        import re

        parts = re.findall(r"[A-Z][a-z]?\d*", f)
        return " ".join(parts)

    formula = spaced_formula(formula_raw)
    mw = Descriptors.MolWt(mol)

    # comp basic info
    key = comp_id or "ZCA"
    comp_name = comp_name or comp_id or "ZCA"

    # === Atom multivalue fields (backbone-first order) ===
    ordered_atom_indices = _ordered_atom_indices_for_ccd(mol, name_map)
    atom_order_map = {idx: pos for pos, idx in enumerate(ordered_atom_indices)}

    charges: List[str] = []
    comp_ids: List[str] = []
    aromatic_flags: List[str] = []
    comp_atom_ids: List[str] = []
    xs: List[str] = []
    ys: List[str] = []
    zs: List[str] = []
    type_symbols: List[str] = []

    for i in ordered_atom_indices:
        atom = mol.GetAtomWithIdx(i)
        nm = name_map.get(i, atom.GetSymbol().upper())
        comp_atom_ids.append(nm)
        comp_ids.append(comp_id)
        aromatic_flags.append(aromatic_flag(atom))
        charges.append(str(int(atom.GetFormalCharge())))
        p = conf.GetAtomPosition(i)
        xs.append(f"{p.x:.3f}")
        ys.append(f"{p.y:.3f}")
        zs.append(f"{p.z:.3f}")
        type_symbols.append(atom.GetSymbol().upper())

    # Backbone-bond-first display
    bond_a1: List[str] = []
    bond_a2: List[str] = []
    bond_orders: List[str] = []
    bond_aromatic_flags: List[str] = []

    sorted_bonds = _sorted_bonds_for_ccd(mol, name_map, atom_order_map)

    for b in sorted_bonds:
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        nm1 = name_map.get(a1, mol.GetAtomWithIdx(a1).GetSymbol().upper())
        nm2 = name_map.get(a2, mol.GetAtomWithIdx(a2).GetSymbol().upper())
        bond_a1.append(nm1)
        bond_a2.append(nm2)

        # Use kekulized molecule to determine bond order (single/double/triple)
        b_kekulized = mol_kekulized.GetBondWithIdx(b.GetIdx())
        bond_orders.append(bond_order_str(b_kekulized))

        # Use original molecule for aromatic flag
        bond_aromatic_flags.append("Y" if b.GetIsAromatic() else "N")

    # Compute most similar amino acid
    res_name = most_similar_amino_acid(mol)

    def fmt(vals: List[str]) -> str:
        if not vals:
            return ""
        if len(vals) == 1:
            return vals[0]
        return ", ".join(vals)

    row = {
        "Key": tuple([key]),
        "_chem_comp.formula": tuple([formula]),
        "_chem_comp.formula_weight": tuple([f"{mw:.3f}"]),
        "_chem_comp.id": tuple([comp_id]),
        "_chem_comp.mon_nstd_parent_comp_id": tuple(["?"]),
        "_chem_comp.name": tuple([comp_name]),
        "_chem_comp.pdbx_synonyms": tuple(["?"]),
        "_chem_comp.pdbx_type": tuple([comp_type]),
        "_chem_comp.type": tuple(["PEPTIDE LINKING"]),
        "_chem_comp_atom.charge": tuple(charges),
        "_chem_comp_atom.comp_id": tuple([comp_id] * len(charges)),
        "_chem_comp_atom.pdbx_aromatic_flag": tuple(aromatic_flags),
        "_chem_comp_atom.pdbx_component_atom_id": tuple(comp_atom_ids),
        "_chem_comp_atom.atom_id": tuple(comp_atom_ids),
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal": tuple(xs),
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal": tuple(ys),
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal": tuple(zs),
        "_chem_comp_atom.type_symbol": tuple(type_symbols),
        "_chem_comp_bond.atom_id_1": tuple(bond_a1),
        "_chem_comp_bond.atom_id_2": tuple(bond_a2),
        "_chem_comp_bond.value_order": tuple(bond_orders),
        "_chem_comp_bond.pdbx_aromatic_flag": tuple(bond_aromatic_flags),
        "res": res_name,
    }

    return row


def build_ligand_ccd_from_smiles(
    ligand_name: str,
    smiles_string: str,
):

    mol = Chem.MolFromSmiles(smiles_string)

    mol = rdkit_utils.assign_atom_names_from_graph(mol)
    Chem.AllChem.EmbedMolecule(mol)
    ccd_cif = rdkit_utils.mol_to_ccd_cif(mol, component_id="fake_cif")
    ccd_cif_dict = dict(ccd_cif)
    ccd_cif_dict["_chem_comp.id"] = (ligand_name,)
    ccd_cif_dict["_chem_comp.name"] = (ligand_name,)
    ccd_cif_dict["_chem_comp.type"] = ("NON-POLYMER",)
    ccd_cif_dict["_chem_comp.mon_nstd_parent_comp_id"] = ("?",)
    ccd_cif_dict["_chem_comp.pdbx_synonyms"] = ("?",)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    formula_weight = Descriptors.MolWt(mol)
    ccd_cif_dict["_chem_comp.formula"] = (f"{formula}",)
    ccd_cif_dict["_chem_comp.formula_weight"] = (f"{formula_weight}",)
    return ccd_cif_dict


def build_residue_ccd_from_smiles(
    ligand_name: str,
    smiles_string: str,
):

    mol = Chem.MolFromSmiles(smiles_string)
    name_map = assign_names(mol)
    compatibility_report = validate_alphafold3_compatibility(name_map)
    logger.debug(f"AlphaFold3 compatibility report: {compatibility_report}")
    ccd_cif_dict = export_ccd_dict(mol, name_map, ligand_name, comp_type="ATOMP")
    return ccd_cif_dict


def build_ligand_ccd_from_cdmol(
    ligand_name: str,
    cdxml: str,
):

    cdmol = Chem.MolsFromCDXMLFile(cdxml)[0]

    mol = rdkit_utils.assign_atom_names_from_graph(mol)
    Chem.AllChem.EmbedMolecule(mol)
    ccd_cif = rdkit_utils.mol_to_ccd_cif(mol, component_id="fake_cif")
    ccd_cif_dict = dict(ccd_cif)
    ccd_cif_dict["_chem_comp.id"] = (ligand_name,)
    ccd_cif_dict["_chem_comp.name"] = (ligand_name,)
    ccd_cif_dict["_chem_comp.type"] = ("NON-POLYMER",)
    ccd_cif_dict["_chem_comp.mon_nstd_parent_comp_id"] = ("?",)
    ccd_cif_dict["_chem_comp.pdbx_synonyms"] = ("?",)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    formula_weight = Descriptors.MolWt(mol)
    ccd_cif_dict["_chem_comp.formula"] = (f"{formula}",)
    ccd_cif_dict["_chem_comp.formula_weight"] = (f"{formula_weight}",)
    return ccd_cif_dict


def build_residue_ccd_from_cdmol(
    ligand_name: str,
    cdxml: str,
):

    cdmol = Chem.MolsFromCDXMLFile(str(cdxml))[0]

    name_map = assign_names(cdmol)
    compatibility_report = validate_alphafold3_compatibility(name_map)
    logger.debug(f"AlphaFold3 compatibility report: {compatibility_report}")
    ccd_cif_dict = export_ccd_dict(cdmol, name_map, ligand_name, comp_type="ATOMP")

    return ccd_cif_dict


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
        logger.error(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        logger.error(f"读取文件时发生错误: {str(e)}")
    return None


def build_ccd_from_csv(csv_file: Path, ccd_type="NON-POLYMER"):
    ccd_data = []
    if csv_file.suffix == ".csv":
        ccd_data = read_csv_file(csv_file)
    else:
        logger.error(f"错误: 不支持的文件类型 {csv_file.suffix}")
        return None

    ccd_dict = {}
    for key, val in zip(ccd_data[0], ccd_data[1]):
        ccd_dict[key] = tuple(val.strip().split(", "))

    ccd_dict["_chem_comp.type"] = (ccd_type,)

    return ccd_dict
