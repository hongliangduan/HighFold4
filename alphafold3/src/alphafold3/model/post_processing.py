# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Post-processing utilities for AlphaFold inference results."""

import dataclasses
import datetime
import os
from pathlib import Path
from alphafold3 import version
from alphafold3.model import confidence_types
from alphafold3.model import mmcif_metadata
from alphafold3.model import model
from alphafold3.utils.metric_uitls import (
    process_single_description,
    write_scores_to_pdb_comment,
    convert_cit_to_pdb,
)
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ProcessedInferenceResult:
    """Stores attributes of a processed inference result.

    Attributes:
      cif: CIF file containing an inference result.
      mean_confidence_1d: Mean 1D confidence calculated from confidence_1d.
      ranking_score: Ranking score extracted from CIF metadata.
      structure_confidence_summary_json: Content of JSON file with structure
        confidences summary calculated from CIF file.
      structure_full_data_json: Content of JSON file with structure full
        confidences calculated from CIF file.
      model_id: Identifier of the model that produced the inference result.
    """

    cif: bytes
    mean_confidence_1d: float
    ranking_score: float
    structure_confidence_summary_json: bytes
    structure_full_data_json: bytes
    model_id: bytes


def post_process_inference_result(
    inference_result: model.InferenceResult,
) -> ProcessedInferenceResult:
    """Returns cif, confidence_1d_json, confidence_2d_json, mean_confidence_1d, and ranking confidence."""

    # Add mmCIF metadata fields.
    timestamp = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    cif_with_metadata = mmcif_metadata.add_metadata_to_mmcif(
        old_cif=inference_result.predicted_structure.to_mmcif_dict(),
        version=f"{version.__version__} @ {timestamp}",
        model_id=inference_result.model_id,
    )
    cif = mmcif_metadata.add_legal_comment(cif_with_metadata.to_string())
    cif = cif.encode("utf-8")
    confidence_1d = confidence_types.AtomConfidence.from_inference_result(
        inference_result
    )
    mean_confidence_1d = np.mean(confidence_1d.confidence)
    structure_confidence_summary_json = (
        confidence_types.StructureConfidenceSummary.from_inference_result(
            inference_result
        )
        .to_json()
        .encode("utf-8")
    )
    structure_full_data_json = (
        confidence_types.StructureConfidenceFull.from_inference_result(inference_result)
        .to_json()
        .encode("utf-8")
    )
    return ProcessedInferenceResult(
        cif=cif,
        mean_confidence_1d=mean_confidence_1d,
        ranking_score=float(inference_result.metadata["ranking_score"]),
        structure_confidence_summary_json=structure_confidence_summary_json,
        structure_full_data_json=structure_full_data_json,
        model_id=inference_result.model_id,
    )


def write_output(
    inference_result: model.InferenceResult,
    output_dir: os.PathLike[str] | str,
    terms_of_use: str | None = None,
    name: str | None = None,
) -> None:
    """Writes processed inference result to a directory."""
    processed_result = post_process_inference_result(inference_result)

    prefix = f"{name}_" if name is not None else ""
    cif_path = Path(os.path.join(output_dir, f"{prefix}model.cif"))
    summary_path = Path(os.path.join(output_dir, f"{prefix}summary_confidences.json"))
    conf_path = Path(os.path.join(output_dir, f"{prefix}confidences.json"))

    with open(cif_path, "wb") as f:
        f.write(processed_result.cif)

    with open(summary_path, "wb") as f:
        f.write(processed_result.structure_confidence_summary_json)

    with open(conf_path, "wb") as f:
        f.write(processed_result.structure_full_data_json)

    if terms_of_use is not None:
        with open(os.path.join(output_dir, "TERMS_OF_USE.md"), "wt") as f:
            f.write(terms_of_use)

    # convert cif to pdb and write metrics to file
    pdb_file = Path(os.path.join(output_dir, f"{prefix}model.pdb"))
    convert_cit_to_pdb(cif_path, pdb_file)
    metrics_dict = process_single_description(
        conf_path=conf_path, pdb_path=pdb_file, summary_path=summary_path
    )
    write_scores_to_pdb_comment(pdb_file, pdb_file, metrics_dict)


def write_embeddings(
    embeddings: dict[str, np.ndarray],
    output_dir: os.PathLike[str] | str,
    name: str | None = None,
) -> None:
    """Writes embeddings to a directory."""
    prefix = f"{name}_" if name is not None else ""

    with open(os.path.join(output_dir, f"{prefix}embeddings.npz"), "wb") as f:
        np.savez_compressed(f, **embeddings)
