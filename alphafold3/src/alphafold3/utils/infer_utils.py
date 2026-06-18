import datetime
import os
import csv
import numpy as np
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from time import time
from typing import List

import jax
from alphafold3.common.folding_input import Input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data.pipeline import DataPipeline, DataPipelineConfig
from alphafold3.model import post_processing
from alphafold3.model.model import InferenceResult
from loguru import logger
from numpy.typing import NDArray

from alphafold3.utils.model_utils import ModelRunner


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
      embeddings: The final trunk single and pair embeddings, if requested.
    """

    seed: int
    inference_results: List[InferenceResult]
    full_fold_input: Input
    embeddings: dict[str, NDArray] | None = None
    distogram: NDArray | None = None


def predict_structure(
    fold_input: Input,
    model_runner: ModelRunner,
    buckets: List[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
) -> List[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    logger.info(f"Featurising data with {len(fold_input.rng_seeds)} seed(s)...")
    featurisation_start_time = time()
    ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
    featurised_examples = featurisation.featurise_input(
        fold_input=fold_input,
        buckets=buckets,
        ccd=ccd,
        verbose=False,
        ref_max_modified_date=ref_max_modified_date,
        conformer_max_iterations=conformer_max_iterations,
        resolve_msa_overlaps=resolve_msa_overlaps,
        use_bonds=model_runner._model_config.evoformer.enable_polymer_bonds,
        use_offsets=model_runner._model_config.evoformer.enable_cyclic_offset,
    )

    logger.info(
        f"Featurising data with {len(fold_input.rng_seeds)} seed(s) took"
        f" {time() - featurisation_start_time:.2f} seconds."
    )
    logger.info(
        "Running model inference and extracting output structure samples with"
        f" {len(fold_input.rng_seeds)} seed(s)..."
    )
    all_inference_start_time = time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        logger.info(f"Running model inference with seed {seed}...")
        inference_start_time = time()
        rng_key = jax.random.PRNGKey(seed)
        result = model_runner.run_inference(example, rng_key)
        logger.info(
            f"Running model inference with seed {seed} took"
            f" {time() - inference_start_time:.2f} seconds."
        )
        logger.info(f"Extracting output structure samples with seed {seed}...")
        extract_structures = time()
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        num_tokens = len(inference_results[0].metadata["token_chain_ids"])
        embeddings = model_runner.extract_embeddings(
            result=result, num_tokens=num_tokens
        )
        distogram = model_runner.extract_distogram(result=result, num_tokens=num_tokens)

        logger.info(
            f"Extracting {len(inference_results)} output structure samples with"
            f" seed {seed} took {time() - extract_structures:.2f} seconds."
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
                embeddings=embeddings,
                distogram=distogram,
            )
        )
    logger.info(
        "Running model inference and extracting output structures with"
        f" {len(fold_input.rng_seeds)} seed(s) took"
        f" {time() - all_inference_start_time:.2f} seconds."
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{fold_input.sanitised_name()}_data.json")
    logger.info(f"Writing model input JSON to {path}")
    with open(path, "wt") as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: List[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(
                output_dir, f"{job_name}_seed-{seed}_sample-{sample_idx}"
            )
            os.makedirs(sample_dir, exist_ok=True)
            sample_job_name = f"{job_name}_seed-{seed}_sample-{sample_idx}"
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir, name=sample_job_name
            )
            ranking_score = float(result.metadata["ranking_score"])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

        if embeddings := results_for_seed.embeddings:
            embeddings_dir = os.path.join(
                output_dir, f"{job_name}_seed-{seed}_embeddings"
            )
            os.makedirs(embeddings_dir, exist_ok=True)
            post_processing.write_embeddings(
                embeddings=embeddings, output_dir=embeddings_dir, name=job_name
            )

        if (distogram := results_for_seed.distogram) is not None:
            distogram_dir = os.path.join(
                output_dir, f"{job_name}_seed-{seed}_distogram"
            )
            os.makedirs(distogram_dir, exist_ok=True)
            distogram_path = os.path.join(
                distogram_dir, f"{job_name}_seed-{seed}_distogram.npz"
            )
            with open(distogram_path, "wb") as f:
                np.savez_compressed(f, distogram=distogram.astype(np.float16))

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=None,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(
            os.path.join(output_dir, f"{job_name}_ranking_scores.csv"), "wt"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "sample", "ranking_score"])
            writer.writerows(ranking_scores)


def process_fold_input(
    fold_input: Input,
    data_pipeline_config: DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: List[int] | None = None,
    ref_max_modified_date: datetime.date | None = None,
    conformer_max_iterations: int | None = None,
    resolve_msa_overlaps: bool = True,
    save_data: bool = True,
) -> Input | List[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.
      conformer_max_iterations: Optional override for maximum number of iterations
        to run for RDKit conformer search.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    logger.info(f"Running fold job {fold_input.name}...")

    if not fold_input.chains:
        raise ValueError("Fold input has no chains.")

    logger.info(f"Output will be written in {output_dir}")
    if Path(output_dir).exists():
        json_path = Path(
            os.path.join(output_dir, f"{fold_input.sanitised_name()}_data.json")
        )
        if json_path.exists():
            logger.info(f"Output directory {output_dir} already exists, skipping...")
            with open(json_path, "r") as f:
                json_str = f.read()
            try:
                fold_input = Input.from_json(json_str, json_path)
            except JSONDecodeError as e:
                logger.warning(f"bad json {json_path}")

    if data_pipeline_config is None:
        logger.info("Skipping data pipeline...")
    else:
        logger.info("Running data pipeline...")
        fold_input = DataPipeline(data_pipeline_config).process(fold_input)

    if save_data:
        write_fold_input_json(fold_input, output_dir)

    if model_runner is None:
        logger.info("Skipping model inference...")
        output = fold_input
    else:
        logger.info(
            f"Predicting 3D structure for {fold_input.name} with"
            f" {len(fold_input.rng_seeds)} seed(s)..."
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            ref_max_modified_date=ref_max_modified_date,
            conformer_max_iterations=conformer_max_iterations,
            resolve_msa_overlaps=resolve_msa_overlaps,
        )
        logger.info(f"Writing outputs with {len(fold_input.rng_seeds)} seed(s)...")
        write_outputs(
            all_inference_results=all_inference_results,
            output_dir=output_dir,
            job_name=fold_input.sanitised_name(),
        )
        output = all_inference_results

    logger.info(f"Fold job {fold_input.name} done, output written to {output_dir}\n")
    return output
