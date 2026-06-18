from collections.abc import Callable
import functools
from alphafold3.model import features
from alphafold3.model.model import Model, ModelResult, InferenceResult
from alphafold3.model import params
from alphafold3.model.components import utils
import haiku as hk
import jax
from jax import numpy as jnp
from pathlib import Path
import numpy as np
import tokamax


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        config: Model.Config,
        device: jax.Device,  # type: ignore
        model_dir: Path,
    ):
        self._model_config = config
        self._device = device
        self._model_dir = model_dir

    @functools.cached_property
    def model_params(self) -> hk.Params:
        """Loads model parameters from the model directory."""
        return params.get_model_haiku_params(model_dir=self._model_dir)

    @functools.cached_property
    def _model(
        self,
    ) -> Callable[[jnp.ndarray, features.BatchDict], ModelResult]:
        """Loads model parameters and returns a jitted model forward pass."""

        @hk.transform
        def forward_fn(batch):
            return Model(self._model_config)(batch)

        return functools.partial(
            jax.jit(forward_fn.apply, device=self._device), self.model_params
        )

    def run_inference(
        self, featurised_example: features.BatchDict, rng_key: jnp.ndarray
    ) -> ModelResult:
        """Computes a forward pass of the model on a featurised example."""
        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self._device,
        )

        result = self._model(rng_key, featurised_example)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        result = dict(result)
        identifier = self.model_params["__meta__"]["__identifier__"].tobytes()
        result["__identifier__"] = identifier
        return result

    def extract_inference_results(
        self,
        batch: features.BatchDict,
        result: ModelResult,
        target_name: str,
    ) -> list[InferenceResult]:
        """Extracts inference results from model outputs."""
        return list(
            Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def extract_embeddings(
        self, result: ModelResult, num_tokens: int
    ) -> dict[str, np.ndarray] | None:
        """Extracts embeddings from model outputs."""
        embeddings = {}
        if "single_embeddings" in result:
            embeddings["single_embeddings"] = result["single_embeddings"][
                :num_tokens
            ].astype(np.float16)
        if "pair_embeddings" in result:
            embeddings["pair_embeddings"] = result["pair_embeddings"][
                :num_tokens, :num_tokens
            ].astype(np.float16)
        return embeddings or None

    def extract_distogram(
        self, result: ModelResult, num_tokens: int
    ) -> np.ndarray | None:
        """Extracts distogram from model outputs."""
        if "distogram" not in result["distogram"]:
            return None
        distogram = result["distogram"]["distogram"][:num_tokens, :num_tokens, :]
        return distogram


def make_model_config(
    *,
    flash_attention_implementation: tokamax.DotProductAttentionImplementation = "triton",
    num_diffusion_samples: int = 5,
    num_recycles: int = 10,
    return_embeddings: bool = False,
    return_distogram: bool = False,
    use_bonds: bool = False,
    use_offsets: bool = False,
) -> Model.Config:
    """Returns a model config with some defaults overridden."""
    config = Model.Config()
    config.global_config.flash_attention_implementation = flash_attention_implementation
    config.heads.diffusion.eval.num_samples = num_diffusion_samples
    config.num_recycles = num_recycles
    config.return_embeddings = return_embeddings
    config.return_distogram = return_distogram
    config.evoformer.enable_cyclic_offset = use_offsets
    config.evoformer.enable_polymer_bonds = use_bonds
    return config
