import vllm
import os
import math
import json
from vllm.lora.lora import LoRALayerWeights,PackedLoRALayerWeights
from vllm.lora.models import LoRAModel
from vllm.lora.utils import is_regex_target_modules
from vllm.lora.models import get_lora_id
import torch
import safetensors.torch
from typing import Sequence as GenericSequence
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from vllm.utils import is_pin_memory_available
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.lora.layers import (BaseLayerWithLoRA,
                              LinearScalingRotaryEmbeddingWithLora)


class LoRALayerWeights:
    """LoRA weights for a layer composed of two low rank matrixes."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alpha: int,
        lora_a: torch.Tensor,
        lora_t: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor] = None,
        scaling: Optional[float] = None,
    ) -> None:
        self.module_name = module_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_a = lora_a
        self.lora_t = lora_t
        self.lora_b = lora_b
        self.embeddings_tensor = embeddings_tensor

        if scaling is None:
            self.scaling = self.lora_alpha / self.rank
        else:
            self.scaling = scaling

    def optimize(self) -> "LoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        if self.scaling == 1:
            return self
        self.lora_b *= self.scaling
        self.scaling = 1
        return self

    @property
    def input_dim(self) -> int:
        return self.lora_a.shape[0]

    @property
    def output_dim(self) -> int:
        return self.lora_b.shape[1]

    @property
    def is_packed(self) -> bool:
        return False

    @property
    def extra_vocab_size(self) -> int:
        return self.embeddings_tensor.shape[
            0] if self.embeddings_tensor is not None else 0

    @classmethod
    def create_dummy_lora_weights(
            cls,
            module_name: str,
            input_dim: int,
            output_dim: int,
            rank: int,
            dtype: torch.dtype,
            device: torch.types.Device,
            embeddings_tensor_dim: Optional[int] = None) -> "LoRALayerWeights":
        pin_memory = str(device) == "cpu" and is_pin_memory_available()
        lora_a = torch.zeros([input_dim, rank],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        lora_t = torch.zeros([rank, rank],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        lora_b = torch.zeros([rank, output_dim],
                             dtype=dtype,
                             device=device,
                             pin_memory=pin_memory)
        embeddings_tensor = torch.rand(
            10,
            embeddings_tensor_dim,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory) if embeddings_tensor_dim else None
        return cls(
            module_name,
            rank=rank,
            lora_alpha=1,
            lora_a=lora_a,
            lora_t=lora_t,
            lora_b=lora_b,
            embeddings_tensor=embeddings_tensor,
        )

class PackedLoRALayerWeights(LoRALayerWeights):
    """LoRA used for packed layers (eg. qkv_proj)."""

    def __init__(
        self,
        module_name: str,
        rank: int,
        lora_alphas: List[Optional[int]],
        lora_a: List[Optional[torch.Tensor]],
        lora_t: List[Optional[torch.Tensor]],
        lora_b: List[Optional[torch.Tensor]],
        scaling: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            module_name=module_name,
            rank=rank,
            lora_alpha=0,
            lora_a=lora_a,
            lora_t=lora_t,
            lora_b=lora_b,
            scaling=scaling,  # type: ignore
            embeddings_tensor=None,
        )
        self.lora_alphas = lora_alphas
        if scaling is None:
            self.scaling = [  # type: ignore
                lora_alpha / self.rank  # type: ignore # noqa
                for lora_alpha in self.lora_alphas
            ]

    @classmethod
    def pack(
        cls, loras: GenericSequence[Optional["LoRALayerWeights"]]
    ) -> "PackedLoRALayerWeights":
        """Pack a list of LoRAs into a single LoRA.

        If LoRA is None, it signifies that the submodule does not have a LoRA.
        """
        first_lora = next(lora for lora in loras if lora is not None)
        for lora in loras:
            if lora is None:
                continue
            lora.optimize()
        rank = first_lora.rank
        module_name = first_lora.module_name
        obj = cls(
            module_name,
            rank,
            [lora.lora_alpha if lora is not None else None for lora in loras],
            [lora.lora_a if lora is not None else None for lora in loras],
            [lora.lora_t if lora is not None else None for lora in loras],
            [lora.lora_b if lora is not None else None for lora in loras],
            scaling=[
                1 if lora is not None else None  # type: ignore
                for lora in loras
            ])
        return obj

    def optimize(self) -> "PackedLoRALayerWeights":
        """Optimize the LoRA by merging the scaling into lora_b."""
        for i in range(len(self.lora_b)):
            if self.scaling[i] == 1 or self.lora_b[i] is None:  # type: ignore
                continue
            self.lora_b[i] *= self.scaling[i]  # type: ignore
            self.scaling[i] = 1  # type: ignore
        return self

    @property
    def input_dim(self) -> int:
        raise NotImplementedError()

    @property
    def output_dim(self) -> int:
        raise NotImplementedError()

    @property
    def is_packed(self) -> bool:
        return True

def parse_fine_tuned_lora_name(name: str) -> Tuple[str, bool]:
    """Parse the name of lora weights.

    args:
        name: the name of the fine-tuned LoRA, e.g.
            base_model.model.dense1.weight
    return:
        Tuple(module_name, is_lora_a):
            module_name: the name of the module, e.g. model.dense1,
            is_lora_a whether the tensor is lora_a or lora_b.
    """
    parts = name.split(".")

    if len(parts) >= 2 and parts[0] == "base_model" and parts[1] == "model":
        if parts[-1] == "weight":
            if parts[-2] == "lora_A" or parts[-2] == "lora_B" or parts[-2] == "lora_T":
                return ".".join(parts[2:-2]), parts[-2]
            elif parts[-2] == "lora_lT" or parts[-2] == "lora_mT":
                return ".".join(parts[2:-2]), parts[-2]
        elif parts[-1] == "lora_embedding_A" or parts[-1] == "lora_embedding_B" or parts[-1] == "lora_embedding_T":
            return ".".join(parts[2:-1]), parts[-1] == "lora_embedding_A"

    raise ValueError(f"{name} is unsupported LoRA weight")

@classmethod
def from_local_checkpoint(
    cls,
    lora_dir: str,
    expected_lora_modules: List[str],
    *,
    max_position_embeddings: Optional[int] = None,
    lora_model_id: Optional[int] = None,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    target_embedding_padding: Optional[int] = None,
    embedding_modules: Optional[Dict[str, str]] = None,
    embedding_padding_modules: Optional[List[str]] = None,
) -> "LoRAModel":
    """Create a LoRAModel from a local checkpoint.
    
    Args:
        lora_dir: The local path that has lora data.
        expected_lora_modules: Name of modules that are expected to be
            replaced by lora.
        max_position_embeddings: Max position embedding length. Used to
            scaling the largest context length. If None, the lora model's
            context length is not scaled.
        lora_model_id: Lora model id. If not given, automatically set by
            a global counter.
        device: Device where the lora model is loaded.
        dtype: dtype of the lora model weights.

    Returns:
        Loaded LoRA Model.
    """
    lora_config_path = os.path.join(lora_dir, "adapter_config.json")
    lora_tensor_path = os.path.join(lora_dir, "adapter_model.safetensors")
    lora_bin_file_path = os.path.join(lora_dir, "adapter_model.bin")
    new_embeddings_tensor_path = os.path.join(
        lora_dir, "new_embeddings.safetensors")
    new_embeddings_bin_file_path = os.path.join(lora_dir,
                                                "new_embeddings.bin")
    with open(lora_config_path) as f:
        config = json.load(f)
    if os.path.isfile(lora_tensor_path):
        tensors: Dict[str, torch.Tensor] = {}
        # Find unexpected modules.
        # Use safetensor key as a source of truth to find expected modules.
        # in peft if you have target_modules A, B, C and C does not exist
        # in the model it won’t error and model will be trained with A, B
        # loraified. C won’t exist in the safetensor but it will exist in
        # the target_modules of the adapter_config.json.
        unexpected_modules = []
        with safetensors.safe_open(lora_tensor_path,
                                    framework="pt") as f:  # type: ignore
            for lora_module in f.keys():  # noqa
                module_name, _ = parse_fine_tuned_lora_name(lora_module)
                part_name = module_name.split(".")[-1]
                if part_name not in expected_lora_modules:
                    unexpected_modules.append(module_name)
            if unexpected_modules:
                raise ValueError(
                    f"While loading {lora_dir}, expected"
                    f" target modules in {expected_lora_modules}"
                    f" but received {unexpected_modules}."
                    f" Please verify that the loaded LoRA module is correct"
                )
            # Load tensors if there are only expected modules.
            for module in f.keys():  # noqa
                part_name=module.split(".")
                if part_name[-2] == "lora_A" or part_name[-2] == "lora_B" or part_name[-2] == "lora_T":
                    tensors[module] = f.get_tensor(module)
    elif os.path.isfile(lora_bin_file_path):
        # When a bin file is provided, we rely on config to find unexpected
        # modules.
        unexpected_modules = []
        target_modules = config["target_modules"]
        if not isinstance(target_modules, list):
            target_modules = [target_modules]
        for module in target_modules:
            # Compatible with more modules,
            # such as:layers.11.self_attn.k_proj
            part_name = module.split(".")[-1]
            if part_name not in expected_lora_modules:
                unexpected_modules.append(module)
        # loaded lora's target modules must be a subset of
        # expected_lora_modules. It is not reliable. See
        # https://github.com/vllm-project/vllm/pull/5909. But there's no
        # other better mechanism.
        if unexpected_modules and not is_regex_target_modules(
                config["target_modules"], expected_lora_modules):
            raise ValueError(
                f"While loading {lora_dir}, expected"
                f" target modules in {expected_lora_modules}"
                f" but received {unexpected_modules}."
                f" Please verify that the loaded LoRA module is correct")
        tensors = torch.load(lora_bin_file_path, map_location=device)
    else:
        raise ValueError(f"{lora_dir} doesn't contain tensors")

    embeddings = None
    if os.path.isfile(new_embeddings_tensor_path):
        embeddings = safetensors.torch.load_file(
            new_embeddings_tensor_path)
    elif os.path.isfile(new_embeddings_bin_file_path):
        embeddings = torch.load(new_embeddings_bin_file_path,
                                map_location=device)

    rank = config["r"]
    lora_alpha = config["lora_alpha"]
    context_length = config.get("context_length", None)
    scaling_factor = None
    if context_length:
        if max_position_embeddings is None:
            max_position_embeddings = context_length
        scaling_factor = float(
            math.ceil(context_length / max_position_embeddings))

    return cls.from_lora_tensors(
        lora_model_id=get_lora_id()
        if lora_model_id is None else lora_model_id,
        rank=rank,
        lora_alpha=lora_alpha,
        tensors=tensors,
        device=device,
        dtype=dtype,
        embeddings=embeddings,
        target_embedding_padding=target_embedding_padding,
        scaling_factor=scaling_factor,
        embedding_modules=embedding_modules,
        embedding_padding_modules=embedding_padding_modules,
    )

@classmethod
def from_lora_tensors(
    cls,
    lora_model_id: int,
    rank: int,
    lora_alpha: int,
    tensors: Dict[str, torch.Tensor],
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    embeddings: Optional[Dict[str, torch.Tensor]] = None,
    target_embedding_padding: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    embedding_modules: Optional[Dict[str, str]] = None,
    embedding_padding_modules: Optional[List[str]] = None,
) -> "LoRAModel":
    """Create a LoRAModel from a dictionary of tensors."""
    pin_memory = str(device) == "cpu" and is_pin_memory_available()
    loras: Dict[str, LoRALayerWeights] = {}
    for tensor_name, tensor in tensors.items():
        module_name, is_lora_a = parse_fine_tuned_lora_name(tensor_name)
        if module_name not in loras:
            lora_embeddings_tensor = None
            if embeddings:
                assert embedding_modules is not None
                embeddings_module = next(
                    (k for k in embedding_modules if k in module_name),
                    None)
                if embeddings_module:
                    lora_embeddings_tensor = embeddings[
                        embedding_modules[embeddings_module]].to(
                            device=device, dtype=dtype)
                    if pin_memory:
                        lora_embeddings_tensor = (
                            lora_embeddings_tensor.pin_memory())
            loras[module_name] = LoRALayerWeights(module_name, rank,
                                                    lora_alpha, None, None, None,
                                                    lora_embeddings_tensor)
        if is_lora_a=='lora_A':
            loras[module_name].lora_a = tensor.to(device=device,
                                                    dtype=dtype).t()
            if pin_memory:
                loras[module_name].lora_a = loras[
                    module_name].lora_a.pin_memory()
        elif is_lora_a=='lora_T':
            loras[module_name].lora_t = tensor.to(device=device,
                                                    dtype=dtype).t()
            if pin_memory:
                loras[module_name].lora_t = loras[
                    module_name].lora_t.pin_memory()
        elif is_lora_a=='lora_B':
            loras[module_name].lora_b = tensor.to(device=device,
                                                    dtype=dtype).t()
            assert embedding_padding_modules is not None
            if any(name in module_name
                    for name in embedding_padding_modules
                    ) and target_embedding_padding is not None:
                lora_b = loras[module_name].lora_b
                assert target_embedding_padding >= lora_b.shape[1]
                addition = target_embedding_padding - lora_b.shape[1]
                loras[module_name].lora_b = torch.nn.functional.pad(
                    lora_b, (0, addition))
            if pin_memory:
                loras[module_name].lora_b = loras[
                    module_name].lora_b.pin_memory()

    for lora in loras.values():
        lora.optimize()
    return cls(lora_model_id, rank, loras, scaling_factor=scaling_factor)

def create_dummy_lora(
            self,
            lora_id: int,
            rank: int,
            scaling_factor: Optional[float],
            embedding_modules: Optional[Dict[str, str]] = None) -> LoRAModel:
        """Create zero-initialized LoRAModel for warmup."""
        model = LoRAModel(lora_id, rank, {}, scaling_factor)
        for module_name, module in self.model.named_modules():
            if (not self._match_target_modules(module_name)
                    or not isinstance(module, BaseLayerWithLoRA)
                    or isinstance(module, LinearScalingRotaryEmbeddingWithLora)
                    or self._filter_unsupported_mm_module(module_name)):
                continue
            parts = module_name.split(".")
            if module_name not in self.packed_modules:
                assert embedding_modules is not None
                if parts[-1] in embedding_modules:
                    input_dim = (module.base_layer.org_vocab_size +
                                 self.lora_config.lora_extra_vocab_size if
                                 hasattr(module.base_layer, "org_vocab_size")
                                 else module.base_layer.weight.shape[1])
                    output_dim = module.base_layer.embedding_dim if hasattr(
                        module.base_layer,
                        "embedding_dim") else module.base_layer.weight.shape[0]
                    embeddings_tensor_dim = (module.base_layer.embedding_dim if
                                             hasattr(module.base_layer,
                                                     "embedding_dim") else
                                             module.base_layer.weight.shape[1])
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        input_dim,
                        output_dim,
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                        embeddings_tensor_dim=embeddings_tensor_dim)
                else:
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name,
                        module.lora_a_stacked.shape[-1],
                        module.lora_b_stacked.shape[-2],
                        rank,
                        module.lora_a_stacked.dtype,
                        "cpu",
                    )
                lora.optimize()
            else:
                parts = module_name.split(".")
                replacements = self.packed_modules_mapping[parts[-1]]
                subloras: List[Optional["LoRALayerWeights"]] = []
                for i, r in enumerate(replacements):
                    lora = LoRALayerWeights.create_dummy_lora_weights(
                        module_name + "." + r,
                        module.lora_a_stacked[i].shape[-1],
                        module.lora_b_stacked[i].shape[-2],
                        rank,
                        module.lora_a_stacked[i].dtype,
                        "cpu",
                    )
                    lora.optimize()
                    subloras.append(lora)
                lora = PackedLoRALayerWeights.pack(subloras)
            model.loras[module_name] = lora
        return model

def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
        for module_name, new_module_names in self.packed_modules.items():
            replacement_loras: List[Optional[LoRALayerWeights]] = []
            has_replacement = False
            for r in new_module_names:
                lora = lora_model.get_lora(r)
                replacement_loras.append(lora)
                if lora:
                    has_replacement = True
            if not has_replacement:
                continue
            for i in range(len(replacement_loras)):
                if replacement_loras[i]:
                    continue
                replacement_loras[i] = None
            lora_model.loras[module_name] = PackedLoRALayerWeights.pack(
                replacement_loras)

def reset_fun():
    vllm.lora.utils.parse_fine_tuned_lora_name=parse_fine_tuned_lora_name
    vllm.lora.lora.LoRALayerWeights=LoRALayerWeights
    vllm.lora.lora.PackedLoRALayerWeights=PackedLoRALayerWeights
    vllm.lora.models.LoRAModel.from_lora_tensors=from_lora_tensors
    vllm.lora.models.LoRAModel.from_local_checkpoint=from_local_checkpoint
    vllm.lora.models.LoRAModelManager.create_dummy_lora=create_dummy_lora
    vllm.lora.models.LoRAModelManager._create_merged_loras_inplace=_create_merged_loras_inplace