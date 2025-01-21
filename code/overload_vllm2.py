import vllm
from vllm.config import LoRAConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.lora.models import LoRAModelManager,logger
from typing import Any, Callable, Dict, List, Optional, Type, Tuple, Union, Set
from transformers import PretrainedConfig
from vllm.distributed.utils import divide
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.vocab_parallel_embedding import     VocabParallelEmbedding
from vllm.lora.fully_sharded_layers import (
    ColumnParallelLinearWithShardedLoRA,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora, QKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA)
from vllm.lora.layers import (BaseLayerWithLoRA,
                            _get_lora_device,
                            _not_fully_sharded_can_replace,ColumnParallelLinearWithLoRA,
                            LinearScalingRotaryEmbeddingWithLora,
                            LogitsProcessorWithLoRA,
                            ReplicatedLinearWithLoRA,
                            QKVParallelLinearWithLora)
from vllm.lora.utils import _all_lora_classes

def activate_adapter(
    self,
    lora_id: int,
) -> bool:
    """Move LoRA into a GPU buffer to be used in the forward pass."""
    if lora_id in self._active_adapters:
        return False
    first_free_slot = next(
        ((i, lora_id) for i, lora_id in enumerate(self.lora_index_to_id)
            if lora_id is None), None)
    if first_free_slot is None:
        raise ValueError("No free lora slots")
    index, _ = first_free_slot
    self._active_adapters[lora_id] = None
    lora_model = self._registered_adapters[lora_id]
    logger.debug("Activating LoRA. int id: %d, slot index: %d",
                    lora_model.id, index)
    self.lora_index_to_id[index] = lora_model.id
    for module_name, module in self.modules.items():
        module_lora = lora_model.get_lora(module_name)
        # print(type(module_lora))
        if module_lora:
            module_lora.optimize()
            # print(type(module))
            module.set_lora(index, 
                            module_lora.lora_a,
                            module_lora.lora_t, 
                            module_lora.lora_b,
                            module_lora.embeddings_tensor)
        else:
            module.reset_lora(index)
    return True

class RowParallelLinearWithLoRA(BaseLayerWithLoRA):

    def __init__(self, base_layer: RowParallelLinear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.input_size = self.base_layer.input_size_per_partition
        self.output_size = self.base_layer.output_size
        self.device = _get_lora_device(self.base_layer)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                self.input_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        tp_size = get_tensor_model_parallel_world_size()
        lora_b_output_size_per_partition = (
            self.output_size if not lora_config.fully_sharded_loras else
            divide(self.output_size, tp_size))
        
        self.lora_t_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_config.max_lora_rank,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                lora_b_output_size_per_partition,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_t_stacked[index] = 0
        self.lora_b_stacked[index] = 0

    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor:
        tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        shard_size = self.input_size
        start_idx = tensor_model_parallel_rank * shard_size
        end_idx = (tensor_model_parallel_rank + 1) * shard_size
        lora_a = lora_a[start_idx:end_idx, :]
        return lora_a
    
    def slice_lora_t(self, lora_t: torch.Tensor) -> torch.Tensor:
        return lora_t

    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor:
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_t: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.base_layer.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_t = self.slice_lora_t(lora_t)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[index,
                            0, :lora_a.shape[1], :lora_a.shape[0]].copy_(
                                lora_a.T, non_blocking=True)
        self.lora_t_stacked[index,
                            0, :lora_t.shape[1], :lora_t.shape[0]].copy_(
                                lora_t.T, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x)
        self.punica_wrapper.add_lora(output,
                                    x,
                                    self.lora_a_stacked,
                                    self.lora_t_stacked,
                                    self.lora_b_stacked, 1.0)
        return output

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: tensor whose last dimension is `input_size`. If
                    `input_is_parallel` is set, then the last dimension
                    is `input_size // tp_size`.

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            # TODO: simplify code below
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        output_parallel = self.apply(input_parallel)
        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (output_ + self.base_layer.bias
                      if self.base_layer.bias is not None else output_)
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias

    @property
    def weight(self):
        return (self.base_layer.weight if hasattr(self.base_layer, "weight")
                else self.base_layer.qweight)

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is RowParallelLinear

class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: VocabParallelEmbedding) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.embeddings_slice: Optional[Tuple[int, int]]
        self.embeddings_weights: Optional[torch.Tensor]

    def create_lora_weights(
            self,
            max_loras: int,
            lora_config: LoRAConfig,
            model_config: Optional[PretrainedConfig] = None) -> None:

        if self.base_layer.num_added_embeddings_per_partition > 0:
            # We can start adding lora weights
            self.embeddings_weights = self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:self.
                base_layer.num_org_embeddings_per_partition +
                self.base_layer.num_added_embeddings_per_partition]
            self.embeddings_slice = (
                self.base_layer.shard_indices.added_vocab_start_index -
                self.base_layer.org_vocab_size,
                self.base_layer.shard_indices.added_vocab_end_index -
                self.base_layer.org_vocab_size)
            self.base_layer.weight.data[
                self.base_layer.num_org_embeddings_per_partition:].fill_(0)
        else:
            self.embeddings_slice = None
            self.embeddings_weights = None

        self.embeddings_tensors = torch.zeros(
            (
                max_loras,
                lora_config.lora_extra_vocab_size,
                self.base_layer.embedding_dim,
            ),
            dtype=self.base_layer.weight.dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked = torch.zeros(
            (
                max_loras,
                self.base_layer.org_vocab_size +
                lora_config.lora_extra_vocab_size,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_t_stacked = torch.zeros(
            (
                max_loras,
                lora_config.max_lora_rank,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_b_stacked = torch.zeros(
            (
                max_loras,
                1,
                self.base_layer.embedding_dim,
                lora_config.max_lora_rank,
            ),
            dtype=lora_config.lora_dtype,
            device=self.base_layer.weight.device,
        )
        self.lora_a_stacked_2d = self.lora_a_stacked.view(
            self.lora_a_stacked.shape[0] * self.lora_a_stacked.shape[1],
            self.lora_a_stacked.shape[2],
        )

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = 0
        self.lora_t_stacked[index] = 0
        self.lora_b_stacked[index] = 0
        self.embeddings_tensors[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_t: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)
        self.lora_a_stacked[index, :lora_a.shape[0], :lora_a.shape[1]].copy_(
            lora_a, non_blocking=True)
        self.lora_t_stacked[index, :lora_t.shape[0], :lora_t.shape[1]].copy_(
            lora_t, non_blocking=True)
        self.lora_b_stacked[index,
                            0, :lora_b.shape[1], :lora_b.shape[0]].copy_(
                                lora_b.T, non_blocking=True)
        if embeddings_tensor is not None:
            self.embeddings_tensors[
                index, :embeddings_tensor.shape[0], :embeddings_tensor.
                shape[1], ].copy_(embeddings_tensor, non_blocking=True)
            if self.embeddings_slice is not None:
                # TODO(yard1): Optimize this copy, we don't need to copy
                # everything, just the modified part
                embeddings = self.embeddings_tensors.view(
                    self.embeddings_tensors.shape[0] *
                    self.embeddings_tensors.shape[1],
                    self.embeddings_tensors.shape[2],
                )[self.embeddings_slice[0]:self.embeddings_slice[1]]
                assert self.embeddings_weights is not None
                self.embeddings_weights[:embeddings.shape[0]].copy_(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        added_tokens_mask = x > self.base_layer.org_vocab_size - 1
        embeddings_indices = self.punica_wrapper.embeddings_indices
        indices = embeddings_indices[1].view_as(x)
        full_lora_a_embeddings = F.embedding(
            x + indices,
            self.lora_a_stacked_2d,
        )
        indices = embeddings_indices[0].view_as(x)
        full_output = self.base_layer.forward(
            x.add_(indices * added_tokens_mask))

        full_output_org = full_output
        if full_output.ndim == 3:
            full_output = full_output.view(
                full_output.shape[0] * full_output.shape[1], -1)
        if full_lora_a_embeddings.ndim == 3:
            full_lora_a_embeddings = full_lora_a_embeddings.view(
                full_lora_a_embeddings.shape[0] *
                full_lora_a_embeddings.shape[1],
                -1,
            )

        # Embedding layer only need expand op
        self.punica_wrapper.add_expand(full_output,
                                       full_lora_a_embeddings,
                                       self.lora_b_stacked,
                                       add_input=True)
        return full_output.view_as(full_output_org)
    
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return type(source_layer) is VocabParallelEmbedding

class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """ColumnParallelLinear layer that is composed of 2 sublayers (slices)
    packed together (eg. gate_proj + up_proj -> gate_up_proj).

    This means we have 2 LoRAs, each applied to one half of the layer.

    Both slices must have the same size.
    """

    def __init__(self, base_layer: MergedColumnParallelLinear) -> None:
        super().__init__(base_layer)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        n_slices = 2
        if not (len(self.base_layer.output_sizes) == n_slices
                and self.base_layer.output_sizes[0]
                == self.base_layer.output_sizes[1]):
            raise ValueError(
                "LoRAColumnParallelLinear2Slice requires 2 slices with "
                "the same size.")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(n_slices))
        
        self.lora_t_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                lora_a_output_size_per_partition,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(n_slices))
        
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                self.output_size // 2,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ) for _ in range(n_slices))

        self.output_dim = self.lora_b_stacked[0].shape[2]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_t_stacked[0][index] = 0
        self.lora_t_stacked[1][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_b_stacked[1][index] = 0

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_a
    
    def slice_lora_t(
        self, lora_t: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_t

    def slice_lora_b(
        self, lora_b: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        if lora_b[0] is None or lora_b[1] is None:
            return lora_b
        shard_size = self.output_dim
        start_idx = self.tp_rank * shard_size
        end_idx = (self.tp_rank + 1) * shard_size
        lora_b = [
            lora_b[0][:, start_idx:end_idx],
            lora_b[1][:, start_idx:end_idx],
        ]
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_t: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_t = self.slice_lora_t(lora_t)
            lora_b = self.slice_lora_b(lora_b)

        if lora_a[0] is not None:
            self.lora_a_stacked[0][
                index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                    lora_a[0].T, non_blocking=True)
            self.lora_t_stacked[0][
                index, 0, :lora_t[0].shape[1], :lora_t[0].shape[0]].copy_(
                    lora_t[0].T, non_blocking=True)
            self.lora_b_stacked[0][
                index, 0, :lora_b[0].shape[1], :lora_b[0].shape[0]].copy_(
                    lora_b[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][
                index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                    lora_a[1].T, non_blocking=True)
            self.lora_t_stacked[1][
                index, 0, :lora_t[1].shape[1], :lora_t[1].shape[0]].copy_(
                    lora_t[1].T, non_blocking=True)
            self.lora_b_stacked[1][
                index, 0, :lora_b[1].shape[1], :lora_b[1].shape[0]].copy_(
                    lora_b[1].T, non_blocking=True)

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        self.punica_wrapper.add_lora_packed_nslice(
            output,
            x, 
            self.lora_a_stacked, 
            self.lora_t_stacked,
            self.lora_b_stacked, 
            1.0,
            (self.output_dim, self.output_dim))
        return output

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is MergedColumnParallelLinear
                and len(packed_modules_list) == 2)

class MergedQKVParallelLinearWithLora(ColumnParallelLinearWithLoRA):
    """ColumnParallelLinear layer that is composed of 3 sublayers (slices)
    packed together in qkv proj fashion
    (q_proj + k_proj + v_proj -> qkv_proj).

    This means we have 3 LoRAs, each applied to one slice of the layer.

    Q slice may have different shape than K and V slices (which both have
    the same shape).
    """

    def __init__(self, base_layer: QKVParallelLinear) -> None:
        super().__init__(base_layer)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: Optional[PretrainedConfig] = None,
    ) -> None:
        self.lora_config = lora_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.q_proj_shard_size = (self.base_layer.num_heads *
                                  self.base_layer.head_size)
        self.kv_proj_shard_size = (self.base_layer.num_kv_heads *
                                   self.base_layer.head_size)
        self.q_shard_id = self.tp_rank
        self.kv_shard_id = self.tp_rank // self.base_layer.num_kv_head_replicas

        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size))
        # q, k, v
        self.lora_a_stacked = (
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )
        self.lora_t_stacked = (
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                lora_a_output_size_per_partition,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                lora_a_output_size_per_partition,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                lora_a_output_size_per_partition,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )
        self.lora_b_stacked = (
            torch.zeros(
                max_loras,
                1,
                self.q_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
            torch.zeros(
                max_loras,
                1,
                self.kv_proj_shard_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            ),
        )

        self.output_slices = (
            self.q_proj_shard_size,
            self.kv_proj_shard_size,
            self.kv_proj_shard_size,
        )
        self.packed_indices: Optional[torch.Tensor] = None
        self.standard_indices: Optional[torch.Tensor] = None
        # lazily initialized.
        self.indices: torch.Tensor
        self.indices_len: List[int]

    def reset_lora(self, index: int):
        self.lora_a_stacked[0][index] = 0
        self.lora_t_stacked[0][index] = 0
        self.lora_b_stacked[0][index] = 0
        self.lora_a_stacked[1][index] = 0
        self.lora_t_stacked[1][index] = 0
        self.lora_b_stacked[1][index] = 0
        self.lora_a_stacked[2][index] = 0
        self.lora_t_stacked[2][index] = 0
        self.lora_b_stacked[2][index] = 0

    def slice_lora_a(
        self, lora_a: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_a
    
    def slice_lora_t(
        self, lora_t: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        return lora_t

    def slice_lora_b(
        self, lora_b: List[Union[torch.Tensor, None]]
    ) -> List[Union[torch.Tensor, None]]:
        lora_b_q, lora_b_k, lora_b_v = None, None, None
        if lora_b[0] is not None:
            lora_b_q = lora_b[0][:, self.q_proj_shard_size *
                                 self.q_shard_id:self.q_proj_shard_size *
                                 (self.q_shard_id + 1), ]
        if lora_b[1] is not None:
            lora_b_k = lora_b[1][:, self.kv_proj_shard_size *
                                 self.kv_shard_id:self.kv_proj_shard_size *
                                 (self.kv_shard_id + 1), ]
        if lora_b[2] is not None:
            lora_b_v = lora_b[2][:, self.kv_proj_shard_size *
                                 self.kv_shard_id:self.kv_proj_shard_size *
                                 (self.kv_shard_id + 1), ]
        lora_b = [lora_b_q, lora_b_k, lora_b_v]
        return lora_b

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_t: torch.Tensor,
        lora_b: torch.Tensor,
        embeddings_tensor: Optional[torch.Tensor],
    ):
        self.reset_lora(index)

        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        if lora_b[0] is not None:
            lora_b_q = lora_b[0]
            self.lora_b_stacked[0][
                index, 0, :lora_b_q.shape[1], :lora_b_q.shape[0]].copy_(
                    lora_b_q.T, non_blocking=True)
        if lora_b[1] is not None:
            lora_b_k = lora_b[1]
            self.lora_b_stacked[1][
                index, 0, :lora_b_k.shape[1], :lora_b_k.shape[0]].copy_(
                    lora_b_k.T, non_blocking=True)
        if lora_b[2] is not None:
            lora_b_v = lora_b[2]
            self.lora_b_stacked[2][
                index, 0, :lora_b_v.shape[1], :lora_b_v.shape[0]].copy_(
                    lora_b_v.T, non_blocking=True)
            
        if lora_t[0] is not None:
            self.lora_t_stacked[0][
                index, 0, :lora_t[0].shape[1], :lora_t[0].shape[0]].copy_(
                    lora_t[0].T, non_blocking=True)
        if lora_t[1] is not None:
            self.lora_t_stacked[1][
                index, 0, :lora_t[1].shape[1], :lora_t[1].shape[0]].copy_(
                    lora_t[1].T, non_blocking=True)
        if lora_t[2] is not None:
            self.lora_t_stacked[2][
                index, 0, :lora_t[2].shape[1], :lora_t[2].shape[0]].copy_(
                    lora_t[2].T, non_blocking=True)

        if lora_a[0] is not None:
            self.lora_a_stacked[0][
                index, 0, :lora_a[0].shape[1], :lora_a[0].shape[0]].copy_(
                    lora_a[0].T, non_blocking=True)
        if lora_a[1] is not None:
            self.lora_a_stacked[1][
                index, 0, :lora_a[1].shape[1], :lora_a[1].shape[0]].copy_(
                    lora_a[1].T, non_blocking=True)
        if lora_a[2] is not None:
            self.lora_a_stacked[2][
                index, 0, :lora_a[2].shape[1], :lora_a[2].shape[0]].copy_(
                    lora_a[2].T, non_blocking=True)

    def apply(self, x: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
        output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        self.punica_wrapper.add_lora_packed_nslice(output, x,
                                                   self.lora_a_stacked,
                                                   self.lora_t_stacked,
                                                   self.lora_b_stacked, 1.0,
                                                   self.output_slices)
        return output

    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: List,
        model_config: Optional[PretrainedConfig],
    ) -> bool:
        return (type(source_layer) is QKVParallelLinear
                and len(packed_modules_list) == 3)

_all_lora_classes: Set[Type[BaseLayerWithLoRA]] = {
    VocabParallelEmbeddingWithLoRA,
    ColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    QKVParallelLinearWithLora,
    MergedQKVParallelLinearWithLora,
    RowParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    LogitsProcessorWithLoRA,
    ColumnParallelLinearWithShardedLoRA,
    QKVParallelLinearWithShardedLora,
    MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithShardedLora,
    RowParallelLinearWithShardedLoRA,
    LinearScalingRotaryEmbeddingWithLora,
}

def add_lora(self,
                y: torch.Tensor,
                x: torch.Tensor,
                wa_t_all: torch.Tensor,
                wt_t_all: torch.Tensor,
                wb_t_all: torch.Tensor,
                scale: float,
                y_offset: Optional[int] = None,
                y_slice_size: Optional[int] = None,
                *,
                buffer: Optional[torch.Tensor] = None) -> None:
    """
    Semantics:
    y[i] += (
        x[i].unsqueeze(0)
        @ wa_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        @ wb_t_all[indices[i], layer_idx, :, :].transpose(-1, -2)
        * scale
        ).squeeze(0)
    Args:
        y (torch.Tensor):  Output tensor. Will be changed in-place.
        x (torch.Tensor): Input tensor
        wa_t_all (torch.Tensor): lora_a's weight
        wb_t_all (torch.Tensor): lora_b's weight
        scale (float): Scaling factor.
        y_offset (Optional[int], optional): Offset to apply to the starting
            column of y.
        y_slice_size (Optional[int], optional): Size of the y column slice.
        buffer (Optional[torch.Tensor], optional): Defaults to None.
    """
    y_org = y
    y = y.view(-1, y.shape[-1])
    x = x.view(-1, x.shape[-1])
    r = wb_t_all.size(-1)
    if buffer is None:
        # We set the buffer to be float32 by default ,refer to:
        # https://github.com/triton-lang/triton/issues/1387
        buffer = torch.zeros((x.size(0), r),
                                dtype=torch.float32,
                                device=x.device)

    self.add_shrink(buffer, x, wa_t_all, scale)
    # print(buffer.shape,wt_t_all.shape)
    buffer=buffer@wt_t_all.reshape(-1,wt_t_all.shape[-1]).float()+buffer
    if y_offset is None and y_slice_size is None:
        self.add_expand(y, buffer, wb_t_all, add_input=True)
    else:
        self.add_expand_slice(y,
                                buffer,
                                wb_t_all,
                                y_offset,
                                y_slice_size,
                                add_input=True)
    y = y.view_as(y_org)

def add_lora_packed_nslice(self, 
                           y: torch.Tensor,
                           x: torch.Tensor,
                            lora_a_stacked: Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor],
                            lora_t_stacked: Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor],
                            lora_b_stacked: Tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor],
                            scale: float,
                            output_slices: Tuple[int, ...]) -> None:
    """
    Applies lora to each input. Similar to add_lora, This method is 
    used for layers that are composed of multiple sublayers
    (slices) packed together.
    """
    y_org = y
    x = x.view(-1, x.shape[-1])
    y = y.view(-1, y.shape[-1])
    offset_left = 0
    # TODO fuse these kernels
    for slice_idx in range(len(output_slices)):
        self.add_lora(y, x, 
                      lora_a_stacked[slice_idx],
                      lora_t_stacked[slice_idx],
                        lora_b_stacked[slice_idx], scale, offset_left,
                        output_slices[slice_idx])
        offset_left += output_slices[slice_idx]

    y = y.view_as(y_org)

def reset_fun():
    vllm.lora.layers.MergedQKVParallelLinearWithLora=MergedQKVParallelLinearWithLora
    vllm.lora.layers.MergedColumnParallelLinearWithLoRA=MergedColumnParallelLinearWithLoRA
    vllm.lora.layers.RowParallelLinearWithLoRA=RowParallelLinearWithLoRA
    vllm.lora.layers.VocabParallelEmbeddingWithLoRA=VocabParallelEmbeddingWithLoRA
    vllm.lora.models.LoRAModelManager.activate_adapter=activate_adapter
    vllm.lora.utils._all_lora_classes=_all_lora_classes
    vllm.lora.punica.PunicaWrapper.add_lora=add_lora
    vllm.lora.punica.PunicaWrapper.add_lora_packed_nslice=add_lora_packed_nslice
    