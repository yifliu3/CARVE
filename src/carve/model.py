"""Ref2VA loading with LoRA and frozen shared input/output projections."""

from dataclasses import dataclass

import torch

from solarwm.backends.minimax_h3.distributed import init_sequence_parallel
from solarwm.backends.minimax_h3.model import SolarMiniMaxH3Transformer3DModel

from .config import validate_conditioning


@dataclass
class ModelHandle:
    model: object
    trainables: dict
    anyflow: bool


def build(cfg, device, *, anyflow=False, frozen=False, root_wrap=False):
    from peft import LoraConfig

    validate_conditioning(cfg)
    mc = cfg["model"]
    if mc["tiny"]:
        # The actual H3 class, with small dimensions; not a surrogate network.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["train"]["seed"])
            model = SolarMiniMaxH3Transformer3DModel.from_config(
                dict(
                    num_attention_heads=2,
                    attention_head_dim=16,
                    hidden_size=32,
                    num_layers=2,
                    num_refiner_layers=1,
                    ffn_dim=64,
                    in_channels=24,
                    audio_in_channels=32,
                    patch_size=(1, 2, 2),
                    text_dim=5120,
                    freq_dim=16,
                    time_embed_hidden_dim=32,
                    time_embed_dim=16,
                    rope_freq_dim=2,
                )
            )
    else:
        model = SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(
            mc["checkpoint"],
            subfolder=mc["subfolder"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    if anyflow:
        model.enable_anyflow(cfg["train"]["anyflow_gate"])
    model.requires_grad_(False)
    model.proxy_projection_trainable = False
    suffixes = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith(("transformer_blocks.", "token_refiner.refiner_blocks."))
        and name.endswith(suffixes)
    ]
    if not targets:
        raise ValueError("No H3 LoRA targets found")
    model.add_adapter(
        LoraConfig(
            r=mc["lora_rank"],
            lora_alpha=mc["lora_alpha"],
            target_modules=targets,
            lora_dropout=0.0,
            bias="none",
        )
    )
    # Train the new target-time MLP and carry it explicitly across checkpoints.
    if anyflow:
        model.delta_embedding.requires_grad_(True)
    model.proj_in.requires_grad_(False)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    from solarwm.backends.minimax_h3.distributed import get_sp_size

    if model.transformer_blocks[0].attn.heads % get_sp_size():
        raise ValueError("sp_size must divide the H3 attention head count")
    if mc["tiny"]:
        model.to(device)
        if mc["activation_checkpointing"]:
            model.enable_gradient_checkpointing()
    elif torch.distributed.is_initialized():
        model = shard_frozen_blocks(
            model,
            device,
            activation_checkpointing=mc["activation_checkpointing"],
            activation_checkpointing_exempt_layers=mc.get(
                "activation_checkpointing_exempt_layers", 0
            ),
            frozen=frozen,
            sharding=cfg["distributed"].get("fsdp_sharding", "full_shard"),
            root_wrap=True,
        )
    else:
        model.to(device)
        if mc["activation_checkpointing"]:
            model.enable_gradient_checkpointing()
    if frozen:
        for p in params.values():
            p.requires_grad_(False)
    model.cwm_checkpoint_segments = mc.get("attention_checkpointing", True)
    model.eval()  # All dropout is zero; deterministic AnyFlow targets.
    return ModelHandle(model, params, anyflow)


def initialize_distributed(cfg):
    import os

    import torch.distributed as dist

    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    if cfg["runtime"]["device"] == "cuda":
        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    init_sequence_parallel(cfg["distributed"]["sp_size"])
    return device, rank, world


def hybrid_process_groups():
    """Shard within each torchrun node; replicate matching shards across nodes."""
    import os

    import torch.distributed as dist

    from solarwm.backends.minimax_h3.distributed import state

    if not dist.is_initialized():
        raise ValueError("HYBRID_SHARD requires an initialized distributed process group")
    world, rank = dist.get_world_size(), dist.get_rank()
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if local_world < 2 or world % local_world:
        raise ValueError("HYBRID_SHARD requires LOCAL_WORLD_SIZE >= 2 dividing WORLD_SIZE")
    if int(os.environ.get("LOCAL_RANK", "-1")) != rank % local_world:
        raise ValueError("HYBRID_SHARD requires contiguous node ranks from torchrun")
    topology = state()
    if local_world % topology.sp_size:
        raise ValueError("HYBRID_SHARD requires each sequence-parallel group to fit within a node")
    if (
        topology.sp_size == local_world
        and topology.sp_group is not None
        and topology.dp_group is not None
    ):
        return topology.sp_group, topology.dp_group
    shard_group = replicate_group = None
    # Every rank creates all groups once, in the same order; every block reuses
    # these handles. This also supports SP smaller than a physical node.
    for start in range(0, world, local_world):
        ranks = list(range(start, start + local_world))
        group = dist.new_group(ranks)
        if rank in ranks:
            shard_group = group
    for local_rank in range(local_world):
        ranks = list(range(local_rank, world, local_world))
        group = dist.new_group(ranks)
        if rank in ranks:
            replicate_group = group
    return shard_group, replicate_group


def shard_frozen_blocks(
    model,
    device,
    *,
    activation_checkpointing=True,
    activation_checkpointing_exempt_layers=0,
    frozen=False,
    sharding="full_shard",
    root_wrap=False,
):
    """Shard frozen blocks; trainable adapters remain replicated for explicit sync."""
    from functools import partial

    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    if sharding not in {"full_shard", "hybrid_shard"}:
        raise ValueError("FSDP sharding must be full_shard or hybrid_shard")
    hybrid = sharding == "hybrid_shard"
    process_group = hybrid_process_groups() if hybrid else None
    strategy = ShardingStrategy.HYBRID_SHARD if hybrid else ShardingStrategy.FULL_SHARD
    checkpointed_blocks = checkpoint_block_ids(
        model.transformer_blocks, activation_checkpointing_exempt_layers
    )
    # Materialize each frozen block on device then shard, avoiding a full
    # 33B GPU copy while loading teacher/critic after the student.
    for name, module in list(model.named_children()):
        if name in {"transformer_blocks"}:
            for index, block in enumerate(module):
                block.to(device)
                module[index] = FSDP(
                    block,
                    device_id=device,
                    use_orig_params=True,
                    ignored_states=[p for p in block.parameters() if p.requires_grad],
                    sharding_strategy=strategy,
                    process_group=process_group,
                )
    model.to(device)
    if activation_checkpointing and not frozen:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: (
                isinstance(m, MiniMaxH3TransformerBlock) and id(m) in checkpointed_blocks
            ),
        )
    if root_wrap:
        # An independent FSDP root keeps its gathered parameters after forward.
        # Enclose the block wrappers in a parameter-free root so blocks become
        # non-root FSDP units and FULL_SHARD actually releases each full buffer.
        # Keep the original replicated non-block modules and adapter Parameter
        # objects; only the frozen block bases are managed by child FSDP units.
        block_parameters = {id(p) for block in model.transformer_blocks for p in block.parameters()}
        ignored = [
            p for p in model.parameters() if p.requires_grad or id(p) not in block_parameters
        ]
        model = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            ignored_states=ignored,
            sharding_strategy=strategy,
            process_group=process_group,
        )
    return model


def checkpoint_block_ids(blocks, exempt_layers):
    """Retain activations for the final N blocks; checkpoint the preceding blocks."""
    if type(exempt_layers) is not int or not 0 <= exempt_layers <= len(blocks):
        raise ValueError("activation_checkpointing_exempt_layers exceeds the block count")
    return {id(block) for block in list(blocks)[: len(blocks) - exempt_layers]}
