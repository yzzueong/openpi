import dataclasses
import functools
import logging
import platform
from typing import Any
from flax.core import freeze, unfreeze
import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.models.pi0 as pi0


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info



def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        num_workers=config.num_workers,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    step = int(train_state.step)

    def convert_to_base_model(train_state):
        """convert lora model to base model"""
        logging.info("Starting transform LoRA model to Base model...")
        
        # 1. switch train_state to dict
        params_dict = train_state.params.to_pure_dict()
        flattened = traverse_util.flatten_dict(params_dict)
        converted = {}
        
        # 2. find all layers include lora
        lora_layers = []
        for path, value in flattened.items():
            path_str = "/".join(str(p) for p in path)
            if isinstance(value, (jnp.ndarray, jax.Array)):  # 直接检查数组类型
                if "lora_A" in path_str or "lora_a" in path_str:
                    logging.info(f"Find LoRA Layer: {path_str}")
                    lora_layers.append(path)
        
        logging.info(f"Totally find {len(lora_layers)} LoRA Layers")
        
        # 3. process every lora layer
        for i, lora_a_path in enumerate(lora_layers):
            path_str = "/".join(str(p) for p in lora_a_path)
            logging.info(f"processing {i+1}/{len(lora_layers)} LoRA Layer: {path_str}")
            
            # get lora layer and original layer
            if "/mlp" not in path_str:
                base_path = tuple(str(p).replace("lora_a", "w") if isinstance(p, str) else p for p in lora_a_path)
                lora_b_path = tuple(str(p).replace("lora_a", "lora_b") if isinstance(p, str) else p for p in lora_a_path)
            else:
                base_path = tuple(str(p).replace("_lora_a", "") if isinstance(p, str) else p for p in lora_a_path)
                lora_b_path = tuple(str(p).replace("lora_a", "lora_b") if isinstance(p, str) else p for p in lora_a_path)

            # get weight
            lora_a = flattened[lora_a_path]
            lora_b = flattened[lora_b_path]
            base_weight = flattened[base_path]

            
            # use bfloat16 to save memory
            lora_a = lora_a.astype(jnp.bfloat16)
            lora_b = lora_b.astype(jnp.bfloat16)

            logging.info("cal merge weight...")
            logging.info(f"lora_a shape {lora_a.shape}, dtype {lora_a.dtype}")
            logging.info(f"lora_b shape {lora_b.shape}, dtype {lora_b.dtype}")
            logging.info(f"w shape {base_weight.shape}, dtype {base_weight.dtype}")
            
            # merge lora to original layer
            lora_weight = jnp.matmul(lora_a, lora_b)
            merged_weight = base_weight + lora_weight.astype(base_weight.dtype)
            logging.info(f"merged_weight shape {merged_weight.shape}, dtype {merged_weight.dtype}")
            
            # update weight
            converted[base_path] = merged_weight
            
            # release variables
            del lora_weight, merged_weight
            jax.clear_caches()
        
        # 4. repeat non-lora weights
        for path, value in flattened.items():
            path_str = "/".join(str(p) for p in path)
            if not any(x in path_str.lower() for x in ["lora_a", "lora_b", "lora_rank"]):
                converted[path] = value
        
        # 5. convert dict back to PyTree format
        converted_dict = traverse_util.unflatten_dict(converted)
        converted_params = nnx.State(converted_dict)
        
        # 6. remove lora node in graph
        def clean_graph(graph_def):
            if isinstance(graph_def, dict):
                cleaned = {k: v for k, v in graph_def.items() 
                         if not any(lora_type in str(k).lower() 
                                  for lora_type in ["lora_a", "lora_b", "lora_rank"])}
                return {k: clean_graph(v) for k, v in cleaned.items()}
            elif isinstance(graph_def, (list, tuple)):
                return type(graph_def)(clean_graph(x) for x in graph_def)
            else:
                return graph_def
        
        cleaned_def = clean_graph(train_state.model_def)
        
        # 7. create new train_state
        return dataclasses.replace(
            train_state,
            params=converted_params,
            model_def=cleaned_def
        )

    # transform model
    train_state = convert_to_base_model(train_state)
    logging.info("Transformation Completed")
    
    # save model, index=step+1
    _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step+1)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
