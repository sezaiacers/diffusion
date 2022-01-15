from diffusion.unet import UNetModel

from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.gaussian_diffusion import get_beta_schedule
from diffusion.gaussian_diffusion import MeanType, VarType, LossType
from diffusion import input_pipeline

import jax.numpy as jnp
import jax

from flax.training import train_state
from flax.training import checkpoints
from flax.training import common_utils
from flax import jax_utils

from flax import optim

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds

import functools


def train_and_sample(config: ml_collections.ConfigDict, workdir: str):
    tf.io.gfile.makedirs(workdir)

    rng = jax.random.PRNGKey(config.seed)

    train_ds, valid_ds = input_pipeline.get_datasets(config)

    model = UNetModel(
        config.num_classes,
        config.out_ch,
        config.ch,
        config.ch_mult,
        config.num_res_blocks,
        config.attn_resolutions,
        config.dropout_rate,
        config.resamp_with_conv,
    )

    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    input_shape = (config.batch, config.image_size, config.image_size, 3)

    # Use JIT to make sure params reside in CPU memory.
    variables = jax.jit(
        lambda: model.init({'params': init_rng, 'dropout': dropout_rng},
                           jnp.ones(input_shape),
                           jnp.arange(config.batch),
                           None,
                           True),
        backend='cpu')()

    optimizer_def = optim.Adam(config.lr, eps=config.eps)
    optimizer = optimizer_def.create(variables)
    ##optimizer, ema = checkpoints.restore_checkpoint(workdir, (optimizer, variables))

    optimizer, ema = jax_utils.replicate((optimizer, variables))

    diffusion = GaussianDiffusion(get_beta_schedule(
        config.schedule_type, config.start, config.end, config.num_timesteps))

    loss_type = LossType[config.loss_type.upper()]
    mean_type = MeanType[config.mean_type.upper()]
    var_type  = VarType[config.var_type.upper()]

    ema_decay = config.ema_decay

    def train_step(optimizer, ema, xs, ts, noise, rng):
        denoise_fn = lambda params, xs, ts: model.apply(
            params, xs, ts, None, True, rngs={'dropout': rng})

        loss_fn = lambda params: diffusion.compute_loss(
            functools.partial(denoise_fn, params), xs, ts, noise, loss_type, mean_type, var_type)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(optimizer.target)

        loss = jax.lax.pmean(loss, 'batch')
        grad = jax.lax.pmean(grad, 'batch')

        optimizer = optimizer.apply_gradient(grad)

        ema = jax.tree_multimap(lambda ema, p: ema * ema_decay + (1 - ema_decay) * p,
                                ema, optimizer.target)

        return optimizer, ema, loss

    train_step = jax.pmap(jax.jit(train_step), axis_name='batch')

    n_devices = jax.device_count()
    input_size = (n_devices, config.batch // n_devices,
                  config.image_size, config.image_size, 3)

    noise_shape = (n_devices, config.batch // n_devices,
                   config.image_size, config.image_size, 3)
    ts_shape = (n_devices, config.batch // n_devices)

    train_iter = iter(train_ds)
    while True:
        rng, step_rng, noise_key, ts_key = jax.random.split(rng, 4)
        sharded_rngs = common_utils.shard_prng_key(step_rng)
        noise = jax.random.normal(noise_key, noise_shape)
        ts = jax.random.randint(ts_key, shape=ts_shape, dtype=jnp.int32,
                                minval=0, maxval=config.num_timesteps)
        xs = next(train_iter)['image']._numpy()
        optimizer, ema, loss = train_step(optimizer, ema, xs, ts, noise, sharded_rngs)
        print(loss[0])
