from diffusion.unet import UNetModel

from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.gaussian_diffusion import get_beta_schedule
from diffusion.gaussian_diffusion import MeanType, VarType, LossType
from diffusion import input_pipeline

import numpy as np
import jax.numpy as jnp
import jax

from absl import logging
from clu import metric_writers
from clu import periodic_actions

import flax

from flax.training import train_state
from flax.training import checkpoints as flax_checkpoints
from flax.training import common_utils
from flax import jax_utils

from flax import optim

import ml_collections

import tensorflow as tf
import tensorflow_datasets as tfds

import functools

import time

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

    initial_step, total_steps = 1, config.total_steps
    optimizer, ema, initial_step = flax_checkpoints.restore_checkpoint(
        workdir, (optimizer, variables, initial_step))
    logging.info('Will start/continue training at initial_step=%d', initial_step)

    optimizer, ema = jax_utils.replicate((optimizer, ema))

    diffusion = GaussianDiffusion(get_beta_schedule(
        config.schedule_type, config.start, config.end, config.num_timesteps))

    loss_type = LossType[config.loss_type.upper()]
    mean_type = MeanType[config.mean_type.upper()]
    var_type  = VarType[config.var_type.upper()]

    ema_decay = config.ema_decay

    shape = (8, 32, 32, 3)
    def sample(params, rng):
        rng, noise_rng = jax.random.split(rng)
        denoise_fn = lambda xs, ts: model.apply(
            params, xs, ts, None, False, rngs={'dropout': rng})
        x_0 = diffusion.p_sample_loop(denoise_fn, jax.random.normal, noise_rng,
                                       shape, mean_type, var_type, True)
        return jnp.asarray(x_0 * 127.5 + 127.5, jnp.uint8)[..., ::-1]

    sample = jax.pmap(sample, axis_name='batch')


    '''
    import cv2 as cv
    import os
    os.makedirs('images', exist_ok=True)
    for i, image in enumerate(out):
        cv.imwrite('images/{}.jpg'.format(i), image)
    '''

    def train_step(optimizer, ema, xs, ts, noise, rng):
        denoise_fn = lambda params, xs, ts: model.apply(
            params, xs, ts, None, True, rngs={'dropout': rng})

        loss_fn = lambda params: diffusion.compute_loss(
            functools.partial(denoise_fn, params), xs, ts, noise, loss_type, mean_type, var_type)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(optimizer.target)

        loss = jax.lax.pmean(loss, 'batch')
        grad = jax.lax.pmean(grad, 'batch')

        if config.grad_norm_clip > 0.0:
            grad_norm = jnp.sqrt(
                jax.tree_util.tree_reduce(
                     lambda x, y: x + jnp.sum(y**2), grad, initializer=0))
            mult = jnp.minimum(1, config.grad_norm_clip / (1e-7 + grad_norm))
            grad = jax.tree_util.tree_map(lambda z: mult * z, grad)

        optimizer = optimizer.apply_gradient(grad)

        ema = jax.tree_multimap(lambda ema, p: ema * ema_decay + (1 - ema_decay) * p,
                                ema, optimizer.target)

        return optimizer, ema, loss

    train_step = jax.pmap(train_step, axis_name='batch')

    n_devices = jax.device_count()
    input_size = (n_devices, config.batch // n_devices,
                  config.image_size, config.image_size, 3)

    noise_shape = (n_devices, config.batch // n_devices,
                   config.image_size, config.image_size, 3)
    ts_shape = (n_devices, config.batch // n_devices)


    # Setup metric writer & hooks.
    writer = metric_writers.create_default_writer(workdir, asynchronous=False)
    writer.write_hparams(config.to_dict())
    hooks = [
        periodic_actions.Profile(logdir=workdir),
        periodic_actions.ReportProgress(
            num_train_steps=total_steps, writer=writer),
    ]

    logging.info('Starting training loop; initial compile can take a while...')
    t0 = lt0 = time.time()
    lstep = initial_step

    import os
    import cv2 as cv
    images_dirname = 'images'
    os.makedirs(images_dirname, exist_ok=True)

    for step, batch in zip(
        range(initial_step, total_steps + 1),
            input_pipeline.prefetch(train_ds, config.prefetch)):

        with jax.profiler.StepTraceContext('train', step_num=step):
            rng, step_rng, noise_key, ts_key = jax.random.split(rng, 4)
            sharded_rngs = common_utils.shard_prng_key(step_rng)
            noise = jax.random.normal(noise_key, noise_shape)
            ts = jax.random.randint(ts_key, shape=ts_shape, dtype=jnp.int32,
                                    minval=0, maxval=config.num_timesteps)
            xs = batch['image']
            optimizer, ema, loss = train_step(optimizer, ema, xs, ts, noise, sharded_rngs)

        for hook in hooks:
            hook(step)

        if step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)
            t0 = time.time()
            lt0, lstep = time.time(), step

        if config.generate_every and step % config.generate_every == 0:
            rng, step_rng = jax.random.split(rng)
            sharded_rngs = common_utils.shard_prng_key(step_rng)
            x_0 = flax.jax_utils.unreplicate(sample(ema, sharded_rngs))
            x_0 = np.asarray(x_0.reshape(-1, 32, 32, 3))
            dirname = os.path.join(images_dirname, str(step))
            os.makedirs(dirname, exist_ok=True)
            for i, image in enumerate(x_0):
                cv.imwrite(os.path.join(dirname, '{}.jpg'.format(i)), image)

        # Report training metrics
        if config.progress_every and step % config.progress_every == 0:
            img_sec_core_train = (config.batch * (step - lstep) /
                                  (time.time() - lt0)) / jax.device_count()
            lt0, lstep = time.time(), step
            writer.write_scalars(
                step,
                dict(
                    train_loss=float(flax.jax_utils.unreplicate(loss)),
                    img_sec_core_train=img_sec_core_train))
            done = step / total_steps
            logging.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '
                         f'img/sec/core: {img_sec_core_train:.1f}, '
                         f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

        if ((config.checkpoint_every and step % config.eval_every == 0) or
                step == total_steps):
            chekpoint = (flax.jax_utils.unreplicate(optimizer),
                         flax.jax_utils.unreplicate(ema),
                         step)
            checkpoint_path = flax_checkpoints.save_checkpoint(
                workdir, chekpoint, step, keep=3)
            logging.info('Stored checkpoint at step %d to "%s"', step,
                         checkpoint_path)
