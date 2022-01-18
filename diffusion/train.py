from diffusion.unet import UNetModel

from diffusion.gaussian_diffusion import GaussianDiffusion
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

import tensorflow as tf
import tensorflow_datasets as tfds

import functools
import time


def make_update_fn(apply_fn,
                   criterion,
                   num_timesteps,
                   decay,
                   grad_norm_clip);

    def update_fn(opt, params, step, batch, rng):
        rng, noise_rng, timesteps_rng, dropout_rng = jax.random.split(rng, 4)

        inputs, labels = batch['image'], batch['label']

        noise = jax.random.normal(noise_key, inputs.shape)
        timesteps = jax.random.randint(timesteps_rng,
                                       minval=0, maxval=num_timesteps,
                                       shape=(inputs[0], ), dtype=jnp.int32)
        def loss_fn(params):
            params, rngs = dict(params=params), dict(dropout=dropout_rng)
            output = apply_fn(params, rngs=rngs inputs=inputs,
                              labels=labels, timesteps=timesteps, train=True)
            return criterion(inputs, noise, timesteps, outputs)

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(optimizer.target)

        loss = jax.lax.pmean(loss, 'batch')
        grad = jax.lax.pmean(grad, 'batch')

        grad_norm = jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(y**2), grad, initializer=0.0))
        mult = jnp.minimum(1.0, grad_norm_clip / (1e-7 + grad_norm))
        grad = jax.tree_util.tree_map(lambda z: mult * z, grad)

        opt = opt.apply_gradient(grad)

        params = jax.tree_multimap(
            lambda shadow_variable, variable : decay * shadow_variable + (1 - decay) * variable,
            params, opt.target)

        return opt, params, rng, loss

    return jax.pmap(update_fn, axis_name='batch')


def make_sample_fn(*, apply_fn, num_timesteps):
    pass


def init_model(config, rng):
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

    init_rng, dropout_rng = jax.random.split(rng)
    init_fn = lambda: model.init(
        dict(params=init_rng, dropout=dropout_rng),
        inputs=jnp.ones((1, config.image_size, config.image_size, 3)),
        labels=None,
        timesteps=jnp.arange(config.batch),
        train=True)
    variables = jax.jit(init_fn, backend='cpu')()
    return model.apply_fn, variables


def train_and_sample(config, workdir):
    tf.io.gfile.makedirs(workdir)

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng, 3)

    train_ds, valid_ds = input_pipeline.get_datasets(config)
    diffusion = GaussianDiffusion.from_config(config)

    apply_fn, params = init_model(config, init_rng)

    opt = optim.Adam(config.lr, eps=config.eps).create(params)

    initial_step, total_steps = 1, config.total_steps
    opt, params, initial_step = flax_checkpoints.restore_checkpoint(
        workdir, (opt, params, initial_step))
    logging.info('Will start/continue training at initial_step=%d', initial_step)

    opt_repl = flax.jax_utils.replicate(opt)
    params_repl = flax.jax_utils.replicate(params_repl)
    update_rng_repl = flax.jax_utils.replicate(rng)
    update_fn_repl = make_update_fn(apply_fn, diffusion.compute_loss,
                                    config.num_timesteps, config.decay,
                                    config.grad_norm_clip)

    del opt
    del params

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

    update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))
    for step, batch in zip(
        range(initial_step, total_steps + 1),
            input_pipeline.prefetch(train_ds, config.prefetch)):

        with jax.profiler.StepTraceContext('train', step_num=step):
            step_repl = flax.jax_utils.replicate(step)
            opt_repl, params_repl, update_rng_repl, loss_repl = update_fn_repl(
                opt_repl, params_repl, step_repl, batch, update_rng_repl)

        for hook in hooks:
            hook(step)

        if step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)
            t0 = time.time()
            lt0, lstep = time.time(), step

        if config.generate_every and step % config.generate_every == 0:
            '''
            rng, step_rng = jax.random.split(rng)
            sharded_rngs = common_utils.shard_prng_key(step_rng)
            x_0 = flax.jax_utils.unreplicate(sample(ema, sharded_rngs))
            x_0 = np.asarray(x_0.reshape(-1, 32, 32, 3))
            dirname = os.path.join(images_dirname, str(step))
            os.makedirs(dirname, exist_ok=True)
            for i, image in enumerate(x_0):
                cv.imwrite(os.path.join(dirname, '{}.jpg'.format(i)), image)

            '''
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
