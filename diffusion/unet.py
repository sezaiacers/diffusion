from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def default_kernel_init(scale, dtype=jnp.float32):
    scale=1e-10 if scale == 0 else scale
    return nn.initializers.variance_scaling(
        scale, mode='fan_avg', distribution='uniform',
        dtype=dtype)

def timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2

    freqs = jnp.exp(
        -jnp.log(10000) / (half_dim - 1) * \
        jnp.arange(half_dim, dtype=jnp.float32))

    timesteps = jnp.asarray(timesteps, jnp.float32)
    args = timesteps[:, None] * freqs[None, :]
    embedding = jnp.concatenate(
        (jnp.sin(args), jnp.cos(args)), axis=-1)

    if embedding_dim % 2 == 1:
        embedding = jnp.pad(embedding, [[0, 0], [0, 1]])
    return embedding

def conv2d(inputs, out_ch, scale=1.0):
    kernel_init = default_kernel_init(scale)
    return nn.Conv(out_ch, (3, 3), (1, 1), 'SAME',
                   kernel_init=kernel_init)(inputs)

def dense(inputs, out_ch, scale=1.0):
    kernel_init = default_kernel_init(scale)
    return nn.Dense(out_ch,
                    kernel_init=kernel_init)(inputs)

def normalize(inputs, time_emb):
    return nn.GroupNorm(num_groups=32)(inputs)

def dropout(inputs, rate, deterministic):
    return nn.Dropout(rate)(inputs, deterministic)

def upsample(inputs, with_conv):
    n, h, w, c = inputs.shape
    outputs = jax.image.resize(inputs,
                               [n, h * 2, w * 2, c],
                               jax.image.ResizeMethod.NEAREST)
    if with_conv:
        outputs = conv2d(outputs, c, scale=1.0)
    return outputs

def downsample(inputs, with_conv):
    ch = inputs.shape
    if with_conv:
        kernel_init = default_kernel_init(1.0)
        return nn.Conv(ch, (3, 3), (2, 2), 'SAME',
                       kernel_init=kernel_init)(inputs)
    else:
        return nn.avg_pool(inputs, (2, 2), (2, 2), 'SAME')

def resnet_block(inputs, time_emb, deterministic,
                 out_ch=None, conv_shortcut=False,
                 dropout_rate=0.0):
    in_ch = inputs.shape[-1]
    out_ch = out_ch or in_ch

    outputs = nn.swish(normalize(inputs, time_emb))
    outputs = conv2d(outputs, out_ch, scale=1.0)

    outputs += jnp.expand_dims(
        dense(nn.swish(time_emb), out_ch, scale=1.0),
        (-2, -3))

    outputs = nn.swish(normalize(outputs, time_emb))
    outputs = dropout(outputs, dropout_rate, deterministic)
    outputs = conv2d(outputs, out_ch, scale=0.0)

    if in_ch != out_ch:
        if conv_shortcut:
            inputs = conv2d(inputs, out_ch, scale=1.0)
        else:
            inputs = dense(inputs, out_ch, scale=1.0)
    return inputs + outputs

def attention_block(inputs, time_emb):
    ch = inputs.shape[-1]

    outputs = normalize(inputs, time_emb)

    q = dense(outputs, ch, scale=1.0)
    k = dense(outputs, ch, scale=1.0)
    v = dense(outputs, ch, scale=1.0)

    scale = ch ** (-0.5)
    logits = jnp.einsum('bhwc, bHWc -> bhwHW', q, k) * scale
    probs = nn.softmax(logits, (-1, -2))

    outputs = jnp.einsum('bhwHW, bHWc -> bhwc', probs, v)
    outputs = dense(outputs, ch, scale=0.0)
    return inputs + outputs




class UNetModel(nn.Module):
    num_classes: int
    out_ch: int
    ch: int
    ch_mult: Sequence[int]
    num_res_blocks: int
    attn_resolutions: Sequence[int]
    dropout_rate: float
    resamp_with_conv: bool

    @nn.compact
    def __call__(self, inputs, timesteps, labels, train):
        assert self.num_classes == 1 and labels is None
        del labels

        time_emb = timestep_embedding(timesteps, self.ch)
        time_emb = dense(time_emb, self.ch * 3, scale=1.0)
        time_emb = dense(nn.swish(time_emb), self.ch * 4, scale=1.0)

        num_resolutions = len(self.ch_mult)

        # Downsampling
        hs = [conv2d(inputs, self.ch, scale=1.0)]
        for level in range(num_resolutions):
            for block in range(self.num_res_blocks):
                ch = self.ch * self.ch_mult[level]
                h = resnet_block(hs[-1], time_emb, not train,
                                 ch, dropout_rate=self.dropout_rate)

                if h.shape[1] in self.attn_resolutions:
                    h = attention_block(h, time_emb)

                hs.append(h)

            if level != num_resolutions - 1:
                hs.append(downsample(hs[-1], self.resamp_with_conv))


        # Middle
        h = hs[-1]
        h = resnet_block(h, time_emb, not train,
                         dropout_rate=self.dropout_rate)
        h = attention_block(h, time_emb)
        h = resnet_block(h, time_emb, not train,
                         dropout_rate=self.dropout_rate)


        # Upsampling
        for level in reversed(range(num_resolutions)):
            for block in range(self.num_res_blocks + 1):
                ch = self.ch * self.ch_mult[level]
                inputs = jnp.concatenate([h, hs.pop()], -1)
                h = resnet_block(inputs, time_emb, not train,
                                 ch, dropout_rate=self.dropout_rate)

                if h.shape[1] in self.attn_resolutions:
                    h = attention_block(h, time_emb)

            if level != 0:
                h = upsample(h, self.resamp_with_conv)

        assert not hs

        h = nn.swish(normalize(h, time_emb))
        h = conv2d(h, self.out_ch, scale=0.0)
        return h
