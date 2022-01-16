import jax.numpy as jnp
import jax
import enum


class MeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class VarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

class GaussianDiffusion:

    def __init__(self, betas):
        assert (betas > 0).all() and (betas <= 1).all()

        self.betas = betas
        self.num_timesteps = len(betas)

        alphas = 1. - self.betas
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1. / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (
            1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = jnp.log(jnp.append(
            self.posterior_variance[1], self.posterior_variance[1:]))

        self.posterior_mean_coef1 = self.betas * (
            jnp.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * (
            jnp.sqrt(alphas) / (1. - self.alphas_cumprod))

    def q_mean_variance(self, x_0, timesteps):
        #  q(x{t} | x{0})

        broadcast_shape = x_0.shape
        mean = self._extract(self.sqrt_alphas_cumprod, timesteps,
                             broadcast_shape) * x_0
        var  = self._extract(1.- self.alphas_cumprod, timesteps,
                             broadcast_shape)
        log_var = self._extract(self.log_one_minus_alphas_cumprod,
                                timesteps, broadcast_shape)
        return mean, var, log_var

    def q_sample(self, x_0, timesteps, noise):
        #  q(x{t} | x{0})

        broadcast_shape = x_0.shape
        mean = self._extract(self.sqrt_alphas_cumprod, timesteps,
                             broadcast_shape) * x_0
        std = self._extract(self.sqrt_one_minus_alphas_cumprod,
                            timesteps, broadcast_shape)
        return mean + std * noise

    def q_posterior_mean_variance(self, x_0, x_t, timesteps):
        # q(x{t-1} | x{t}, x{0})

        broadcast_shape = x_0.shape
        mean = (
            self._extract(self.posterior_mean_coef1, timesteps,
                          broadcast_shape) * x_0 +
            self._extract(self.posterior_mean_coef2, timesteps,
                          broadcast_shape) * x_t
        )
        var = self._extract(self.posterior_variance, timesteps,
                            broadcast_shape)
        log_var = self._extract(self.posterior_log_variance_clipped,
                                timesteps, broadcast_shape)
        return mean, var, log_var


    def _predict_xstart_from_xprev(self, x_t, timesteps, x_prev):
        broadcast_shape = x_t.shape
        return (
            self._extract(1. / self.posterior_mean_coef1, timesteps,
                          broadcast_shape) * x_prev -
            self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1,
                          timesteps, broadcast_shape) * x_t
        )

    def _predict_xstart_from_eps(self, x_t, timesteps, eps):
        broadcast_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps,
                          broadcast_shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps,
                          broadcast_shape) * eps
        )

    def compute_loss(self,
                     denoise_fn,
                     x_0,
                     timesteps,
                     noise,
                     loss_type,
                     mean_type,
                     var_type):

        assert loss_type == LossType.MSE
        assert var_type != VarType.LEARNED

        x_t = self.q_sample(x_0, timesteps, noise)

        if mean_type == MeanType.PREVIOUS_X:
            target = self.q_posterior_mean_variance(
                x_0, x_t, timesteps)[0]
        elif mean_type == MeanType.START_X:
            target = x_0
        elif mean_type == MeanType.EPSILON:
            target = noise
        else:
            raise NotImplementedError(mean_type)

        output = denoise_fn(x_t, timesteps)
        return jnp.mean((target - output) ** 2)

    def p_mean_variance(self,
                        denoise_fn,
                        x_t,
                        timesteps,
                        mean_type,
                        var_type,
                        clip_denoised):

        # p(x{t-1})

        broadcast_shape = x_t.shape
        ones_like_x_t = jnp.ones_like(x_t)

        output = denoise_fn(x_t, timesteps)
        if var_type == VarType.LEARNED:
            output, log_var = jnp.split(output, 2, -1)
            var = jnp.exp(log_var)
        elif var_type == VarType.FIXED_SMALL:
            var = self._extract(self.posterior_variance, timesteps,
                                broadcast_shape) * ones_like_x_t
            log_var = self._extract(self.posterior_log_variance_clipped,
                                    timesteps, broadcast_shape) * ones_like_x_t
        elif var_type == VarType.FIXED_LARGE:
            var = self._extract(self.betas, timesteps,
                                broadcast_shape) * ones_like_x_t

            log_var = jnp.log(jnp.append(self.posterior_variance[1], self.betas[1:]))
            log_var = self._extract(log_var, timesteps,
                                    broadcast_shape) * ones_like_x_t
        else:
            raise NotImplementedError(var_type)

        clip = lambda x: (jnp.clip(x, -1., 1.) if clip_denoised else x)
        if mean_type == MeanType.PREVIOUS_X:
            x_0 = clip(self._predict_xstart_from_xprev(x_t, timesteps, output))
            mean = output
        elif mean_type == MeanType.START_X:
            x_0 = clip(output)
            mean = self.q_posterior_mean_variance(x_0, x_t, timesteps)[0]
        elif mean_type == MeanType.EPSILON:
            x_0 = clip(self._predict_xstart_from_eps(x_t, timesteps, output))
            mean = self.q_posterior_mean_variance(x_0, x_t, timesteps)[0]
        else:
            raise NotImplementedError(mean_type)
        return mean, var, log_var, x_0


    def p_sample(self,
                 denoise_fn,
                 x_t,
                 timesteps,
                 noise_fn,
                 rng,
                 mean_type,
                 var_type,
                 clip_denoised):

        mean, var, log_var, x_0 = self.p_mean_variance(
            denoise_fn, x_t, timesteps, mean_type, var_type, clip_denoised)
        noise = noise_fn(rng, shape=x_t.shape, dtype=x_t.dtype)

        mask = jnp.reshape(1.0 - jnp.asarray(jnp.equal(timesteps, 0)),
                           [x_t.shape[0]] + [1] * (len(x_t.shape) - 1))
        sample = mean + mask * jnp.exp(0.5 * log_var) * noise
        return sample, x_0

    def p_sample_loop(self,
                      denoise_fn,
                      noise_fn,
                      rng,
                      shape,
                      mean_type,
                      var_type,
                      clip_denoised):

        rng, key = jax.random.split(rng)

        x_t = noise_fn(key, shape=shape, dtype=jnp.float32)
        i_t = self.num_timesteps - 1

        def cond_fun(inputs):
            i_t, x_t, rng = inputs
            return i_t >= 0

        def body_fun(inputs):
            i_t, x_t, rng = inputs
            rng, key = jax.random.split(rng)
            timesteps = jnp.full([shape[0]], i_t)
            xprev = self.p_sample(denoise_fn, x_t,
                                  timesteps, noise_fn, key,
                                  mean_type, var_type, clip_denoised)[0]
            return i_t - 1, xprev, rng
        return jax.lax.while_loop(cond_fun, body_fun, (i_t, x_t, rng))[1]

    @staticmethod
    def _extract(x, timesteps, broadcast_shape):
        shape = [broadcast_shape[0]] + [1] * (len(broadcast_shape) - 1)
        y = jnp.take(x, timesteps)
        return jnp.reshape(y, shape)


def get_beta_schedule(schedule_type, start, end, num_timesteps):

    def warmup_beta(start, end, num_timesteps, warmup_frac):
        betas = end * jnp.ones(num_timesteps)
        warmup_time = int(num_timesteps * warmup_frac)
        return jnp.append(jnp.linspace(start, end, warmup_time),
                          betas[warmup_time:])

    if schedule_type == 'quad':
        return jnp.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule_type == 'linear':
        return jnp.linspace(start, end, num_timesteps)
    elif schedule_type == 'warmup10':
        return warmup_beta(start, end, num_timesteps, 0.1)
    elif schedule_type == 'warmup50':
        return warmup_beta(start, end, num_timesteps, 0.5)
    elif schedule_type == 'const':
        return end * jnp.ones(num_timesteps)
    elif schedule_type == 'jsd':
        return 1. / jnp.linspace(num_timesteps, 1, num_timesteps)
    else:
        raise NotImplementedError(schedule_type)
