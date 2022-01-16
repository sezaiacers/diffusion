import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 61

    config.pretrained_dir = '.'
    config.checkpoint_every = 1_000

    config.dataset = ''
    config.tfds_manual_dir = None
    config.tfds_data_dir = None
    config.image_size = 128
    config.ext = 'png'
    config.train_split = 'train'
    config.val_split = 'test'
    config.one_hot = False

    config.shuffle_buffer = 50_000
    config.prefetch = 2


    config.batch = 512
    config.batch_eval = 512

    config.num_classes = 1
    config.out_ch = 3
    config.ch = 128
    config.ch_mult = (1, 1, 2, 2, 4, 4)
    config.num_res_blocks = 2
    config.attn_resolutions = [16]
    config.dropout_rate = 0.0
    config.resamp_with_conv = False

    config.lr = 2e-5
    config.eps = 1e-8
    config.ema_decay = 0.9999
    config.grad_norm_clip = 1.0

    config.loss_type = 'MSE'
    config.mean_type = 'EPSILON'
    config.var_type = 'FIXED_LARGE'

    config.schedule_type = 'linear'
    config.start = 0.0001
    config.end = 0.02
    config.num_timesteps = 1000

    config.total_steps = 2000
    config.progress_every = 10
    config.checkpoint_every = 100
    config.eval_every = 50
    config.generate_every = 100

    return config
