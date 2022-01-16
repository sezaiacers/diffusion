import glob
import os

from absl import logging
import flax
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import sys
if sys.platform != 'darwin':
  import resource
  low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

def get_tfds_info(dataset, split, ext=None):
    info = tfds.builder(dataset).info

    num_examples = info.splits[split].num_examples
    num_classes = info.features['label'].num_classes if 'label' in info.features else 1
    int2str = info.features['label'].int2str if 'label' in info.features else None

    return dict(
        num_examples=num_examples,
        num_classes=num_classes,
        int2str=int2str,
        examples_glob=None
    )

def get_directory_info(directory, ext):
    examples_glob = f'{directory}/*/*.{ext}'
    paths = glob.glob(examples_glob)

    return dict(
        num_examples=len(paths),
        num_classes=1,
        int2str=None,
        examples_glob=examples_glob,
    )

def get_dataset_info(dataset, split, ext=None):
    directory = os.path.join(dataset, split)
    if os.path.isdir(directory):
        return get_directory_info(directory, ext)
    return get_tfds_info(dataset, split)

def get_datasets(config):
    if os.path.isdir(config.dataset):
        train_dir = os.path.join(config.dataset, config.train_split)
        test_dir = os.path.join(config.dataset, config.val_split)
        if not os.path.isdir(train_dir):
            raise ValueError('Expected to find directories"{}" and "{}"'.format(
                train_dir,
                test_dir,
            ))
        logging.info('Reading dataset from directories "%s" and "%s"', train_dir,
                     test_dir)
        ds_train = get_data_from_directory(
            config=config, directory=train_dir, mode=config.train_split, ext=config.ext)
        ds_test = get_data_from_directory(
            config=config, directory=test_dir, mode=config.val_split, ext=config.ext)
    else:
        logging.info('Reading dataset from tfds "%s"', config.dataset)
        ds_train = get_data_from_tfds(config=config, mode=config.train_split)
        ds_test = get_data_from_tfds(config=config, mode=config.val_split)

    return ds_train, ds_test

def get_data_from_tfds(*, config, mode):
    data_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)

    data_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            manual_dir=config.tfds_manual_dir))

    data = data_builder.as_dataset(
        split=mode,
        decoders={'image': tfds.decode.SkipDecoding()},
        shuffle_files=mode == config.train_split)

    image_decoder = data_builder.info.features['image'].decode_example
    dataset_info = get_tfds_info(config.dataset, mode)
    return get_data(
        data=data,
        mode=mode,
        num_classes=dataset_info['num_classes'],
        image_decoder=image_decoder,
        repeats=None if mode == 'train' else 1,
        batch_size=config.batch_eval if mode == config.val_split else config.batch,
        image_size=config.image_size,
        shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
        one_hot=config.one_hot)

def get_data_from_directory(*, config, directory, mode, ext):
    dataset_info = get_directory_info(directory, ext)
    data = tf.data.Dataset.list_files(dataset_info['examples_glob'])

    image_decoder = lambda path: tf.image.decode_jpeg(tf.io.read_file(path), 3)
    return get_data(
        data=data,
        mode=mode,
        num_classes=dataset_info['num_classes'],
        image_decoder=image_decoder,
        repeats=None if mode == 'train' else 1,
        batch_size=config.batch_eval if mode == 'test' else config.batch,
        image_size=config.image_size,
        shuffle_buffer=min(dataset_info['num_examples'], config.shuffle_buffer),
        one_hot=config.one_hot,
        preprocess=lambda path: dict(image=path))

def get_data(*,
             data,
             mode,
             num_classes,
             image_decoder,
             repeats,
             batch_size,
             image_size,
             shuffle_buffer,
             one_hot,
             preprocess=None):

    def _pp(data):
        image = image_decoder(data['image'])

        if image.shape[-1] == 1:
            image= tf.repeat(im, 3, axis=-1)

        image = tf.image.resize(image, [image_size, image_size])
        image = (image - 127.5) / 127.5

        label = data['label'] if 'label' in data else 0
        label = tf.one_hot(label, num_classes) if one_hot else label
        return dict(image=image, label=label)

    data = data.repeat(repeats)
    if mode == 'train':
        data = data.shuffle(shuffle_buffer)

    if preprocess is not None:
        data = data.map(preprocess, tf.data.experimental.AUTOTUNE)
    data = data.map(_pp, tf.data.experimental.AUTOTUNE)
    data = data.batch(batch_size, drop_remainder=True)

    num_devices = jax.local_device_count()
    def _shard(data):
        data['image'] = tf.reshape(data['image'],
                                   [num_devices, -1, image_size, image_size, 3])
        if one_hot:
            data['label'] = tf.reshape(data['label'],
                                       [num_devices, -1, num_classes])
        else:
            data['label'] = tf.reshape(data['label'], [num_devices, -1])
        return data

    if num_devices is not None:
        data = data.map(_shard, tf.data.experimental.AUTOTUNE)
    return data.prefetch(1)

def prefetch(dataset, n_prefetch):
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                  ds_iter)
    if n_prefetch:
        ds_iter = flax.jax_utils.prefetch_to_device(ds_iter, n_prefetch)
    return ds_iter
