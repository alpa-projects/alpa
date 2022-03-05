from utils import logger
from visualizations import show, masks

import imageio
import random
import numpy as np
import os
import tqdm
import tensorflow as tf
import multiprocessing

class DataType:
    TRAIN, TEST, VAL = 'train', 'test', 'val'

    @staticmethod
    def get():
        return list(map(lambda x: DataType.__dict__[x], list(filter(lambda k: not k.startswith("__") and type(DataType.__dict__[k]) == str, DataType.__dict__))))


class Dataset(object):

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def summary(self):
        logger.info("======================%s========================" % self.__class__.__name__)
        logger.info("dataset has %d classes" % self.num_classes)
        logger.info("labels: %s" % self.labels)
        examples = [(data_type, self.num_examples(data_type)) for data_type in [DataType.TRAIN, DataType.VAL, DataType.TEST]]
        logger.info("examples: %s" % str(examples))
        logger.info("total: %d" % sum([l for _, l in examples]))
        logger.info("================================================")

    @property
    def labels(self):
        return []

    @property
    def num_classes(self):
        return len(self.labels)

    def raw(self):
        return {
            DataType.TRAIN: [],
            DataType.VAL: [],
            DataType.TEST: []
        }

    def parse_example(self, example):
        return list(map(imageio.imread, example))

    def num_examples(self, data_type):
        return len(self.raw()[data_type])

    def total_examples(self):
        return sum([self.num_examples(dt) for dt in DataType.get()])

    def get_random_item(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]
        example = random.choice(data)
        return self.parse_example(example)

    def show_random_item(self, data_type=DataType.TRAIN):
        example = self.get_random_item(data_type=data_type)
        show.show_images([example[0], example[1].astype(np.float32)])

    def save_random_item(self, data_type=DataType.TRAIN, image_path='image.png', mask_path='mask.png', mask_mode='rgb', alpha=0.5):
        assert(mask_mode in ['gray', 'rgb', 'overlay']), 'mask mode must be in %s' % str(['gray', 'rgb', 'overlay'])
        item = self.get_random_item(data_type=data_type)
        imageio.imwrite(image_path, item[0])
        
        mask = item[1]
        if mask_mode == 'rgb':
            image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colors = masks.get_colors(self.num_classes)
            mask = masks.overlay_classes(image, mask, colors, self.num_classes, alpha=1.0)

        elif mask_mode == 'overlay':
            colors = masks.get_colors(self.num_classes)
            mask = masks.overlay_classes(item[0], mask, colors, self.num_classes, alpha=alpha)

        elif mask_mode == 'gray': # scale mask to 0 to 255
            mask = (mask * (255. / (self.num_classes - 1))).astype(np.uint8)

        imageio.imwrite(mask_path, mask)

    def get(self, data_type=DataType.TRAIN):
        data = self.raw()[data_type]

        def gen():
            for example in data:
                try:
                    yield self.parse_example(example)
                except:
                    logger.error("could not read either one of these files %s" % str(example))
        return gen


def write_labels(path, labels):
    with open(path, 'w') as writer:
        for label in labels:
            writer.write(label.strip() + "\n")

def read_labels(path):
    with open(path, 'r') as reader:
        lines = reader.readlines()
        labels = list(map(lambda x: x[:-1], lines))
    return labels

from enum import IntEnum
class ColorMode(IntEnum):
    RGB, GRAY, NONE = range(3)

def select_patch(image, mask, patch_size, color_mode):
    """
    Select a random patch on image, mask at the same location

    Args:
        image (tf.Tensor): Tensor for the input image of shape (h x w x (1 or 3)), dtype: float32
        mask (tf.Tensor): Tensor for the mask image of shape (h x w x 1), dtype: uint8
        patch_size (tuple): Size of patch (height, width)
    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Tuple of tensors (image, mask) with shape (patch_size[0], patch_size[1], 3)
    """
    image = tf.image.convert_image_dtype(image, tf.uint8)
    if color_mode == ColorMode.RGB:
        # use alpha channel for mask
        concat = tf.concat([image, mask], axis=-1)
        patches = tf.image.random_crop(concat, size=[patch_size[0], patch_size[1], 4])
        patch_image = tf.image.convert_image_dtype(patches[:, :, :3], tf.float32)
        patch_mask = tf.expand_dims(patches[:, :, 3], axis=-1)
    else:
        stack = tf.stack([image, mask], axis=0)
        patches = tf.image.random_crop(stack, size=[2, patch_size[0], patch_size[1], 1])
        patch_image = patches[0]
        patch_mask = patches[1]
    
    return (patch_image, patch_mask)

def resize_and_change_color(image, mask, size, color_mode, resize_method='resize_with_pad', mode='graph'):
    """
    Arguments:

    - image: 3d or 4d tensor
    - mask: 2d, 3d tensor or None
    - size: new_height, new_width
    - color_mode: ColorMode
    - resize_method: how to resize the images, options: 'resize', 'resize_with_pad', 'resize_with_crop_or_pad'

    Returns:
    tuple (image, mask)
    """
    if mode == 'graph':
        # len(tf.shape(image)) == 1 and tf.shape(image)[0] == 2 seems to be a wierd hack when width and height
        # are not defined
        is2d = False
        if len(tf.shape(image)) == 2 or (len(tf.shape(image)) == 1 and tf.shape(image)[0] == 2):
            image = tf.expand_dims(image, axis=-1)
            is2d = True

        if color_mode == ColorMode.RGB and (tf.shape(image)[-1] == 1 or is2d):
            image = tf.image.grayscale_to_rgb(image)

        elif color_mode == ColorMode.GRAY and tf.shape(image)[-1] != 1:
            image = tf.image.rgb_to_grayscale(image)

    elif mode == 'eager':
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        
        if color_mode == ColorMode.RGB and image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)

        elif color_mode == ColorMode.GRAY and image.shape[-1] != 1:
            image = tf.image.rgb_to_grayscale(image)
    else:
        raise Exception("unknown mode %s" % mode)
    if size is not None:
        # make 3dim for tf.image.resize
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)

        # augmentatations
        if resize_method == 'resize':
            image = tf.image.resize(image, size, antialias=True)
            if mask is not None:
                mask = tf.image.resize(mask, size, method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_pad':
            image = tf.image.resize_with_pad(image, size[0], size[1], antialias=True)
            if mask is not None:
                mask = tf.image.resize_with_pad(mask, size[0], size[1], method='nearest')  # use nearest for no interpolation

        elif resize_method == 'resize_with_crop_or_pad':
            image = tf.image.resize_with_crop_or_pad(image, size[0], size[1])
            if mask is not None:
                mask = tf.image.resize_with_crop_or_pad(mask, size[0], size[1])  # use nearest for no interpolation
        
        elif resize_method == 'patch':
            image, mask = select_patch(image, mask, size, color_mode)
        
        else:
            raise Exception("unknown resize method %s" % resize_method)

        # reduce dim added before
        if mask is not None:
            mask = tf.squeeze(mask, axis=-1)

    return image, mask

def convert2tfdataset(dataset, data_type, randomize=True):
    """
    Converts a tf_semantic_segmentation.datasets.dataset.Dataset class to tf.data.Dataset which returns a tuple
    (image, mask, num_classes)

    Arguments:
        dataset (tf_semantic_segmentation.datasets.dataset.Dataset): dataset to convert
        data_type (tf_semantic_segmentation.datasets.dataset.DataType): one of (train, val, test)
        randomize (bool): randomize the order of the paths, otherwise every epoch it yields the same
    Returns:
        tf.data.Dataset
    """

    def gen():
        indexes = np.arange(dataset.num_examples(data_type))
        if randomize:
            indexes = np.random.permutation(indexes)

        data = dataset.raw()[data_type]
        for idx in indexes:
            example = data[idx]
            try:
                image, mask = dataset.parse_example(example)
            except Exception as e:
                logger.warning("could not parse example %s, error: %s, skipping" % (str(example), str(e)))
                continue

            shape = image.shape

            if len(shape) == 2:
                continue
                # print("len shape == 2")
                shape = [shape[0], shape[1], 1]

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            if len(mask.shape) > 2:
                logger.warning("mask of example %s has invalid shape: %s, skipping" % (str(example), str(mask.shape)))
                continue

            yield image, mask, dataset.num_classes, shape

    def map_fn(image, mask, num_classes, shape):
        image = tf.reshape(image, shape)
        mask = tf.reshape(mask, (shape[0], shape[1]))
        return image, mask, num_classes

    if hasattr(dataset, 'tfdataset'):
        logger.info("convert2tfdataset is using tfdataset optimized wrapper function")
        ds = dataset.tfdataset(data_type, randomize=randomize)
    else:
        ds = tf.data.Dataset.from_generator(
            gen, (tf.uint8, tf.uint8, tf.int64, tf.int64), ([None, None, None], [None, None], [], [3]))
        ds = ds.map(map_fn, num_parallel_calls=multiprocessing.cpu_count())

    #ds[0] = tf.reshape(ds[0], ds[2])
    #ds[1] = tf.reshape(ds[1], [ds[2][0], ds[2][1], tf.convert_to_tensor(dataset.num_classes, tf.int64)])
    return ds

def dataloader(ds, output_dir=None, size=None, resize_method="resize_with_pad", color_mode=ColorMode.NONE, overwrite=False, batch_size=32):

    # os.makedirs(output_dir, exist_ok=True)
    # write_labels(os.path.join(output_dir, 'labels.txt'), ds.labels)

    def preprocess_fn(image, mask, num_classes=80):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image, mask = resize_and_change_color(image, mask, (384, 384), color_mode, resize_method)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        return image, mask
    
    datasets = []
    for data_type in [DataType.TRAIN, DataType.VAL]:
        # if data_type != DataType.VAL:
        #     continue

        # masks_dir = os.path.join(output_dir, data_type, 'masks')
        # images_dir = os.path.join(output_dir, data_type, 'images')

        # os.makedirs(masks_dir, exist_ok=True)
        # os.makedirs(images_dir, exist_ok=True)

        tfds = convert2tfdataset(ds, data_type)
        tfds = tfds.map(preprocess_fn, num_parallel_calls=multiprocessing.cpu_count())
        tfds = tfds.batch(batch_size, drop_remainder=True)
        datasets.append(tfds)

    return tuple(datasets)
        # total_examples = ds.num_examples(data_type)    
        # for k, (images, masks) in tqdm.tqdm(enumerate(tfds), total=total_examples // batch_size, desc="exporting",
        #                                     postfix=dict(data_type=data_type, batch_size=batch_size, total=total_examples)):
        #     # print(images.shape) #(32, 384, 384, 3)
        #     # print(masks.shape)  #(32, 384, 384)
        #     pass
        #     # exit()