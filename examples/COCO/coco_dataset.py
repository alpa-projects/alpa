from utils import DataType
from utils import download_and_extract, get_files, parallize_v2

import json
import os
import tqdm
import numpy as np
import cv2
import base64
import imageio

from ray import util
from utils import logger
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

def dataloader(ds, output_dir=None, size=None, resize_method="resize_with_pad", color_mode=ColorMode.NONE, overwrite=False, batch_size=32, num_epochs=1):

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
        # tfds = tfds.repeat(num_epochs)
        tfds = tfds.prefetch(10)
        datasets.append(tfds)

    return tuple(datasets)
        # total_examples = ds.num_examples(data_type)    
        # for k, (images, masks) in tqdm.tqdm(enumerate(tfds), total=total_examples // batch_size, desc="exporting",
        #                                     postfix=dict(data_type=data_type, batch_size=batch_size, total=total_examples)):
        #     # print(images.shape) #(32, 384, 384, 3)
        #     # print(masks.shape)  #(32, 384, 384)
        #     pass
        #     # exit()

class CocoAnnotationReader:

    def __init__(self, json_path):
        self.json_path = json_path
        self.categories = self.read_categories()
        self.mode_wrapper = {
            c['id']: c for c in self.categories
        }
        self.supercategories = list(set(list(map(lambda x: x['supercategory'], self.categories))))

    def get_labels(self, mode='category'):
        if mode == 'supercategory':
            return self.supercategories
        elif mode == 'binary':
            return ['false', 'true']
        elif mode == 'category':
            return list(map(lambda x: x['name'], self.mode_wrapper.values()))
        else:
            raise Exception("unknown mode %s" % mode)
    
    def get_labels_size(self, mode='category'):
        return len(self.get_labels(mode))

    def read_annotations(self):
        data = json.load(open(self.json_path, 'r'))
        annotations = {}

        for info in tqdm.tqdm(data['images'], desc='reading image annotation data'):
            annotations[info['id']] = info
            annotations[info['id']]['segmentations'] = []
        cnt = 0
        for ann in tqdm.tqdm(data['annotations'], desc='reading segmentations data'):
            annotations[ann['image_id']]['segmentations'].append(ann)
            # print(ann)
            # cnt += 1
            # if cnt == 5:
            #     exit()

        cnt = 0
        data = {}
        for image_id, d in annotations.items():
            cnt += 1
            data[d['file_name']] = d
        return data

    def read_categories(self):
        return json.load(open(self.json_path, 'r'))['categories']

    def generate_masks(self, image_path, anns, output_dir, data_dir=None, mode='category'):
        if data_dir:
            filename = os.path.relpath(image_path, data_dir)
        else:
            filename = os.path.basename(image_path)

        output_path = os.path.join(output_dir, filename + "_L.png")

        if not os.path.exists(output_path):
            info = anns[filename]
            mask = np.zeros((info['height'], info['width']), dtype=np.uint8)

            for seg in info['segmentations']:
                category = seg['category_id']
                if mode == 'category':
                    mask_idx = category
                elif mode == 'supercategory':
                    supercategory = self.mode_wrapper[category]['supercategory']
                    mask_idx = self.supercategories.index(supercategory)
                elif mode == 'binary':
                    mask_idx = 1
                else:
                    raise Exception("unknown mode %s" % mode)

                for s in seg['segmentation']:
                    try:
                        cnt = np.asarray(s, dtype=np.int32).reshape((-1, 2))
                        cv2.drawContours(mask, [cnt], 0, mask_idx, thickness=cv2.FILLED)
                    except:
                        pass
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            imageio.imwrite(output_path, mask)

        return output_path


def helper(reader, path, anns, output_dir, mode):
    return reader.generate_masks(path, anns, output_dir, mode=mode)


class Coco(Dataset):

    TRAIN_2014_URL = "http://images.cocodataset.org/zips/train2014.zip"
    VAL_2014_URL = "http://images.cocodataset.org/zips/val2014.zip"
    TEST_2014_URL = "http://images.cocodataset.org/zips/test2014.zip"

    TEST_2015_URL = "http://images.cocodataset.org/zips/test2015.zip"

    TRAIN_2017_URL = "http://images.cocodataset.org/zips/train2017.zip"
    VAL_2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
    TEST_2017_URL = "http://images.cocodataset.org/zips/test2017.zip"

    ANNOTATIONS_2014_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    ANNOTATIONS_2017_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    DATA_URLS = {
        "2014": {
            DataType.TRAIN: TRAIN_2014_URL,
            DataType.TEST: TEST_2014_URL,
            DataType.VAL: VAL_2014_URL,
            "annotations": ANNOTATIONS_2014_URL
        },
        "2017": {
            DataType.TRAIN: TRAIN_2017_URL,
            DataType.TEST: TEST_2017_URL,
            DataType.VAL: VAL_2017_URL,
            "annotations": ANNOTATIONS_2017_URL
        }
    }

    def __init__(self, cache_dir, mode='supercategory', year=2014):
        super(Coco, self).__init__(cache_dir)
        self.year = year
        self.mode = mode
        self.reader = None
        assert(self.year == 2014 or self.year == 2017), "year must be either 2014 or 2017"

    @property
    def labels(self):
        if self.reader is None:
            raise Exception("please run raw() method first before getting the labels from the dataset")
        else:
            return self.reader.get_labels(mode=self.mode)

    @property
    def labels_count(self):
        return len(self.labels)

    def raw(self):

        data_urls = self.DATA_URLS[str(self.year)]

        def down(url):
            encoded = str(base64.b64encode(bytes(url, 'utf-8')), 'utf-8')
            # return
            return download_and_extract(url, os.path.join(self.cache_dir, encoded))

        train_dir = down(data_urls[DataType.TRAIN])
        test_dir = down(data_urls[DataType.TEST])
        val_dir = down(data_urls[DataType.VAL])

        train_images = get_files(train_dir, extensions=['jpg'])
        val_images = get_files(val_dir, extensions=['jpg'])
        test_images = get_files(test_dir, extensions=['jpg'])
        annotations_dir = down(data_urls['annotations'])

        anns_train_reader = CocoAnnotationReader(os.path.join(annotations_dir, 'annotations', 'instances_train%d.json' % self.year))
        self.reader = anns_train_reader
        anns_train = anns_train_reader.read_annotations()

        anns_val_reader = CocoAnnotationReader(os.path.join(annotations_dir, 'annotations', 'instances_val%d.json' % self.year))
        anns_val = anns_val_reader.read_annotations()

        train_masks_output_dir = os.path.join(train_dir, 'masks', self.mode)
        os.makedirs(train_masks_output_dir, exist_ok=True)

        args = [(anns_train_reader, path, anns_train, train_masks_output_dir, self.mode, ) for path in train_images]
        train_masks = parallize_v2(helper, args, desc='generating train masks')

        val_masks_output_dir = os.path.join(val_dir, 'masks')
        os.makedirs(val_masks_output_dir, exist_ok=True)

        args = [(anns_val_reader, path, anns_val, val_masks_output_dir, self.mode, ) for path in val_images]
        val_masks = parallize_v2(helper, args, desc='generating val masks')

        return {
            DataType.TRAIN: list(zip(train_images, train_masks)),
            DataType.VAL: list(zip(val_images, val_masks)),
            DataType.TEST: list(zip(test_images, [None] * len(test_images)))
        }


class Coco2014(Coco):

    def __init__(self, cache_dir):
        super(Coco2014, self).__init__(cache_dir, year=2014)


class Coco2017(Coco):

    def __init__(self, cache_dir):
        super(Coco2017, self).__init__(cache_dir, year=2017)


if __name__ == "__main__":
    ds = Coco2014('/l/users/hexu.zhao/coco_dataset')
    data = ds.raw()[DataType.VAL]

    print("The number of labels is : ", len(ds.labels)) # 12
    print(ds.labels) #['kitchen', 'animal', 'sports', 'furniture', 'outdoor', 'accessory', 'vehicle', 'indoor', 'electronic', 'person', 'appliance', 'food']

    # shape_set = list(set([imageio.imread(x[0]).shape for x in data[:1000]]))
    # print(len(shape_set)) # 218
    # print(shape_set[:10]) # [(468, 640, 3), (640, 317, 3), (640, 520, 3), (387, 640, 3), (270, 360, 3), (640, 523, 3), (640, 468, 3), (218, 640, 3), (500, 333, 3), (640, 426, 3)]

    # data = ds.raw()[DataType.TRAIN]
    # print(len(data[DataType.TRAIN])) # 82783

    # img, label = imageio.imread(data[0][0]), imageio.imread(data[0][1])
    # print(type(img)) # <class 'imageio.core.util.Array'>
    # print(img.shape) # (480, 640, 3)
    # print(type(label))
    # print(label.shape) # (480, 640, 3)

    # dataloader(Coco2014('/home/hexu.zhao/coco_dataset'))