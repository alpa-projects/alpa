from utils import DataType
from dataset import Dataset, dataloader
from utils import download_and_extract, get_files
from my_threading import parallize_v2

import json
import os
import tqdm
import numpy as np
import cv2
import base64
import imageio

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