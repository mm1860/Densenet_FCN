import os.path as osp

import numpy as np

from data_layer import DataLoader
from utils.Liver_Kits import get_mhd_list, get_mhd_list_with_liver, mhd_reader


class MedImageLoader2D(DataLoader):
    """ 2D medical image data loader.

    Params
    ------
    `rootdir`: data root directory
    `datadir`: dataset directories, split with `+`. For example: "trainset1+trainset2"
    `batch_size`: batch size of image loader
    `img_channel`: image channel, default is 1 (i.e. gray image)
    `once`: just loop once or not
    `random`: if random flag is set, then the dataset is shuffled according to system
    time. Useful for the validation set.
    """
    def __init__(self, rootdir, datadir, batch_size, img_channel=1, once=False, random=False):
        datadirs = datadir.split("+")
        self._db_path = []
        for d in datadirs:
            self._db_path.append(osp.join(rootdir, d))
        self._images = get_mhd_list_with_liver(osp.join(self._db_path[0], "mask"))
        for path in self._db_path[1:]:
            self._images.extend(get_mhd_list_with_liver(osp.join(path, "mask")))
        self._batch_size = batch_size
        self._height = 512
        self._width = 512

        super(MedImageLoader2D, self).__init__(once, random)

    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width

    def next_minibatch(self, db_inds):
        assert len(db_inds) == self._batch_size

        images = np.zeros((self._batch_size, self.height, self.width, img_channel), 
                            dtype=np.float32)
        masks = np.zeros_like(image, dtype=np.int32)
        image_names = []
        for i, ind in enumerate(db_inds):
            mask_file = self.images[ind]
            image_file = mask_file.replace("mask", "liver").replace("_m_", "_o_")
            _, mask = mhd_reader(mask_file)
            _, image = mhd_reader(image_filea)
            mask = np.reshape(mask, (self.height, self.width, img_channel))
            image = np.reshape(image, (self.height, self.width, img_channel))
            masks[i,...] = (mask / np.max(mask)).astype(np.int32)
            images[i,...] = image

            name = osp.basename(mask_file).replace("_m_", "_p_")
            image_names.append(name)
        # nomalize to [-1, 1]
        images = np.clip(images / 1024.0, -1.0, 1.0)

        blob = {"images": images, "labels": masks, "names": image_names}
        return blob

class MedImageLoader3D(DataLoader):
    """ 2D medical image data loader.

    Params
    ------
    `rootdir`: data root directory
    `datadir`: dataset directories, split with `+`. For example: "trainset1+trainset2"
    `batch_size`: batch size of image loader, only support 1
    `img_channel`: image channel, default is 1 (i.e. gray image)
    `once`: just loop once or not
    `random`: if random flag is set, then the dataset is shuffled according to system
    time. Useful for the validation set.
    """
    def __init__(self, rootdir, datadir, batch_size=1, img_channel=1, once=False, random=False):
        datadirs = datadir.split('+')
        self._db_path = []
        for d in datadirs:
            self._db_path.append(osp.join(rootdir, d))
        self._images = get_mhd_list(osp.join(self._db_path[0], "mask"))
        for path in self._db_path[1:]:
            self._images.extend(get_mhd_list(osp.join(path, "mask")))
        self._batch_size = batch_size
        self._height = 512
        self._width = 512

        super(MedImageLoader3D, self).__init__(once, random)
 
    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width

    def next_minibatch(self, db_inds):
        assert len(db_inds) == self.batch_size

        images = []
        masks = []
        image_names = []
        for i, ind in enumerate(db_inds):
            mask_file = self.images[ind]
            image_file = mask_file.replace("mask", "liver").replace("_m_", "")
            _, mask = mhd_reader(mask_file)
            _, liver = mhd_reader(image_file)
            mask = np.reshape(mask, (-1, self.height, self.width))
            mask = (mask / np.max(mask)).astype(np.int32)
            image = np.reshape(image, (-1, self.height, self.width))
            # normalize to [-1, 1]
            image = np.clip(image / 1024.0, -1.0, 1.0)
            images.append(image)
            masks.append(mask)

            name = osp.basename(mask_file).replace("_m_", "_p_")
            image_names.append(name)
        
        blob = {"images": images, "labels": masks, "names": image_names}

        return blob
