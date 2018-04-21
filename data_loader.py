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
    `wwidth`: medical image window width
    `wlevel`: medical image window level
    `img_channel`: image channel, default is 1 (i.e. gray image)
    `once`: just loop once or not
    `random`: if random flag is set, then the dataset is shuffled(if set) according to 
    system time, else only with numpy random seed.  
    `shuffle`: a boolean, shuffle dataset or not. If shuffle flag is not set, then `random`
    will be disabled.

    Note: For experiemnts repeatable, please set fixed cfg.RNG_SEED and random=False.
    """
    def __init__(self, rootdir, datadir, batch_size, 
                 wwidth=None, 
                 wlevel=None, 
                 img_channel=1, 
                 once=False, 
                 random=False,
                 shuffle=False):
        datadirs = datadir.split("+")
        self._db_path = []
        for d in datadirs:
            self._db_path.append(osp.join(rootdir, d))
        self._images = get_mhd_list_with_liver(osp.join(self._db_path[0], "mask"))
        for path in self._db_path[1:]:
            self._images.extend(get_mhd_list_with_liver(osp.join(path, "mask")))
        self._batch_size = batch_size
        self._wwidth = wwidth
        self._wlevel = wlevel
        self._img_channel = img_channel
        self._height = 512
        self._width = 512

        super(MedImageLoader2D, self).__init__(once, random, shuffle)

    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width

    @property
    def channel(self):
        return self._img_channel

    def next_minibatch(self, db_inds):
        assert len(db_inds) == self._batch_size

        images = np.zeros((self._batch_size, self.height, self.width, self.channel), 
                            dtype=np.float32)
        masks = np.zeros_like(images, dtype=np.int32)
        image_names = []
        for i, ind in enumerate(db_inds):
            mask_file = self.images[ind]
            image_file = mask_file.replace("mask", "liver").replace("_m_", "_o_")
            _, mask = mhd_reader(mask_file)
            _, image = mhd_reader(image_file)
            mask = np.reshape(mask.copy(), (self.height, self.width, self.channel))
            image = np.reshape(image, (self.height, self.width, self.channel))
            thresh = -1000 if mask[0, 0, 0] < -1000 else 0
            mask[mask > thresh] = 1
            mask[mask < thresh] = 0
            masks[i,...] = mask.astype(np.int32)
            images[i,...] = image

            name = osp.basename(mask_file).replace("_m_", "_p_")
            image_names.append(name)
        # set window width and level
        widd2 = self._wwidth / 2
        images = (np.clip(images, self._wlevel - widd2, self._wlevel + widd2) - 
                  (self._wlevel - widd2)) / 2**16 * self._wwidth

        blob = {"images": images, "labels": masks, "names": image_names}
        return blob

class MedImageLoader3D(DataLoader):
    """ 3D medical image data loader.

    Params
    ------
    `rootdir`: data root directory
    `datadir`: dataset directories, split with `+`. For example: "trainset1+trainset2"
    `batch_size`: batch size of image loader, only support 1
    `wwidth`: medical image window width
    `wlevel`: medical image window level
    `img_channel`: image channel, default is 1 (i.e. gray image)
    `once`: just loop once or not
    `random`: if random flag is set, then the dataset is shuffled(if set) according to 
    system time, else only with numpy random seed.  
    `shuffle`: a boolean, shuffle dataset or not. If shuffle flag is not set, then `random`
    will be disabled.

    Note: For experiemnts repeatable, please set fixed cfg.RNG_SEED and random=False.
    """
    def __init__(self, rootdir, datadir, batch_size, 
                 wwidth=None,
                 wlevel=None,
                 img_channel=1, 
                 once=False, 
                 random=False,
                 shuffle=False):
        datadirs = datadir.split('+')
        self._db_path = []
        for d in datadirs:
            self._db_path.append(osp.join(rootdir, d))
        self._images = get_mhd_list(osp.join(self._db_path[0], "mask"))
        for path in self._db_path[1:]:
            self._images.extend(get_mhd_list(osp.join(path, "mask")))
        self._batch_size = batch_size
        self._wwidth = wwidth
        self._wlevel = wlevel
        self._volume_channels = img_channel
        self._height = 512
        self._width = 512

        super(MedImageLoader3D, self).__init__(once, random, shuffle)
 
    @property
    def height(self):
        return self._height
    
    @property
    def width(self):
        return self._width

    @property
    def channels(self):
        return self._volume_channels

    def next_minibatch(self, db_inds):
        assert len(db_inds) == self.batch_size

        images = []
        masks = []
        image_names = []
        meta_datas = []
        for i, ind in enumerate(db_inds):
            mask_file = self.images[ind]
            image_file = mask_file.replace("mask", "liver").replace("_m", "")
            _, mask = mhd_reader(mask_file)
            meta_data, image = mhd_reader(image_file)
            mask = np.reshape(mask.copy(), (-1, self.height, self.width, self.channels))   # depth is finally determined
            thresh = -1000 if mask[0, 0, 0] < -1000 else 0
            mask[mask > thresh] = 1
            mask[mask < thresh] = 0
            mask = mask.astype(np.int32)
            image = np.reshape(image, (-1, self.height, self.width, self.channels))
            # set window width and level
            widd2 = self._wwidth / 2
            image = (np.clip(image, self._wlevel - widd2, self._wlevel + widd2) - 
                      (self._wlevel - widd2)) / 2**16 * self._wwidth
            images.append(image)
            masks.append(mask)

            name = osp.basename(mask_file).replace("_m", "_p")
            image_names.append(name)
            meta_datas.append(meta_data)
        
        blob = {"images": images, "labels": masks, "names": image_names, "meta": meta_datas}

        return blob
