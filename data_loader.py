from data_layer import DataLoader
import numpy as np
from config import cfg
import os.path as osp
from utils.Liver_Kits import get_mhd_list_with_liver, mhd_reader

class FullImageLoader(DataLoader):
    """ Dataset specified data loader.

    Params
    ------
    `datadir`: dataset directories, split with `+`. For example: "trainset1+trainset2"
    `once`: just loop once or not
    `random`: if random flag is set, then the dataset is shuffled according to system
    time. Useful for the validation set.
    """
    def __init__(self, datadir, once=False, random=False):
        datadirs = datadir.split("+")
        self._db_path = []
        for d in datadirs:
            self._db_path.append(osp.join(cfg.DATA.ROOT_DIR, d))
        self._images = get_mhd_list_with_liver(osp.join(self._db_path[0], "mask"))
        for path in self._db_path[1:]:
            self._images.extend(get_mhd_list_with_liver(osp.join(path, "mask")))
        self._batch_size = cfg.TRAIN.BS
        self._height = 512
        self._width = 512

        super(FullImageLoader, self).__init__(once, random)

    def next_minibatch(self, db_inds):
        assert len(db_inds) == self._batch_size

        images = np.zeros((self._batch_size, self._height, self._width, cfg.IMG_CHANNEL), 
                            dtype=np.float32)
        masks = np.zeros_like(image, dtype=np.int32)
        image_names = []
        for i, ind in enumerate(db_inds):
            mask_file = self._images[ind]
            image_file = self._images[ind].replace("mask", "liver").replace("_m_", "_o_")
            _, mask = mhd_reader(mask_file)
            _, image = mhd_reader(image_filea)
            mask = np.reshape(mask, (self._height, self._width, cfg.IMG_CHANNEL))
            image = np.reshape(image, (self._height, self._width, cfg.IMG_CHANNEL))
            masks[i,...] = (mask / np.max(mask)).astype(np.int32)
            images[i,...] = image

            name = osp.basename(image_file).replace("_o_", "_p_")
            image_names.append(name)
        # nomalize to [-1, 1]
        images = np.clip(images / 1024., -1., 1.)

        blob = {"images": images, "labels": masks, "names": image_names}
        return blob
