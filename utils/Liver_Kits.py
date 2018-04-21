import os
import os.path as osp
from glob import glob
import numpy as np
import pickle

METType = {
    'MET_CHAR': np.char,
    'MET_SHORT': np.int16,
    'MET_LONG': np.int32,
    'MET_INT': np.int32,
    'MET_UCHAR': np.uint8,
    'MET_USHORT': np.uint16,
    'MET_ULONG': np.uint32,
    'MET_UINT': np.uint32,
    'MET_FLOAT': np.float32,
    'MET_FLOAT': np.float64
}

def extract_data(SrcDir, mode, DstDir_o, DstDir_m, worker, origin, mask, name=None):
    """ Convert vol data into mhd data

    Params
    ------
    `SrcDir`: source directory
    `mode`: 0 or 1, extract 3D volume or 2D slices
    `DstDir_o`: destination directory to store origin image
    `DstDir_m`: destination directory to store mask image
    `worker`: converter path
    """
    SrcDirs = os.listdir(SrcDir)

    for i, src in enumerate(SrcDirs):
        if name:
            dst = "{:s}{:03d}".format(name, i)
        else:
            dst = src
        src_liver = osp.join(SrcDir, src, origin)
        src_liver_mask = osp.join(SrcDir, src, mask)

        dst_liver = osp.join(DstDir_o, (dst + "_o") if mode == 1 else dst)
        dst_liver_mask = osp.join(DstDir_m, dst + "_m")
        if mode == 0:
            os.system(worker + " 0 1 " + src_liver + " " + dst_liver)
            os.system(worker + " 0 0 " + src_liver_mask + " " +  dst_liver_mask)
        elif mode == 1:
            os.system(worker + " 1 1 " + src_liver + " " + dst_liver)
            os.system(worker + " 1 0 " + src_liver_mask + " " +  dst_liver_mask)
        else:
            raise ValueError("Wrong mode.")

def mhd_reader(mhdpath, only_meta=False):
    """ Implementation of `.mhd` file reader

    Params
    ------
    `mhdpath`: file path to a mhd file
    `only_meta`: if True, raw image will not be loaded and the second return is None

    Returns
    -------
    `meta_info`: a dictonary contains all the information in mhd file
    `raw_image`: raw data of this image. If `only_meta` is True, this return will be None.  

    Note: the returned `raw_image` is read-only.
    """
    meta_info = {}
    # read .mhd file 
    with open(mhdpath, 'r') as fmhd:
        for line in fmhd.readlines():
            parts = line.split()
            meta_info[parts[0]] = ' '.join(parts[2:])
    
    PrimaryKeys = ['NDims', 'DimSize', 'ElementType', 'ElementSpacing', 'ElementDataFile']
    for key in PrimaryKeys:
        if not key in meta_info:
            raise KeyError("Missing key `{}` in meta data of the mhd file".format(key))

    meta_info['NDims'] = int(meta_info['NDims'])
    meta_info['DimSize'] = [eval(ele) for ele in meta_info['DimSize'].split()]
    meta_info['ElementSpacing'] = [eval(ele) for ele in meta_info['ElementSpacing'].split()]
    #meta_info['ElementByteOrderMSB'] = eval(meta_info['ElementByteOrderMSB'])

    raw_image = None
    if not only_meta:
        rawpath = osp.join(osp.dirname(mhdpath), meta_info['ElementDataFile'])

        # read .raw file
        with open(rawpath, 'rb') as fraw:
            buffer = fraw.read()
    
        raw_image = np.frombuffer(buffer, dtype=METType[meta_info['ElementType']])
        raw_image = np.reshape(raw_image, list(reversed(meta_info['DimSize'])))
        
    return meta_info, raw_image 

def mhd_writer(mhdpath, image:np.ndarray):
    """ Implementation of `.mhd` file writer

    Params
    ------
    `mhdpath`: file path to write at
    `image`: image to write
    """
    image = np.squeeze(image)

    meta_info = {}
    meta_info["NDims"] = image.ndim
    meta_info["DimSize"] = reversed(image.shape)
    raise NotImplementedError

def bbox_from_mask(mask, bk_value=None):
    """ Calculate bounding box from a mask image 
    """
    if bk_value is None:
        bk_value = mask[0, 0]
    mask_pixels = np.where(mask > bk_value)
    if mask_pixels[0].size == 0:
        return None
    
    bbox = [
        np.min(mask_pixels[1]),
        np.min(mask_pixels[0]),
        np.max(mask_pixels[1]),
        np.max(mask_pixels[0])
    ]

    return bbox

def get_mhd_list(SrcDir):
    if not osp.exists(SrcDir):
        raise FileNotFoundError("{} can not found!".format(SrcDir))
        
    mhd_list = glob(osp.join(SrcDir, "*.mhd"))
    return mhd_list

def get_mhd_list_with_liver(SrcDir, verbose=False):
    """ Get mhd files list in a specific directory and remove the slices
    which does not contain liver.

    Note: SrcDir should be a mask dir 
    """
    cache_file = osp.join(SrcDir, "liver_slices.pkl")
    if osp.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            try:
                keep_mhd_list = pickle.load(fid)
            except:
                keep_mhd_list = pickle.load(fid, encoding='bytes')
        print("mhd list loaded from {}".format(cache_file))
        return keep_mhd_list
    
    all_mhd_list = get_mhd_list(SrcDir)
    keep_mhd_list = []
    for mhdfile in all_mhd_list:
        if verbose:
            print(mhdfile)
        _, raw = mhd_reader(mhdfile)
        bbox = bbox_from_mask(raw)
        if bbox:
            keep_mhd_list.append(mhdfile)
    
    with open(cache_file, 'wb') as fid:
        pickle.dump(keep_mhd_list, fid, pickle.HIGHEST_PROTOCOL)
    print("Write mhd list to {}".format(cache_file))

    return keep_mhd_list

if __name__ == '__main__':
    if False:
        SrcDir = "D:/DataSet/LiverQL/Liver-Ref/"
        SrcDir_o = "D:/DataSet/LiverQL/Liver_slices_train/liver/"
        SrcDir_m = "D:/DataSet/LiverQL/Liver_slices_train/mask/"
        extract_slices(SrcDir, SrcDir_o, SrcDir_m)
    
    if False:
        SrcDir_m = "D:/DataSet/LiverQL/3Dircadb1_slices_train/mask/"
        print(len(get_mhd_list_with_liver(SrcDir_m, False)))

    if False:
        SrcDir = "D:/DataSet/LiverQL/Target-New-Training"
        worker = "D:/DataSet/LiverQL/VolConverter.exe"
        DstDir_o = "D:/DataSet/LiverQL/Liver_2018_train_3D/liver"
        DstDir_m = "D:/DataSet/LiverQL/Liver_2018_train_3D/mask"
        origin = "Study_Phase2.vol"
        mask = "Study_Phase2_Label.vol"
        extract_data(SrcDir, 0, DstDir_o, DstDir_m, worker, origin, mask, name="R")

    if True:
        SrcDir = "D:/DataSet/LiverQL/Target-New-Training"
        worker = "D:/DataSet/LiverQL/VolConverter.exe"
        DstDir_o = "D:/DataSet/LiverQL/Liver_2018_train/liver"
        DstDir_m = "D:/DataSet/LiverQL/Liver_2018_train/mask"
        origin = "Study_Phase2.vol"
        mask = "Study_Phase2_Label.vol"
        extract_data(SrcDir, 1, DstDir_o, DstDir_m, worker, origin, mask, name="R")
