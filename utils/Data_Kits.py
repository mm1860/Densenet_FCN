import matplotlib.pyplot as plt
from Liver_Kits import mhd_reader
from glob import glob
import os.path as osp
import cv2
import re
import numpy as np
import skimage.measure as measure

def show_a_pred(image, mask, pred, save_path=None, contour=True, mask_thresh=-500, alpha=0.3):
    fig = plt.figure()
    fig.set_size_inches((12, 6))

    if not contour: # mask
        image_pred = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        preded = np.where(pred > 125)
        image_pred[preded[0], preded[1]] = (1 - alpha) * image_pred[preded[0], preded[1]] + alpha * np.array([255, 255, 0])
        image_mask = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        masked = np.where(mask > 0)
        image_mask[masked[0], masked[1]] = (1 - alpha) * image_mask[masked[0], masked[1]] + alpha * np.array([255, 0, 255])
    else:   # contour
        image_pred = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        pred_contours = measure.find_contours(pred, 128)
        for cont in pred_contours:
            cont = cont.astype(np.int32)
            image_pred[cont[:,0], cont[:,1]] = np.array([0, 255, 0])
        image_mask = np.repeat(image[:,:,np.newaxis], 3, axis=2)
        masked = mask.copy()
        masked[masked > 0] = 255
        masked[masked < 0] = 0
        mask_contours = measure.find_contours(masked, 128)
        for cont in mask_contours:
            cont = cont.astype(np.int32)
            image_mask[cont[:,0], cont[:,1]] = np.array([255, 0, 0])

    images = [image_mask, image_pred]
    
    last = 0.0
    for i, w in enumerate(np.linspace(0, 1, 2, endpoint=False)):
        ax = plt.Axes(fig, [w, 0.0, 1/2, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(images[i], cmap="gray")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def show_all_preds(pred_dir, data_dir, save=False, contour=True, mask_thresh=-500, alpha=0.3, filter=None):
    preds = glob(osp.join(pred_dir, "*"))
    for pred_file in preds:
        if filter and filter not in pred_file:
            continue
        basename = osp.basename(pred_file)
        print(basename)
        # read pred
        pred = plt.imread(pred_file)
        # read liver
        liver_file = osp.join(data_dir, "liver", basename.replace("_p_", "_o_").replace(".jpg", ".mhd"))
        _, liver = mhd_reader(liver_file)
        liver = ((np.clip(liver, -80, 170) + 80) / 250.0 * 255).astype(np.uint8)
        # read mask
        mask_file = osp.join(data_dir, "mask", basename.replace("_p_", "_m_").replace(".jpg", ".mhd"))
        _, mask = mhd_reader(mask_file)

        if save:
            save_path = liver_file.replace("liver", "segmentation").replace("_o_", "-").replace(".mhd", ".jpg")
        else:
            save_path = None
        show_a_pred(liver, mask, pred, save_path, contour=contour, mask_thresh=mask_thresh, alpha=alpha)

def show_liver_and_mask(data_dir, case, alpha=0.3):
    livers = glob(osp.join(data_dir, "liver", case + "*.mhd"))

    for liver_file in livers:
        print(liver_file)
        mask_file = liver_file.replace("liver", "mask").replace("_o_", "_m_")
        _, liver = mhd_reader(liver_file)
        _, mask = mhd_reader(mask_file)
        liver = ((np.clip(liver, -80, 170) + 80) / 250.0 * 255).astype(np.uint8)
        image = np.repeat(liver[:,:,np.newaxis], 3, axis=2)

        fig = plt.figure()
        fig.set_size_inches((6, 6))

        masked = np.where(mask > 0)
        image[masked[0], masked[1]] = (1 - alpha) * image[masked[0], masked[1]] + alpha * np.array([255, 255, 0])

        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(image, cmap="gray")
        plt.show()
        plt.close()

def parse_log_2D(filepath):
    pattern_Dice = re.compile("batch Dice: (\d\.\d{3})")
    pattern_VOE = re.compile("batch VOE:  (\d+\.\d{3})")
    patterm_VD = re.compile("batch VD:   (\d+\.\d{3})")
    
    with open(filepath) as f:
        obj = f.read()
    
    Dices =[float(v) for v in pattern_Dice.findall(obj)]
    VOEs = [float(v) for v in pattern_VOE.findall(obj)]
    VDs = [float(v) for v in patterm_VD.findall(obj)]

    print("Dice: {:.3f} {:.3f}".format(np.mean(Dices), np.std(Dices)))
    print("VOE: {:.3f} {:.3f}".format(np.mean(VOEs), np.std(VOEs)))
    print("VD: {:.3f} {:.3f}".format(np.mean(VDs), np.std(VDs)))

def parse_log_3D(filepath):
    pattern_Dice = re.compile("mean Dice: (\d+\.\d{3})")
    pattern_VOE  = re.compile("batch VOE:  (\d+\.\d{3})")
    patterm_VD   = re.compile("batch VD:   (\d+\.\d{3})")
    patterm_ASD  = re.compile("batch ASD:  (\d+\.\d{3})")
    patterm_RMSD = re.compile("batch RMSD: (\d+\.\d{3})")
    patterm_MSD  = re.compile("batch MSD:  (\d+\.\d{3})")
    
    with open(filepath) as f:
        obj = f.read()
    
    Dices =[float(v) for v in pattern_Dice.findall(obj)]
    Dices = Dices[:-1]
    VOEs = [float(v) for v in pattern_VOE.findall(obj)]
    VDs = [float(v) for v in patterm_VD.findall(obj)]
    ASDs = [float(v) for v in patterm_ASD.findall(obj)]
    RMSDs = [float(v) for v in patterm_RMSD.findall(obj)]
    MSDs = [float(v) for v in patterm_MSD.findall(obj)]
    
    print("Dice: {:.3f} {:.3f}".format(np.mean(Dices), np.std(Dices)))
    print("VOE:  {:.3f} {:.3f}".format(np.mean(VOEs), np.std(VOEs)))
    print("VD:   {:.3f} {:.3f}".format(np.mean(VDs), np.std(VDs)))
    print("ASD:  {:.3f} {:.3f}".format(np.mean(ASDs), np.std(ASDs)))
    print("RMSD: {:.3f} {:.3f}".format(np.mean(RMSDs), np.std(RMSDs)))
    print("MSD:  {:.3f} {:.3f}".format(np.mean(MSDs), np.std(MSDs)))



if __name__ == "__main__":
    if True:
        pred_dir = osp.join(osp.dirname(__file__), "..", "prediction", "temp")
        data_dir = "D:/DataSet/LiverQL/Liver_2016_train"
        show_all_preds(pred_dir, data_dir, save=False, contour=False, mask_thresh=0, alpha=0.4, filter="A")

    if False:
        filepath = "C:/documents/MLearning/MultiOrganDetection/core/Densenet_FCN/logs/20180429104523_test_skipv2_default_iter_200000"
        parse_log_2D(filepath)
        print()
        filepath = "C:/documents/MLearning/MultiOrganDetection/core/Densenet_FCN/logs/20180429104948_test_skipv2_default_iter_200000"
        parse_log_3D(filepath)
    
    if False:
        filepath = r"C:\documents\MLearning\MultiOrganDetection\core\Densenet_FCN\prediction\db_mv1\R071_p_10.jpg"
        image = plt.imread(filepath)
        image = image[:,:]
        plt.imshow(image)
        plt.show()

    if False:
        data_dir = "D:/DataSet/LiverQL/Liver_2017_train"
        case = "Q001"
        show_liver_and_mask(data_dir, case)
