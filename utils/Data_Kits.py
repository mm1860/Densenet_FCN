import matplotlib.pyplot as plt
from Liver_Kits import mhd_reader
from glob import glob
import os.path as osp
import cv2
import re
import numpy as np

def show_a_pred(image, mask, pred, save_path=None, alpha=0.3):
    fig = plt.figure()
    fig.set_size_inches((12, 6))

    image_mask = np.repeat(image[:,:,np.newaxis], 3, axis=2)
    preded = np.where(pred > 125)
    image_mask[preded[0], preded[1]] = (1 - alpha) * image_mask[preded[0], preded[1]] + alpha * np.array([0, 255, 0])
    masked = np.where(mask > 0)
    image_mask[masked[0], masked[1]] = (1 - alpha * 1.5) * image_mask[masked[0], masked[1]] + alpha * 1.5 * np.array([255, 0, 0])
    images = [image, image_mask]
    
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

def show_all_preds(pred_dir, data_dir):
    preds = glob(osp.join(pred_dir, "*"))
    for pred_file in preds:
        if "R" in pred_file:
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

        save_path = liver_file.replace("liver", "segmentation").replace("_o_", "-").replace(".mhd", ".jpg")
        show_a_pred(liver, mask, pred, save_path)

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
    if False:
        pred_dir = osp.join(osp.dirname(__file__), "..", "prediction", "db_mv1")
        data_dir = "C:/DataSet/LiverQL/Liver_2017_test"
        show_all_preds(pred_dir, data_dir)

    if True:
        filepath = "C:/documents/MLearning/MultiOrganDetection/core/Densenet_FCN/logs/20180425221600_test_unet_default_iter_250000"
        parse_log_2D(filepath)
        print()
        filepath = "C:/documents/MLearning/MultiOrganDetection/core/Densenet_FCN/logs/20180425222611_test_unet_default_iter_250000"
        parse_log_3D(filepath)
    
    if False:
        filepath = r"C:\documents\MLearning\MultiOrganDetection\core\Densenet_FCN\prediction\db_mv1\R071_p_10.jpg"
        image = plt.imread(filepath)
        image = image[:,:]
        plt.imshow(image)
        plt.show()
