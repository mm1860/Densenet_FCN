import matplotlib.pyplot as plt
from Liver_Kits import mhd_reader
from glob import glob
import os.path as osp
import cv2

def show_a_pred(origin, mask, pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    images = [origin, mask, pred]
    names = ["Origin", "Mask", "Prediction"]
    for i in range(3):
        ax[i].imshow(images[i], cmap="gray")
        ax[i].set_title(names[i])
        ax[i].axis("off")
    #mng = plt.get_current_fig_manager()
    #mng.window.showMaximized()
    plt.show()
    plt.close()

def show_all_preds(pred_dir, data_dir):
    preds = glob(osp.join(pred_dir, "*"))
    for pred_file in preds:
        basename = osp.basename(pred_file)
        # read pred
        pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        # read liver
        liver_file = osp.join(data_dir, "liver", basename.replace("_p_", "_o_").replace(".jpg", ".mhd"))
        _, liver = mhd_reader(liver_file)
        # read mask
        mask_file = osp.join(data_dir, "mask", basename.replace("_p_", "_m_").replace(".jpg", ".mhd"))
        _, mask = mhd_reader(mask_file)

        show_a_pred(liver, mask, pred)

if __name__ == "__main__":
    pred_dir = osp.join(osp.dirname(__file__), "..", "prediction", "default_bin")
    data_dir = "C:/DataSet/LiverQL/Liver_2017_test"
    show_all_preds(pred_dir, data_dir)
