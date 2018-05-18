import tensorflow as tf
import argparse
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import cfg
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze a tensorflow model")
    parser.add_argument("--model", dest="model_tag", default=None, type=str,
                        required=True,
                        help="model tag")
    parser.add_argument("--prefix", dest="model_prefix", default=None, type=str,
                        help="model prefix")
    parser.add_argument("--best", dest="best", default=None, type=str,
                        help="best model or not")
    parser.add_argument("--iters", dest="iters", default=None, type=str,
                        help="model iters")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def freeze_model(model_tag, model_prefix="default", **kwargs):
    """ Freeze model into single *.pb file.

    Params
    ------
    model_tag: model tag `xxx`
    model_prefix: model prefix, which is used in model name
    best: boolean, if set, best model will be frozen. This parameter has priority and is default.
    iters: integer, if not None, the specified model will be frozen.
    """
    output_nodes = [
        "FCN-DenseNet/BinaryPred/Pred2Binary",
        "FCN-DenseNet/Prediction/split"
    ]

    if kwargs.get("best"):
        model_name = "{}_best".format(model_prefix)
    elif kwargs.get("iter"):
        model_name = "{}_iter_{}".format(model_prefix, dwargs.get("iters"))
    else:
        model_name = "{}_best".format(model_prefix)

    model_dir = pathlib.Path(cfg.SRC_DIR) / cfg.OUTPUT_DIR / model_tag
    model_path = model_dir / (model_name + ".ckpt.meta")
    pb_path = model_dir / (model_name + ".pb")

    if not model_path.exists():
        raise FileNotFoundError("Cannot find model file: {}".format(ckpt))

    print("Freeze model begin!")
    # 1. load model
    saver = tf.train.import_meta_graph(str(model_path), clear_devices=True)

    with tf.Session(graph=tf.get_default_graph()) as sess:
        # serialize model
        input_graph_def = sess.graph.as_graph_def()
        # 2. load weights
        saver.restore(sess, str(model_path)[:-5])
        # 3. weights --> constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_nodes)
        # 4. write to file
        with pb_path.open("wb") as f:
            f.write(output_graph_def.SerializeToString())
    print("Freeze model finished!")

def segmentation(model_path, images):
    """ Segmentation liver with FCN-DenseNet

    """
    graph_def = tf.GraphDef()
    with model.open("rb") as f:
        graph_def.ParseFromString(f.read())
    
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    image_placeholder = graph.get_tensor_by_name("Image:0")
    prediction_tensor = graph.get_tensor_by_name("FCN-DenseNet/BinaryPred/Pred2Binary:0")

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig, graph=graph) as sess:
        preds = []
        for image in images:
            img = image[:, :, np.newaxis]
            img = np.stack([img])
            pred = sess.run(prediction_tensor, feed_dict={image_placeholder: img})
            preds.append(pred[0,...,0])
    
    return preds

if __name__ == "__main__":
    if False:
        kwargs = vars(parse_args())
        freeze_model(**kwargs)

    if True:
        import matplotlib.pyplot as plt
        from Liver_Kits import mhd_reader
        slice = "D:/DataSet/LiverQL/Liver_2018_test/liver/R071_o_14.mhd"
        _, ct = mhd_reader(slice)
        ct = (np.clip(ct, 55 - 125, 55 + 125) - (55 - 125)) / 2**16 * 250
        model = pathlib.Path(cfg.SRC_DIR) / cfg.OUTPUT_DIR / "skipv2_dice" / "FCN-DenseNet.pb"
        preds = segmentation(model, [ct])
        plt.imshow(preds[0], cmap="gray")
        plt.show()
