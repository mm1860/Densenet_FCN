import tensorflow as tf
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import matplotlib.pyplot as plt

def summary_scalar(writer, iter, tags, values):
    """ Summary a scalar in tensorboard manually.

    Params
    ------
    `writer`: a tf.summary.FileWriter instance  
    `iter`: a integer to denote current iteration  
    `tags`: tag of the scalar, multi-level tag should be seperated by `/`.
    You can pass a single tag or a list of tags.  
    `values`: scalar value to be summaried. 
    
    Note: `tags` and `values` should have same length(i.e. both single entry 
    or a list of entries)  
    """

    if not isinstance(tags, (str, list, tuple)):
        raise TypeError("tags should have type of (str, list, tuple), but got {:r}".format(type(tags)))
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(values, (float, int, list, tuple)):
        raise TypeError("values should have type of (float, int, list, tuple), but got {:r}".format(type(values)))
    if isinstance(values, (float, int)):
        values = [values]
    
    all_value = []
    for tag, value in zip(tags, values):
        all_value.append(tf.Summary.Value(tag=tag, simple_value=value))
    
    summary_value = tf.Summary(value=all_value)
    writer.add_summary(summary_value, int(iter))
    
    return

def summary_image(writer, iter, tag, images, max_outputs=3):
    """ Summary a batch images in tensorboard manually.

    Params
    ------
    `writer`: a tf.summary.FileWriter instance  
    `iter`: a integer to denote current iteration  
    `tag`: tag of the image, details please reference `tf.summary.image`  
    `images`: 4D np.ndarray with shape [batch_size, height, width, channels]  
    `max_outputs`: Max number of batch elements to generate images for.  
    """

    all_value = []
    for i, image in enumerate(images):
        buffer = StringIO()
        plt.imsave(buffer, image, format="png")
        image_sum_obj = tf.Summary.Image(height=image[0], width=image[1], 
                                         encoded_image_string=buffer)
        all_value.append(tf.Summary.Value(tag="{:s}/image/{:d}".format(tag, i), image=image_sum_obj))
        if i + 1 >= max_outputs:
            break

    summary_value = tf.Summary(value=all_value)
    writer.add_summary(summary_value, int(iter))
    
    return