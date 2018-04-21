# Densenet_FCN
A tensorflow implementation of densenet with FCN for medical image segmentation

## Still Updating
* First run
  > 2D evaluation  
    >>> mean dice: 0.853  
    >>> mean_voe:  23.186  
    >>> mean_vd:   13.140  
  > 3D evaluation  
    >>> mean Dice: 0.931  
    >>> mean VOE:  12.906  
    >>> mean VD:   5.938  
    >>> mean ASD:  2.987  
    >>> mean RMSD: 7.085  
    >>> mean MSD:  65.420  
* lr(default_best)
  > 2D evaluation  
    >>> mean dice: 0.892  
    >>> mean_voe:  17.645  
    >>> mean_vd:   8.337  
  > 3D evaluation  
    >>> mean Dice: 0.949  
    >>> mean VOE:  9.704  
    >>> mean VD:   3.091  
    >>> mean ASD:  2.152  
    >>> mean RMSD: 6.100  
    >>> mean MSD:  72.357  
* xavier(default_best)
  > 2D evaluation  
    >>> mean dice: 0.897  
    >>> mean_voe:  17.052  
    >>> mean_vd:   8.737  
  > 3D evaluation  
    >>> mean Dice: 0.948  
    >>> mean VOE:  9.791  
    >>> mean VD:   2.944  
    >>> mean ASD:  2.442  
    >>> mean RMSD: 7.345  
    >>> mean MSD:  76.233  



* save 3D prediction is not finished.



## Some Experience
1. Large weight decay (for example 1.0) will impede network to learn image features.
2. Dropout layer in decoder stage(upconv layers) will lead to white noise in prediction.
3. Large learning rate will lead to `Nan`