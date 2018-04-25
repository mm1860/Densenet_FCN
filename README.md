# Densenet_FCN
A tensorflow implementation of densenet with FCN for medical image segmentation

## Still Updating
|           | 2D Dice | VOE | VD | 3D Dice | VOE | VD | ASD | RMSD | MSD |
|:----------|:--------|:----|:---|:--------|:----|:---|:----|:-----|:----|
|First run  |0.853  |23.186  |13.140 |0.931  |12.906 |5.938  |2.987 |7.085  |65.420  |
|lr         |0.892  |17.645  |8.337  |0.949  |9.704  |3.091  |2.152 |6.100  |72.357  |
|xavier     |0.897  |17.052  |8.737  |0.948  |9.791  |2.944  |2.442 |7.345  |76.233  |
|unet       |0.908  |15.174  |9.950  |0.946  |10.266 |3.423  |2.636 |7.830  |85.754  |



## Some Experience
1. Large weight decay (for example 1.0) will impede network to learn image features. Actually a general value is 1e-5.
2. Dropout layer in decoder stage(upconv layers) maybe lead to white noise in prediction.
3. Large learning rate maybe lead to `Nan`
4. Without **batch normalization**, U-Net(I guess it is the same with other fcn-like ANNs) hardly converges(i.e. learn features) when using CT images.