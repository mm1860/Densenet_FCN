# Densenet_FCN
A tensorflow implementation of densenet with FCN for medical image segmentation

## Still Updating
|           | 2D Dice   | VOE       | VD        | 3D Dice   | VOE       | VD        | ASD       | RMSD      | MSD      |
|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:---------|
|First run  |0.853      |23.186     |13.140     |0.931      |12.906     |5.938      |2.987      |7.085      |65.420    |
|lr         |0.892      |17.645     |**8.337** |**0.949**  |9.704      |3.091      |**2.152**  |6.100       |72.357    |
|xavier     |0.897      |17.052     |8.737      |0.948      |9.791      |**2.944** |2.442       |7.345      |76.233    |
|skip       |0.894      |17.274     |10.228     |0.946      |10.246     |3.688      |2.979      |8.186      |88,438    |
|skipv2     |0.900      |16.644     |10.320     |0.943      |10.755     |3.274      |3.190      |8.294      |77.424    |
|theta-bs2  |0.887      |18.565     |9.655      |0.944      |10.562     |3.840      |2.213      |**5.498**  |**65.194**|
|theta-bs4  |0.900      |16.594     |9.565      |**0.949**  |**9.605** |3.070       |2.427      |7.196      |80.521     |
|layer_norm |0.900      |16.501     |9.042      |0.947      |10.031     |3.118      |2.547      |7.368      |78.285     |
|skip-theta |0.894      |17.295     |9.728      |0.944      |10.623     |3.235      |3.130      |8.241      |69.645     |
|skipv2-theta|0.897     |16.889     |9.340      |0.947      |10.093     |3.230      |2.625      |7.492      |77.897     |
|unet       |**0.908**  |**15.174**|9.950      |0.946      |10.266     |3.423      |2.636       |7.830      |85.754    |
|udnet      |0.878      |19.215     |13.365     |0.927      |13.070     |5.354      |3.906       |9.389     |73.603    |

## Some Experience
1. Large weight decay (for example 1.0) will impede network to learn image features. Actually a general value is 1e-5.
2. Dropout layer in decoder stage(upconv layers) maybe lead to white noise in prediction.
3. Large learning rate maybe lead to `Nan`
4. Without **batch normalization**, U-Net(I guess it is the same with other fcn-like ANNs) hardly converges(i.e. learn features) when using CT images.