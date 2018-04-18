# Densenet_FCN
A tensorflow implementation of densenet with FCN for medical image segmentation

## Still Updating
* first run
  > 2D evalidation:  
  > \>>> mean Dice:  0.901  
  > \>>> mean VOE:  15.531  
  > \>>> mean VD:    9.940  
  > 3D evalidation:  
  > \>>> mean Dice:  0.939  
  > \>>> mean VOE:  11.435  
  > \>>> mean VD:    4.351  
  > \>>> mean ASD:   2.989  
  > \>>> mean RMSD:  7.003  
  > \>>> mean MSD:  70.183  
* leaky_relu
  > 2D evalidation:  
  > \>>> mean Dice:  0.905  
  > \>>> mean VOE:  15.146  
  > \>>> mean VD:   11.162  
  > 3D evalidation:  
  > \>>> mean Dice:  0.935  
  > \>>> mean VOE:  12.131  
  > \>>> mean VD:    5.211  
  > \>>> mean ASD:   3.863  
  > \>>> mean RMSD:  9.425  
  > \>>> mean MSD:  97.093  

* save 3D prediction is not finished.


## Some Experience
1. Large weight decay (for example 1.0) will impede network to learn image features.
2. Dropout layer in decoder stage(upconv layers) will lead to white noise in prediction.
3. Large learning rate will lead to 