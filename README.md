
# Reconstructions with Active Appearance model
Given images with landmarks (keypoints) all images are warped (patch size) into the mean keypoint shape
and a PCA is learned over centered (warped) images.

Reconstruction is then done by first centering, reconstruction with the transposed projection matrix of PCA
and then warping again to the original shape

# Run

Exctract the Obama dataset by

```
$ unzip data/100-shot-obama_128/img.zip -d data/100-shot-obama_128/ 
```

Create centered data, learn KPS and Image PCA models and reconstruct test data with 

```
$ python3 main.py
```


<img src="Readme_images/recon-1.png" width="400"> | <img src="Readme_images/recon-6.png" width="400">


# Credits
The Warping code is a simplified version of https://github.com/VipulRamtekkar/Active-Appearance-Models/tree/master/References
Whos work is very much appriciated

# Cites
```
@article{cootes2001active,
  title={Active appearance models},
  author={Cootes, Timothy F. and Edwards, Gareth J. and Taylor, Christopher J.},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  volume={23},
  number={6},
  pages={681--685},
  year={2001},
  publisher={IEEE}
}
```

