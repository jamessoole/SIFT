# SIFT
SIFT exploration for [use as a neural net layer](https://github.com/jamessoole/SIFT_Neural_Net) in place of pooling layers.

SIFT Paper: ["Distinctive Image Features from Scale-Invariant Keypoints", David G. Lowe](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

### Based on [PythonSIFT](https://github.com/rmislam/PythonSIFT)
> This is an implementation of SIFT (David G. Lowe's scale-invariant feature transform) done entirely in Python with the help of NumPy. This implementation is based on OpenCV's implementation and returns OpenCV `KeyPoint` objects and descriptors, and so can be used as a drop-in replacement for OpenCV SIFT. This repository is intended to help computer vision enthusiasts learn about the details behind SIFT.

#### Dependencies
`Python 3`,
`NumPy`,
`OpenCV-Python`.
Last tested successfully using `Python 3.8.5`, `Numpy 1.19.4` and `OpenCV-Python 4.3.0`.

#### Usage
```python
import cv2
import pysift
image = cv2.imread('your_image.png', 0)
keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
```

