import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
import logging
from PIL import Image
logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 0
img1_bgr = cv2.imread('data/books/book.jpg')
img2_bgr = cv2.imread('data/books/book_in_scene2.jpg')
img1_ycc = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2YCR_CB)
img2_ycc = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2YCR_CB)
img1_ycc = np.asarray(img1_ycc)
img2_ycc = np.asarray(img2_ycc)
k = 3
alpha = .5
yc_1 = k * np.sign(np.mean(img1_ycc[:,:,1]) - np.mean(img1_ycc[:,:,2])) * np.sign(img1_ycc[:,:,1] - img1_ycc[:,:,2]) * np.abs(img1_ycc[:,:,1] - img1_ycc[:,:,2]) ** alpha
yc_2 = k * np.sign(np.mean(img2_ycc[:,:,1]) - np.mean(img2_ycc[:,:,2])) * np.sign(img2_ycc[:,:,1] - img2_ycc[:,:,2]) * np.abs(img2_ycc[:,:,1] - img2_ycc[:,:,2]) ** alpha
p_1 = img1_ycc[:,:,0] + yc_1
p_2 = img2_ycc[:,:,0] + yc_2
ye_1 = (128-np.mean(p_1))*np.exp((-((np.linalg.norm(p_1)) - .5) ** 2)/(2*.2**2))
ye_2 = (128-np.mean(p_2))*np.exp((-((np.linalg.norm(p_2)) - .5) ** 2)/(2*.2**2))
img1 = img1_ycc[:,:,0] + yc_1 + ye_1
img2 = img2_ycc[:,:,0] + yc_2 + ye_2
# img1 = Image.fromarray((img1).astype(np.uint8))
# img2 = Image.fromarray((img2).astype(np.uint8))
# img1.save('testbooks1.png')
# img2.save('testbooks2.png')
# img1 = cv2.imread('data/books/book.jpg', 0)           # queryImage
# img2 = cv2.imread('data/books/book_in_scene2.jpg', 0)  # trainImage
# img1 = Image.fromarray((img1).astype(np.uint8))
# img1.save('testbooks1org.png')

if img1.shape[0] >= img2.shape[0] or img1.shape[1] >= img2.shape[1]:
    img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)

# Compute SIFT keypoints and descriptors
print('Start: Compute SIFT keypoints and descriptors')
kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)
print('Done: Compute SIFT keypoints and descriptors')

# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
print(len(good))
if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 0, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    print(hdif)
    print(img1.shape)
    print(img2.shape)
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.figure()
    plt.imshow(newimg)
    plt.savefig('thing.jpg') 
    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
