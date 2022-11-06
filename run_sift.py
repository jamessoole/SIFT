import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
import logging
from PIL import Image
import time


logger = logging.getLogger(__name__)

start_time = time.time()


MIN_MATCH_COUNT = 0


# filename1 = 'data/books/book.jpg'
# filename1 = 'data/books/book2.jpg'
filename1 = 'data/books/book3.jpg'


# filename2 = 'data/books/book_in_scene3.jpg'
# filename2 = 'data/books/book_in_scene4.jpg'
filename2 = 'data/books/book_in_scene6.jpg'

# filename1 = 'data/select/scissors_cropped.jpg'
# filename2 = 'data/select/scissors_098.jpg'


img1_bgr = cv2.imread(filename1)
img2_bgr = cv2.imread(filename2)
print('img1_bgr.shape',img1_bgr.shape)
print('img2_bgr.shape',img2_bgr.shape)




# ----------------------------------------------
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7514695


img1_ycc = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2YCR_CB)
img2_ycc = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2YCR_CB)
print('img1_ycc.shape',img1_ycc.shape)
img1_ycc = np.asarray(img1_ycc)
img2_ycc = np.asarray(img2_ycc)
k = 3
alpha = 0.5
# yc_1 = k * np.sign(np.mean(img1_ycc[:,:,1]) - np.mean(img1_ycc[:,:,2])) * np.sign(img1_ycc[:,:,1] - img1_ycc[:,:,2]) * np.abs(img1_ycc[:,:,1] - img1_ycc[:,:,2]) ** alpha
# yc_2 = k * np.sign(np.mean(img2_ycc[:,:,1]) - np.mean(img2_ycc[:,:,2])) * np.sign(img2_ycc[:,:,1] - img2_ycc[:,:,2]) * np.abs(img2_ycc[:,:,1] - img2_ycc[:,:,2]) ** alpha
yc_1 = k * np.sign(np.mean(img1_bgr[:,:,2]) - np.mean(img1_bgr[:,:,0])) * np.sign(img1_ycc[:,:,1] - img1_ycc[:,:,2]) * np.abs(img1_ycc[:,:,1] - img1_ycc[:,:,2]) ** alpha
yc_2 = k * np.sign(np.mean(img2_bgr[:,:,2]) - np.mean(img2_bgr[:,:,0])) * np.sign(img2_ycc[:,:,1] - img2_ycc[:,:,2]) * np.abs(img2_ycc[:,:,1] - img2_ycc[:,:,2]) ** alpha
p_1 = img1_ycc[:,:,0] + yc_1
p_2 = img2_ycc[:,:,0] + yc_2
sig = 0.2
ye_1 = (128-np.mean(p_1))*np.exp((-(p_1/255 - 0.5) ** 2)/(2*sig**2))
ye_2 = (128-np.mean(p_2))*np.exp((-(p_2/255 - 0.5) ** 2)/(2*sig**2))
img1 = img1_ycc[:,:,0] + yc_1 + ye_1
img2 = img2_ycc[:,:,0] + yc_2 + ye_2


# print('img1.shape',img1.shape)
# print('img2.shape',img2.shape)
if img1.shape[0] >= img2.shape[0] or img1.shape[1] >= img2.shape[1]:
    # resize to make item image smaller for displaying
    print('Adjsuted Image Size!!')
    img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
# print('img1.shape',img1.shape)
# print('img2.shape',img2.shape)


# img1_im = Image.fromarray((img1).astype(np.uint8))
# img2_im = Image.fromarray((img2).astype(np.uint8))
# img1_im.save('edited_image1.png')
# img2_im.save('edited_image2.png')







# ------------------------------------------
# seperate color channels

# img1 = img1_bgr[:,:,0] # b
# img2 = img2_bgr[:,:,0]

# img1 = img1_bgr[:,:,1] # g
# img2 = img2_bgr[:,:,1]

img1 = img1_bgr[:,:,2] # r
img2 = img2_bgr[:,:,2]



# ----------------------------
# # baseline


img1 = cv2.imread(filename1, 0) # queryImage
img2 = cv2.imread(filename2, 0) # trainImage
if img1.shape[0] >= img2.shape[0] or img1.shape[1] >= img2.shape[1]:
    # resize to make item image smaller for displaying
    print('Adjsuted Image Size!!')
    img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)


# -----------------------------------------------------------------

# convert to 8-bit for cv2.detectAndCompute
img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')


# -----------------------------------------------------------------


# Compute SIFT keypoints and descriptors
print('Start: Compute SIFT keypoints and descriptors')
# # # slower, but fine-tunable impelmentation
# kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
# kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

# # faster implementation
sift = cv2.SIFT_create() 
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


# # Orb, might need to ignore the Lowe's ratio test or lower threshold
# orb = cv2.ORB_create()
# kp1 = orb.detect(img1,None)
# kp2 = orb.detect(img2,None)
# kp1, des1 = orb.compute(img1, kp1)
# kp2, des2 = orb.compute(img2, kp2)
# des1 = np.float32(des1)
# des2 = np.float32(des2)


print('Done: Compute SIFT keypoints and descriptors')



# -----------------------------------------------------------------


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
    # good.append(m)
    
print('Num Good Matches:',len(good))




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
    # print(hdif)
    # print(img1.shape)
    # print(img2.shape)
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    end_time = time.time()
    tot_time = end_time - start_time
    print('Time:',tot_time)

    plt.figure()
    plt.imshow(newimg)

    # plt.savefig('res_scene3_baseline.jpg',dpi=2000) 
    # plt.savefig('res_scene4_baseline.jpg',dpi=2000)
    # plt.savefig('res_scene3_color.jpg',dpi=2000) 
    # plt.savefig('res_scene4_color.jpg',dpi=2000) 

    # plt.savefig('res_book2_baseline.jpg',dpi=2000) 
    # plt.savefig('res_book2_color.jpg',dpi=2000) 

    # plt.savefig('res_blue.jpg',dpi=2000) 
    # plt.savefig('res_green.jpg',dpi=2000) 
    # plt.savefig('res_red.jpg',dpi=2000) 





    plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))




