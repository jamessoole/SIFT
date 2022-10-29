=============================================
CMU Kitchen Occlusion Dataset (CMU_KO8)
=============================================
Edward Hsiao
ehsiao@cs.cmu.edu

The dataset is split up into five folders:

calibration:
    - camera calibration images

single:
    train:  
        - contains 8 objects each with an image (*.jpg) and mask (*_mask.png)
        - heights.txt - height in cm used for occlusion reasoning under this viewpoint
    test:
        - contains 800 total images, 100 for each object
        - image is *.jpg
        - bbox is *.bbox in format [x,y,w,h]
        - occlusion pattern is *.occ.png with the occluder regions specified
        - occlusion_statistics.txt - specifies percentage of object that has been occluded.  
              These values were computed by placing the object mask centered on the bounding
              box and computing the portion that overlaps with the occlusion pattern.

multiple:
    train:
        - contains 8 objects with 25 viewpoints in each directory
        - each image (*.jpg) has an object mask (*_mask.png)
        - calibration checkerboard has size 2.2 cm
        - *.info file contains the position and size of the object
            [x,y,z] position in cm relative to the top left corner of the checkerboard pattern
            [w,l,h] object dimensions in cm        
    test:
        - contains 800 total images, 100 for each object
        - image is *.jpg
        - bbox is *.bbox in format [x,y,w,h]    

negative:
    - contains 100 random kitchen scenes taken from the Internet

plots:
    - the values used to plot the FPPI/DR curves for Figure 10 and Figure 11 of the CVPR 2012 paper.
    - each txt file corresponds to a plot and is provided in the format [fppi, dr(LINE2D), dr(rLINE2D), dr(rLINE2D+OPP), dr(rLINE2D+OCLP)]


Please cite the following paper if you use this dataset:
E. Hsiao and M. Hebert. 'Occlusion reasoning for object detection under arbitrary viewpoint.' In CVPR 2012.

