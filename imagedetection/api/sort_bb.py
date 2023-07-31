import math
import cv2
from statistics import mean
import numpy as np


class SortBBAlgorithm:
    def __init__(self):
        self.bb_sorted = []




    def average_height(self, detections_sortedX):
        height_list = []
        for d in detections_sortedX:
            _, ymin, _, ymax = d
            h = int(ymax-ymin)
            height_list.append(h)

        return mean(height_list)


    def reject_outliers(self, data, m = 2.):

        mean, stdev = np.mean(data, axis=0), np.std(data, axis=0)
        print("mean, stdev: ", mean, stdev)
        # mdev = np.median(d)
        # s = d/mdev if mdev else np.zero(len(d))\
        print(np.abs(data[:,0]) - mean[0] > stdev[0])
        print(np.abs(data[:,1]) - mean[1] > stdev[1])
        print(np.abs(data[:,2]) - mean[2] > stdev[2])
        print(np.abs(data[:,3]) - mean[3] > stdev[3])

        ## Find Outliers
        outliers = ((np.abs(data[:,0] - mean[0]) < stdev[0])
                    * (np.abs(data[:,1] - mean[1]) < stdev[1])
                    * (np.abs(data[:,2] - mean[2]) < stdev[2])
                    * (np.abs(data[:,3] - mean[3]) < stdev[3]))



        ## Result
        print("outliers: ", outliers)
        print(data[outliers])



        return data

    def sortBB(self, image, bounding_boxes_list):

        bb_1=[]
        bb_2=[]
        bb_digits = []
        bb_alphabets = []
        cy_list = []
        # sort bounding boxes in the x direction first
        detections_sortedX=sorted(bounding_boxes_list, key=lambda x: x[0])
        # calculate the average height of the bbs
        average_height = self.average_height(detections_sortedX)

        try:
            for i in range(len(detections_sortedX)):
                
                xmin, ymin, xmax, ymax = np.array(bounding_boxes_list[i])
                x = int(xmin)
                y = int(ymin)
                w = int(xmax-xmin)
                h = int(ymax-ymin)
                # (x, y, w, h) = detections_sortedX[i]
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(image, l, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                # cv2.circle(image, (cx,cy), 2, (255, 255, 0), 2)

                cy_list.append(int(cy))
                # Now check if the the next bb is in the same line as the first
                # bb.
                if abs(int(cy) - cy_list[0]) < average_height:
                    bb_1.append([x, y, w, h])
                else:
                    bb_2.append([x, y, w, h])
                    # else if the difference is greater than the average 
                    # height, append to new list


            if bb_1[0][1] < bb_2[0][1]:
                bb_digits = bb_1
                bb_alphabets = bb_2
            else:
                bb_digits = bb_2
                bb_alphabets = bb_1

            # print("cy_list: ", cy_list)
            # print("bb_digits: ", bb_digits)
            # bb_digits_ = self.reject_outliers(np.array(bb_digits))
            # print("bb_alphabets: ", bb_alphabets)
            # # bb_alphabets = self.reject_outliers(bb_alphabets)
            # print("bb_alphabets: ", bb_alphabets)
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            shell_no="---"
            print("error detecting digits: ", e)


        return bb_digits, bb_alphabets


# sort = SortBBAlgorithm()
# bounding_boxes_list=[
#     (614.0, 127.0, 675.0, 211.0), 
#     (410.0, 318.0, 464.0, 404.0), 
#     (476.0, 320.0, 536.0, 401.0), 
#     (824.0, 311.0, 893.0, 405.0), 
#     (687.0, 117.0, 744.0, 211.0), 
#     (687.0, 321.0, 746.0, 397.0), 
#     (561.0, 117.0, 590.0, 208.0), 
#     (551.0, 320.0, 592.0, 401.0), 
#     (763.0, 323.0, 808.0, 399.0)
#     ]
# img = cv2.imread('2008.jpg')
# # # s.sortBB(img, detections)
# (bb_digits, bb_alphabets) = sort.sortBB(img, bounding_boxes_list)
# # xmin, ymin, xmax, ymax = bb_digits
