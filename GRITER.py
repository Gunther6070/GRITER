import numpy as np
import pandas as pd

from glob import glob
from tqdm.notebook import tqdm
from PIL import Image
from sklearn.neighbors import KDTree

import cv2
import matplotlib.pyplot as plt
import IPython.display as IPD
import subprocess
import sys


vid_files = glob(
    'X:/GRIT/Olivine/600um-1000um/Videos/10. z24.1 z19.6/*.MOV')
img_files = glob(
    'X:/GRIT/Nepheline/600um-800um/Photos/0. z19.0  z61.3/*.NEF')

# ar ar ar

glint_means_prev = []
glint_means_prev = np.array(glint_means_prev)


def kmeans(image):

    # K_means variables
    Z = (np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    comp, lab, cent = cv2.kmeans(
        Z, 2, bestLabels=3,  criteria=criteria, attempts=1, flags=cv2.KMEANS_USE_INITIAL_LABELS)

    if len(np.where(lab == 1)[0]) > 0:

        # K_means display
        cent = np.uint8(cent)
        res = cent[lab.copy().flatten()]
        res2 = res.reshape(image.shape)
        lab_frame = np.reshape(lab, (image.shape[0], image.shape[1]))

        plt.imshow(lab_frame)
        plt.colorbar()
        plt.show()
        plt.close()

        # Glint quantification
        print("Bright pixels:", len(np.where(image > 235)[1]))
        print("Total pixels:", len(np.where(lab == 1)[0]))
        print(len(np.where(image > 235)[1]) / len(np.where(lab == 1)[0]), '\n')


class glintOne:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        count = 0

    def cCount(self):
        return self.count

    def increment(self):
        self.count += 1


def nearestN(image, ioc, cords=None):

    if ioc == 'image':
        # extract glint pixel coordinates
        thresholdb = 230
        glint_coords = np.array([
            np.where(image > thresholdb)[0], np.where(image > thresholdb)[1]]).T
    else:
        glint_coords = cords

    # K_Nearest_Neighbor
    kdt = KDTree(glint_coords, metric='euclidean')
    kdt_output = kdt.query(glint_coords, k=len(
        glint_coords)-1, return_distance=True)

    radius_threshold = 30
    glint_groups = []
    for glint_coord_ind in range(len(glint_coords)):
        glint_neighbors_ind = np.where(
            kdt_output[0][glint_coord_ind] <= radius_threshold)
        glint_group_inds = kdt_output[1][glint_coord_ind][glint_neighbors_ind]
        glint_groups.append(glint_coords[glint_group_inds])

    glint_group_means = []
    for glint in glint_groups:
        glinter = glintOne(np.mean(glint, axis=0))
        glint_group_means.append(glinter.coordinates)
    glint_group_means = np.array(glint_group_means)
    glint_group_means = np.unique(glint_group_means, axis=0)

    glint_means_prev = glint_group_means.copy()

    # plt.scatter(glint_group_means[:, 1],
    #             glint_group_means[:, 0],
    #             facecolor='none', edgecolor='tab:red',
    #             alpha=0.4, linewidth=0.2, s=10)

    # # Img display
    # plt.imshow(image, cmap='gray')
    # plt.savefig('image-test', dpi=400)
    # plt.show()
    # plt.close()

    return len(glint_group_means), glint_group_means, glint_means_prev


def nearestN2(prev_frame, cur_frame):  # glint_means_prev, nearestN()[1]
    vlist = np.vstack((prev_frame, cur_frame))
    # print(vlist)

    # K_Nearest_Neighbor
    kdt = KDTree(vlist, metric='euclidean')
    kdt_output = kdt.query(vlist, k=len(vlist)-1, return_distance=True)

    print(kdt_output[0][0], kdt_output[1][0:2])

    # radius_threshold = 30
    # glint_groups = []
    # for glint_coord_ind in range(len(vlist)):
    #     glint_neighbors_ind = np.where(
    #         kdt_output[0][glint_coord_ind] <= radius_threshold)
    #     glint_group_inds = kdt_output[1][glint_coord_ind][glint_neighbors_ind]
    #     glint_groups.append(vlist[glint_group_inds])


def VideoProc(path):

    vidObj = cv2.VideoCapture(path)

    count = 0
    success = 1
    az = []
    az = np.array(az)
    blist = []
    sec = []

    while success:

        success, image = vidObj.read()
        if not success:
            break

        if count % 1 == 0:
            sec.append(count/60)
            # print(sec)

        if count % 1 == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype(np.float32)

            # kmeans(image)

            # Grayscaling
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Glint recognition
            cords = nearestN(image, ioc='image')[1]
            blist.append(nearestN(image=image, cords=cords, ioc='cord')[0])

            nearestN2(nearestN(image, ioc='image')[
                      2], nearestN(image, ioc='image')[1])

            # hist display
            # plt.hist(np.ndarray.flatten(image), bins=75)
            # plt.xlim([0, 255])
            # plt.yscale('log')
            # plt.show()
            # plt.close()

        count += 1

    sec = np.array(sec)
    az = sec.copy()*12
    plt.plot(sec, blist)
    plt.ylabel("Number of glints")
    plt.xlabel("Time (Sec)")
    plt.show()
    plt.close()


def ImageProc(path):

    for img in path:
        image = plt.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32)

        kmeans(image)

        # Histogram display
        plt.hist(np.ndarray.flatten(image), bins=75)
        plt.xlim([0, 255])
        plt.yscale('log')
        plt.show()
        plt.close()

        # Img display
        plt.imshow(image, cmap='gray')
        plt.show()
        plt.close()


# for path in vid_files:
#     VideoProc(path)
VideoProc(vid_files[0])
# ImageProc(img_files)
