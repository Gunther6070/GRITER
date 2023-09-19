import numpy as np

from glob import glob
from sklearn.neighbors import KDTree

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors
import sys


vid_files = glob(
    'X:/GRIT Sand/Olivine/600um-1000um/Videos/60. z37.0 z33.7/*.MOV')
img_files = glob(
    'X:/GRIT/Nepheline/600um-800um/Photos/0. z19.0  z61.3/*.NEF')

brtthreshold = 75
counter = 60


def kmeans(image):

    # # K_means variables
    Z = (np.reshape(image, (image.shape[0]*image.shape[1], image.shape[2])))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 2.0)
    comp, lab, cent = cv2.kmeans(
        Z, 2, bestLabels=2, criteria=criteria, attempts=30, flags=cv2.KMEANS_USE_INITIAL_LABELS)

    if len(np.where(lab == 1)[0]) > 0:

        # # K_means display
        cent = np.uint8(cent)
        res = cent[lab.copy().flatten()]
        res2 = res.reshape(image.shape)
        lab_frame = np.reshape(lab, (image.shape[0], image.shape[1]))

        cmap = matplotlib.colors.ListedColormap(['white', 'red'])

        # plt.imshow(lab_frame, cmap=cmap)
        # plt.colorbar()
        # plt.savefig('kmean', dpi=200)
        # plt.show()
        # plt.close()

        # # Glint quantification
        # print("Bright pixels:", len(np.where(image > brtthreshold)[1]))
        # print("Sand pixels:", len(np.where(lab == lab_frame[500][900])[0]))
        # print(len(np.where(image > brtthreshold)[
        #       1]) / len(np.where(lab == lab_frame[500][900])[0])*100, '%\n')

    return np.where(lab_frame == lab_frame[500][900])[0], np.where(lab_frame == lab_frame[500][900])[1], len(np.where(image > brtthreshold)[
        1]) / len(np.where(lab == lab_frame[500][900])[0])*100


def nearestN(image, ioc, cords=None):

    if ioc == 'image':
        # # extract glint pixel coordinates
        glint_coords = np.array([
            np.where(image > brtthreshold)[0], np.where(image > brtthreshold)[1]]).T
        radius_threshold = 5
    else:
        glint_coords = cords
        radius_threshold = 15

    # # K_Nearest_Neighbor
    kdt = KDTree(glint_coords, metric='euclidean')
    kdt_output = kdt.query(glint_coords, k=len(
        glint_coords)-1, return_distance=True)

    glint_groups = []
    for glint_coord_ind in range(len(glint_coords)):
        glint_neighbors_ind = np.where(
            kdt_output[0][glint_coord_ind] <= radius_threshold)
        glint_group_inds = kdt_output[1][glint_coord_ind][glint_neighbors_ind]
        glint_groups.append(glint_coords[glint_group_inds])

    glint_group_means = []
    for glint in glint_groups:
        glint_group_means.append(np.mean(glint, axis=0))
    glint_group_means = np.array(glint_group_means)
    glint_group_means = np.unique(glint_group_means, axis=0)

    # plt.scatter(glint_group_means[:, 1],
    #             glint_group_means[:, 0],
    #             facecolor='none', edgecolor='tab:red',
    #             alpha=0.8, linewidth=0.2, s=3)

    # # Img display
    # plt.imshow(image, cmap='gray')
    # plt.savefig('image-test', dpi=200)
    # plt.show()
    # plt.close()

    return len(glint_group_means), glint_group_means


def GOT(cordList, countList):

    cordsList = []
    countsList = []

    # # flattens both arrays
    for celement in cordList:
        for cord in celement:
            cordsList.append(cord)

    for element in countList:
        for count in element:
            countsList.append(count)

    # # K_Nearest_Neighbor
    kdt = KDTree(cordsList, metric='euclidean')
    kdt_output = kdt.query(cordsList, k=len(cordsList)-1, return_distance=True)

    radius_threshold = 30
    glint_groups = []
    glint_frames = []
    for glint_coord_ind in range(len(cordsList)):
        glint_neighbors_ind = np.where(
            kdt_output[0][glint_coord_ind] <= radius_threshold)
        glint_group_inds = kdt_output[1][glint_coord_ind][glint_neighbors_ind]
        for inds in glint_group_inds:
            glint_frames.append(countsList[inds])
        glint_groups.append(glint_group_inds)

    # returns master list of indices, array of array of indices
    return glint_groups


def VideoProc(path):

    vidObj = cv2.VideoCapture(path)

    count = 0
    success = 1
    az = []
    az = np.array(az)
    sec = []
    cordlist = []
    numcordlist = []
    countlist = []
    rreflectance = []
    rreflectance = np.array(rreflectance)
    greflectance = []
    greflectance = np.array(greflectance)
    breflectance = []
    breflectance = np.array(breflectance)
    glintpers = []

    while success:

        success, image = vidObj.read()
        if not success:
            break

        if count % counter == 0:
            sec.append(count/60)

        if count % counter == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # plt.imshow(image)
            # plt.show()
            # plt.close()

            image = image.astype(np.float32)

            # rpuck, gpuck, bpuck = Spectralon(vid_files[len(vid_files)-1])

            # ypixels, xpixels, glintper = kmeans(image)
            # glintpers.append(glintper)

            # rpuck = rpuck.astype(np.float32)
            # gpuck = gpuck.astype(np.float32)
            # bpuck = bpuck.astype(np.float32)

            # print(image[ypixels, xpixels, 0], image[ypixels, xpixels, 0].shape)
            # rreflectance = np.append(
            #     rreflectance, image[ypixels, xpixels, 0]/rpuck*.99)
            # greflectance = np.append(
            #     greflectance, image[ypixels, xpixels, 1]/gpuck*.99)
            # breflectance = np.append(
            #     breflectance, image[ypixels, xpixels, 2]/bpuck*.99)

            # # Grayscaling
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # # # Glint recognition
            cords = nearestN(image, ioc='image')[1]
            numcordlist.append(
                nearestN(image=image, cords=cords, ioc='cord')[0])

            # cordlist.append(cords.copy())
            # countlist.append(cords.copy())

            # Countlist declaration
            # countlist[len(countlist)-1].fill(count)
            # countlist[len(countlist)-1] = np.mean(countlist[len(countlist)-1],
            # axis=1)

            # hist display
            # plt.hist(np.ndarray.flatten(rreflectance),
            #          color="tab:red", bins=45)
            # plt.xlim([0, 2.5])
            # plt.ylim([0, 2000])
            # plt.savefig('red', dpi=400)
            # plt.show()
            # plt.close()

            # plt.hist(np.ndarray.flatten(greflectance),
            #          color="tab:green", bins=45)
            # plt.xlim([0, 2.5])
            # plt.ylim([0, 2000])
            # plt.savefig('green', dpi=400)
            # plt.show()
            # plt.close()

            # plt.hist(np.ndarray.flatten(breflectance),
            #          color="tab:blue", bins=45)
            # plt.xlim([0, 2.5])
            # plt.ylim([0, 2000])
            # plt.savefig('blue', dpi=400)
            # plt.show()
            # plt.close()

        count += 1

    print(np.mean(numcordlist))
    # sec = np.array(sec)
    # az = sec.copy()*12
    # plt.plot(az, glintpers)
    # plt.ylabel("Glint percent")
    # plt.xlabel("Az (Degrees)")
    # plt.show()
    # plt.close()


def Spectralon(path):

    vidObj = cv2.VideoCapture(path)

    count = 0
    success = 1
    az = []
    az = np.array(az)
    sec = []
    rframes = []
    gframes = []
    bframes = []

    while success:

        success, image = vidObj.read()
        if not success:
            break

        if count % 1 == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rframes.append(np.array(image[260:450, 1150:1390, 0]))
            gframes.append(np.array(image[260:450, 1150:1390, 1]))
            bframes.append(np.array(image[260:450, 1150:1390, 2]))

    for element in range(len(rframes)):
        rframes[element] = np.mean(rframes[element])
        gframes[element] = np.mean(gframes[element])
        bframes[element] = np.mean(bframes[element])

    rframes = np.mean(rframes)
    gframes = np.mean(gframes)
    bframes = np.mean(bframes)

    return rframes, gframes, bframes


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
