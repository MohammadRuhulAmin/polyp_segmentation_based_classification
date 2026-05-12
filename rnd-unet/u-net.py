import cv2
import numpy as np
import os

height = 256
width = 256

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--preprocess",
    action="store_true"
)
args = parser.parse_args()


allImages = []
maskImages = []

path = "/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/Kvasir-SEG/"
imagePath = path + "/images"
maskPath = path + "/masks"


img = cv2.imread(imagePath+"/cju0qkwl35piu0993l0dewei2.jpg", cv2.IMREAD_COLOR)
img = cv2.resize(img,(width, height))

mask = cv2.imread(maskPath+"/cju0qkwl35piu0993l0dewei2.jpg", cv2.IMREAD_COLOR)
mask = cv2.resize(mask,(width, height))
# cv2.imshow("image", img)
# cv2.imshow("mask", mask)
# cv2.waitKey(0)


# lets look at the values of the mask
# resize temporary
resizeto16 = cv2.resize(mask,(16,16))
print(resizeto16)

# We can see the mask aria is higher than 0 and the background is 0 (Black)
# Let's change the value to binary (all the black would be 0, all the other will be white-1)

# Create the Numpy Arrays:
resizeto16[resizeto16 <=50] = 0
resizeto16[resizeto16 >50] = 1
print(resizeto16)


images = os.listdir(imagePath)
print("image length: ",len(images))
print(args.preprocess, "checking preprocess value")
if args.preprocess:
    for imagefile in images:
        # for images
        file = imagePath + "/" + imagefile
        img = cv2.imread(file,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (width, height))
        img = img/255.0
        img = img.astype(np.float32)
        allImages.append(img)
        # for masks
        file = maskPath + "/" + imagefile
        mask = cv2.imread(file,cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (width, height))
        mask[mask <=50] = 0
        mask[mask >50]  = 1
        maskImages.append(mask)
    allImagesNP = np.array(allImages, dtype=np.float32)
    maskImageNP = np.array(maskImages, dtype=np.int32)
    np.save("/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/Processed-Kvasir-SEG/allImages.npy", allImagesNP)
    np.save("/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/Processed-Kvasir-SEG/maskImages.npy", maskImageNP)

# for compitibility with the tensorflow
# allImagesNP = np.array(allImages)
# maskImageNP = np.array(maskImages)
else:
    print("Loading preprocessed data from .npy files")
    allImagesNP = np.load("/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/Processed-Kvasir-SEG/allImages.npy")
    maskImageNP = np.load("/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/Processed-Kvasir-SEG/maskImages.npy")
    maskImageNP = maskImageNP.astype(int) # all the values should be integer (0 or 1)

    print(allImagesNP.shape)
    print(allImagesNP.dtype)

    print(maskImageNP.shape)
    print(maskImageNP.dtype)

    # split train and test
    from sklearn.model_selection import train_test_split

    #90% train, 10% test
    X_train, X_test, y_train, y_test = train_test_split(allImagesNP, maskImageNP, test_size = 0.1, random_state=42)
    # 80% train, 10% val
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)

    print("X_train, X_val, y_train, t_val-----> shapes:")
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)