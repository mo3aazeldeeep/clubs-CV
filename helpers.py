import cv2
import numpy as np


def extract_features(image):
    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return descriptors


def build_histogram(descriptors, kmeans):
    histogram = np.zeros(len(kmeans.cluster_centers_))
    
    if descriptors is not None:
        clusters = kmeans.predict(descriptors)
        for c in clusters:
            histogram[c] += 1

    return histogram
