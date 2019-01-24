import numpy as np
import cv2 as cv2 

def extract_banana(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,230,255,cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    banana = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, contours, 0, (0,255,0), 3)
    return banana, image

def brown_mask(image, banana = None):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(grey,105,255,cv2.THRESH_BINARY_INV)
    brown_mask = thresh
    return brown_mask

def banana_mask(image, banana):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[banana],0,255,-1)
    return mask

def extract_features(image, banana_contour):
    banana_area = banana_mask(image, banana_contour)
    dark_area = brown_mask(image, banana_contour)
    mask = banana_area - dark_area
    
    [R_mean, G_mean] = RGB_feature(image, mask)
    [dark_percentage, larges_darkspot_percentage] = darkspot_area_feature(image, banana_area, dark_area)
    print([R_mean, G_mean, dark_percentage, larges_darkspot_percentage])
    return [R_mean, G_mean, dark_percentage, larges_darkspot_percentage]

def RGB_feature(image, pure_banana_mask):    
    mean, std = cv2.meanStdDev(image, mask = pure_banana_mask)
    G_mean = mean[1][0]
    R_mean = mean[2][0]
    return [G_mean, R_mean]

def darkspot_area_feature(image, banana_area, dark_area):
    contours, hierarchy = cv2.findContours(dark_area,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    biggest_spot_area = np.zeros(image.shape,np.uint8)
    if len(contours) > 1 : 
        biggest_spot = max(contours, key=cv2.contourArea)
        cv2.drawContours(biggest_spot_area,[biggest_spot],0,255,-1)
    dark_procaentage = (np.sum(dark_area)/np.sum(banana_area))
    larges_darkspot_percentage = (np.sum(biggest_spot_area)/np.sum(banana_area))
    return [dark_procaentage, larges_darkspot_percentage]

