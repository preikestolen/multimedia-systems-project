import cv2
import numpy as np
import math

image = cv2.imread("cells6.jpeg")
# getting the image
# change the image to try different samples
original = image.copy()
# copying the image to make a drawing on it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# converting the image to hsv format

hsvLowerRed = np.array([156,60,0])
hsvUpperRed = np.array([179,115,255])
# define hsv color range for red

purple1 = np.uint8([[[138,43,226]]])
purple2 = np.uint8([[[75,0,130]]])
# getting the colors in rgb format
hsvp1 = cv2.cvtColor(purple1, cv2.COLOR_RGB2HSV)
hsvp2 = cv2.cvtColor(purple2, cv2.COLOR_RGB2HSV)
# converting purple colors to hsv format
hsvLowerPurple = hsvp1 - (10, 100, 100)
hsvUpperPurple = hsvp2 + (10, 255, 255)
# define hsv color range for purple

# we have hsv color range for red and purple
# so we can segment objects in color red and purple now.
# we need to do thresholding on a colored image by looking the hsv ranges
# it is also called color thresholding

maskRed = np.empty_like(hsv)
maskPurple = np.empty_like(hsv)
# returns an empty array with the same size of the image. (like 3x3 matrix)

# shape returns tuple like (x,y) -> (3,4) or (10,10)
# i or shape[0] for rows amount
# j or shape[1] for columns amount

# thresholding for color red
for i in range(0, maskRed.shape[0]):
    for j in range(0, maskRed.shape[1]):
        # color thresholding (works like inRange function)
        if ((hsvUpperRed >= hsv[i,j]) & (hsv[i,j] >= hsvLowerRed)).all():
            # if in that range, color it white
            maskRed[i,j] = 255
        else:
            # if not, color it black
            maskRed[i,j] = 0

# thresholding for color purple
for i in range(0, maskPurple.shape[0]):
    for j in range(0, maskPurple.shape[1]):
        # color thresholding (works like inRange function)
        if ((hsvUpperPurple >= hsv[i,j]) & (hsv[i,j] >= hsvLowerPurple)).all():
            # if in that range, color it white
            maskPurple[i,j] = 255
        else:
            # if not, color it black
            maskPurple[i,j] = 0

# if you have pixel = numpy.array([156, 60, 0])
# then low_blue <= pixel and pixel <= high_blue each give you a boolean array with the element-wise comparison
# unlike with normal Python comparison operators, you can't chain comparisons with NumPy arrays
# so to combine the two comparisons you need to use & (logical and, since you want both to be true)
# finally, to combine the element-wise results to an overall result, use .all 
# since you want all elements to be true (all channels to be within the respective intervals)
# is_blue = ((low_blue <= pixel) & (pixel <= high_blue)).all()

maskRed = maskRed.astype('uint8')
maskPurple = maskPurple.astype('uint8')
# image normalization is done above

maskRed = cv2.cvtColor(maskRed, cv2.COLOR_HSV2BGR)
maskRed = cv2.cvtColor(maskRed, cv2.COLOR_BGR2GRAY)
cv2.imshow('thresholded red', maskRed)
cv2.waitKey()
# converting from hsv to gray for getting thresholded binary image

maskPurple = cv2.cvtColor(maskPurple, cv2.COLOR_HSV2BGR)
maskPurple = cv2.cvtColor(maskPurple, cv2.COLOR_BGR2GRAY)
cv2.imshow('thresholded purple', maskPurple)
cv2.waitKey()
# converting from hsv to gray for getting thresholded binary image

kernelRed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
openingRed = cv2.morphologyEx(maskRed, cv2.MORPH_OPEN, kernelRed, iterations=1)
closeRed = cv2.morphologyEx(openingRed, cv2.MORPH_CLOSE, kernelRed, iterations=2)
# smoothing the image to remove unnecessary details

kernelPurple = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
openingPurple = cv2.morphologyEx(maskPurple, cv2.MORPH_OPEN, kernelPurple, iterations=1)
closePurple = cv2.morphologyEx(openingPurple, cv2.MORPH_CLOSE, kernelPurple, iterations=2)
# smoothing the image to remove unnecessary details

cv2.imshow('removed unnecessary details for red', closeRed)
cv2.waitKey()
cv2.imshow('removed unnecessary details for purple', closePurple)
cv2.waitKey()

contoursRed, hierarchyRed = cv2.findContours(closeRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursPurple, hierarchyPurple = cv2.findContours(closePurple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

minimumArea = 200
averageCellArea = 650
connectedCellArea = 1000
# if the image size is different, you need to change minimumArea
# this is the solution to overlapping cells challenge
# when the cells are so big and connected, counting them as one is wrong
# as a solution, I created pre defined values for average cell sizes
# and when I get big and seemed to be single cell, 
# I make a guess on how many cells it can have inside
# by dividing the area of big cell by the average area of connected/big cell
# also, to understand if the circle is cell,
# I check if the area of circle is larger than the minimum cell area I defined
cellsRed = 0
bigCellsRed = 0
originalRed = original.copy()
for cell in contoursRed:
    areaRed = cv2.contourArea(cell)
    if areaRed > minimumArea:
        cv2.drawContours(originalRed, [cell], -1, (35,255,12), 2)
        if areaRed > connectedCellArea:
            cellsRed += math.ceil(areaRed / averageCellArea)
            # to guess how many cells are there in this big connected cell
            # I divide the total cell area by the average cell area
            # this gives a very good result
            bigCellsRed += 1
        else:
            cellsRed += 1
print("big red cells -> " + str(bigCellsRed))
print("total red cells -> " + str(cellsRed))
cv2.imshow('close', closeRed)
cv2.waitKey()
cv2.imshow('original', originalRed)
cv2.waitKey()

cellsPurple = 0
bigCellsPurple = 0
originalPurple = original.copy()
for cell in contoursPurple:
    areaPurple = cv2.contourArea(cell)
    if areaPurple > minimumArea:
        cv2.drawContours(originalPurple, [cell], -1, (35,255,12), 2)
        if areaPurple > connectedCellArea:
            cellsPurple += math.ceil(areaPurple / averageCellArea)
            # to guess how many cells are there in this big connected cell
            # I divide the total cell area by the average cell area
            bigCellsPurple += 1
        else:
            cellsPurple += 1
print("big purple cells -> " + str(bigCellsPurple))
print("total purple cells -> " + str(cellsPurple))
cv2.imshow('close', closePurple)
cv2.waitKey()
cv2.imshow('original', originalPurple)
cv2.waitKey()