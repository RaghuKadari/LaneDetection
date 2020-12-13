###############################################################################
# Module: LaneDetection
# Brief : This module performs LaneDetection and outputs a video file 
#
#
###############################################################################

import sys
import cv2 as cv2 
import numpy as np
import time
import matplotlib.pyplot as plt
import math

class cLD:
    
    def __init__(self, filename):
        self.filename = filename 
    
    def Houglines(self, img,imcany):

        lines = cv2.HoughLines(imcany, 1, np.pi / 180, 200, None, 50, 10) 
        if lines is not None:

            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(img, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
                
                plt.imshow(img,cmap = 'gray')
                plt.title('Original Image')
                plt.show()

    ###########################################################################
    # Brief : This function will find the cordinates to draw the lines
    #
    #
    ###########################################################################
    def FindCordinates(self, img, params):

        if params is not None: 
            #slope
            m = params[0]

            #intercept
            c = params[1]

            #we want the lane detection should start from bottom of the image
            # which is the height of the image.
            y1 = img.shape[0]


            #equation of a line in y = mx + c
            # we have y, m, c to find x
            x1 = int((y1 - c)/m)


            #we have a point and we know the slope
            # the next point can be decided based on how 
            # far we want to draw the lanes form the bottom of the 
            # image. 
            y2 = int(y1 * 0.6) # try 60%

            x2 = int((y2 - c)/m)

            return np.array([x1,y1,x2,y2])




    ###########################################################################
    # Brief : This function will draw lines on the image.  
    #
    #
    ###########################################################################
    def DrawLines(self, img, lines):
        
        lineimage = np.zeros_like(img)

        left = []
        right = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                
                #find slope and intercept
                params = np.polyfit((x1,x2), (y1, y2), 1)

                slope = params[0]
                intercept = params[1]
               
                if slope < 0:
                    left.append((slope, intercept))
                else: 
                    right.append((slope, intercept))

            # find the average slope to smoothen the 
            # lane detection lines. 
            leftLaneparams = np.average(left, axis=0)
            rightLaneparams = np.average(right, axis=0)
            
            # find the Cordinates
            if (leftLaneparams.any() and rightLaneparams.any()) is not None: 
                x1,y1,x2,y2 = self.FindCordinates(img, leftLaneparams)
                w1,z1,w2,z2 = self.FindCordinates(img, rightLaneparams)

                #Draw lines
                #cv2.line(lineimage, (x1,y1), (x2, y2), (255,0,0),9)
                #cv2.line(lineimage, (w1,z1), (w2, z2), (255,0,0),9)
                
                cv2.line(img, (x1,y1), (x2, y2), (0,255,0),9)
                cv2.line(img, (w1,z1), (w2, z2), (0,255,0),9)
            
                cv2.line(img, (x2, y2), (w2, z2), (255,0,0),9)
                return lineimage       
        else:
            print("no lines found")


    ###########################################################################
    # Brief : This function applies Hough Lines on the iamge 
    #
    #
    ###########################################################################
    def ProbablisticHoughLines(self, maskimage, img):
        
        #Apply Probablistic hough Transform.
        lines = cv2.HoughLinesP(maskimage, 2, np.pi/180, 100, None, minLineLength=40,  maxLineGap=5)
        
        outputImage = self.DrawLines(img, lines)

        convolveimage = cv2.addWeighted(outputImage, 0.8, img, 1, 1)

        return convolveimage  

    ###########################################################################
    # Brief : This function applies Canny Edge detection on the Image. 
    #
    #
    ###########################################################################

    def CannyEdgeDetect(self, img):

        #convert to Gray Image for better detection
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(imgray, (5,5), 0)
        #Apply Canny Edge Detection, threshold should always be in 
        #either 1:2 or 1:3 ratio as per OpenCV, choosing 1:3 ration
        imcany = cv2.Canny(blur,50,150)

        return imcany

    ###########################################################################
    # Brief : This function will focus lane detection on the needed pixels
    #
    #
    ###########################################################################
    def ROI(self, image):
            
        height = image.shape[0]
       
        Trianglemask = np.array([
            [(200,height), (1100, height), (500,250)]
            ])
        
        roi = np.zeros_like(image)
        
        cv2.fillPoly(roi, Trianglemask, 255)
       
        maskImage  = cv2.bitwise_and(image, roi) 

        return maskImage

    ###########################################################################
    # Brief : This function will initiate the LD algorithm
    #
    #
    ###########################################################################
    def run(self):

        #Open the video file 
        frames = cv2.VideoCapture(self.filename)

        # check if you are able to open the video file 
        if (frames.isOpened() == False):
            
            print("Error reading video file")

        else:
            #(Frame width , Frame height) 
            size = (int(frames.get(3)), int(frames.get(4)))

            # Below VideoWriter object will create outputvideo 
            vwrite = cv2.VideoWriter('LaneDetectVideo.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

            while(frames.isOpened()): 
                    
                # Capture frame-by-frame
                ret, image = frames.read()
                
                img = np.copy(image)

                canny = self.CannyEdgeDetect( img)

                mask = self.ROI(canny)

                #cv2.imshow('ROI', mask)
                
                #self.Houglines(self,img, imcany)
                Outputimage = self.ProbablisticHoughLines(mask, image)


                vwrite.write(Outputimage)
                #cv2.imshow('result', Outputimage)

                if ((cv2.waitKey(1) & 0xFF) == ord('q')):
                    break

        # When everything done, release the capture
        video.release()
        cv2.destroyAllWindows()

