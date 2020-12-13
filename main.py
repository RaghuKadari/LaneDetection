###############################################################################
# Module: main
# Brief : This module is the entry point and triggers Lane Detection algorithm  
#
#
#
###############################################################################

from LaneDetection import cLD 

def main():

     #run LD algorithm
     #input Video File name 
     LD = cLD('RoadVideo.mp4')    
     LD.run()

if __name__ == '__main__':
    main()
