# import computer vision library(cv2) in this code
import cv2

# main code
if __name__ == "__main__" :
    cap = cv2.VideoCapture(0)    
    while(cap.isOpened()):
        _, frame = cap.read()
        #gray=cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
            
            # show the frame on the newly created image window
        cv2.imshow('frames',frame)
            #cv2.imwrite('output.jpg',gray)
 
            # this condition is used to run a frames at the interval of 10 mili sec
            # and if in b/w the frame running , any one want to stop the execution .
        if cv2.waitKey(1) & 0xFF == 27 :
            break
    cap.release()
    cv2.destroyAllWindows()

