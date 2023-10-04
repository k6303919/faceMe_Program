import imp
import cv2
from FaceDK import Facedk

cap= cv2.VideoCapture(1)
while True:
    ret,image  = cap.read()
    cv2.imshow("t",image)
    ret, result = Facedk(image)
    print(result)
    if  cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()