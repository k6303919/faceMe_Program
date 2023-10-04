import imp
from time import sleep
from FaceMe.FaceMePython3SDK import WebcamInfo
import FaceMe.FaceMeSDK as fm
import cv2

cap=cv2.VideoCapture(1)
f = fm.FaceMeSDK()
while(cap.isOpened()):
    ret,image  = cap.read()
    #_,image = cv2.flip(image , 1)
    ret,im = f.convert_opencvMat__to_faceMe_image(image)
    result = f.recognize_faces(im)
    cv2.imshow("result_img",image)  
    if cv2.waitKey(1) & 0xFF == 27:
        break
    print(result)


cap.release()   
cv2.destroyAllWindows()


