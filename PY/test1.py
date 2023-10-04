import imp
from time import sleep
import FaceMe.FaceMeSDK as fm
import cv2
cap= cv2.VideoCapture(0)
while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    sleep(3)
image_path = "gray.jpg"
_,image_path  = cap.read()

_,image = fm.FaceMeSDK().convert_opencvMat__to_faceMe_image(image_path)

result = fm.FaceMeSDK().recognize_faces(image.shape[0])

print(result)

"""""
cap.release()
cv2.destroyAllWindows()
"""""