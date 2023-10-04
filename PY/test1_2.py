import FaceMe
from time import sleep
from FaceMe.FaceMePython3SDK import WebcamInfo
from FaceMe.Recognizer import Recognizer
import FaceMe.FaceMePython3SDK as FaceMe
#from FaceMe.FaceMeSDK import  FR_FAILED, FaceMeSDK
import FaceMe.FaceMeSDK as FC
import cv2


class ParseArg():
    command = ""
    chipset = "NV"
    far = "1E-5"
    fdm = "DEFAULT"
    frm = "DEFAULT"

    image_path = ""
    image_path2 = ""
    license_key = ""
    user_id = -1
    user_name = ""
    face_id = -1
    face_attribute_option = FaceMe.FR_FEATURE_OPTION_ALL
    candidate_count = 1
    min_face_width_ratio = 0.05

    device_id = -1
    device_path = ""
    width = 1280
    height = 720
    fps = 0
    detect_thread = 0
    extract_thread = 0
    detect_batchsize = 0   # GPU: 4, 8, 16, VPU: VPU count, CPU: 1
    extract_batchsize = 0   # GPU: 4, 8, 16, VPU: VPU count, CPU: 1

    advfrm = "DEFAULT"
    adv_etract_image_only = True
    need_image_quality_check = False
    need_extract_face_images = False
    need_extract_frame_image = False
    is_adv_extraction_enabled = False
    adv_extraction_interval = 500
    device_paths = []
    device_ids = []

    precision_mode = "STANDARD"

cap=cv2.VideoCapture(1)
f =FC .FaceMeSDK()

while(cap.isOpened()): #True 持續運行
    ret,image  = cap.read()
    #_,image = cv2.flip(image , 1)
    ret = FaceMe.FR_RETURN_OK
    ret = f.create_camera_recognizer_and_initialize()
    if FC.FR_FAILED(ret):
            print("Recognizer initialize failed")  
  
    ret,im = f.convert_opencvMat__to_faceMe_image(image)
    options = {'recognizeOptions': f.FR_FEATURE_OPTION_ALL}
    ret,result = f.recognize_faces(im,options)
    cv2.imshow("result_img",image)  
    if cv2.waitKey(1) & 0xFF == 27:
        break
    print(result)

    
    
"""""
def get_face_thumbnail(f, face_id, options=None):
    ret,Im = f.get_face_thumbnail(201)
    
 """""   
             
   

cap.release()   
cv2.destroyAllWindows()

    #ret,im2 = f.__recognizer_config
    #ret,im2 = recognizer.RecognizeFace(im,argd)
    #image_size = len(im2)

    
    #ret = FaceMeSDK.__create_recognzier_and_initialize()
    #if FR_FAILED(ret):
        #    print("Recognizer initialize failed")
        #   ret, []
    
    
    #ret, detect_result = f.detect_face(image,options)
    #if FR_FAILED(ret):
    #    print("detect_face failed, return: ", ret)
    #    break
    #print("detect_result: ", detect_result)
   
