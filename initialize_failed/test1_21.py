import FaceMe
import os
import platform
import inspect
from time import sleep
from FaceMe.FaceMePython3SDK import WebcamInfo
from FaceMe.Recognizer import Recognizer
from FaceMe.LicenseManager import LicenseManager
import FaceMe.FaceMePython3SDK as FaceMe
#from FaceMe.FaceMeSDK import  FR_FAILED, FaceMeSDK
import FaceMe.FaceMeSDK as FC
import cv2

class FaceMeSDK():
    def __init__(self):
        self.__license_manager = None

        self.__data_manager = None
        self.__feature_scheme = None

        self.__camera_reocognizer_data_manager = None
        self.__camera_recognizer_feature_scheme = None

        self.__camera_reocognizer_adv_data_manager = None
        self.__camera_recognizer_adv_feature_scheme = None

        self.__recognizer = None
        self.__camera_capturers = {}
        self.__camera_recognizer = None
        self.__quality_detector = None
        self.__spoofing_manager = None
        self.__utils = None

        self.__license_key = ""
        self.__app_bundle_path = ""
        self.__app_data_path = ""
        self.__app_cache_path = ""

        self.__license_manager_config = {}
        self.__recognizer_config = {}
        self.__data_manager_config = {}
        self.__camera_recognizer_config = {}
        self.__camera_recognizer_callback = None

        self.__quality_detector_config = {}

        self.__spoofing_manager_config = {}
        self.__spoofing_result_handler = None

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
ret = FaceMe.FR_RETURN_OK
argd = {}
faceMe_images = []
print("=====0")
ret =  f.initialize()
print("=====1")
if FC.FR_FAILED(ret):
    print('initialize_SDK failed')
'''''
cameras = f.list_cameras()
print("======1",cameras)
ret = f.create_camera_recognizer_and_initialize()
if FC.FR_FAILED(ret):
    print('camera initialize failed')
    ret, []
print("======2")
ret, camera_list = f.__camera_recognizer.ListCameras()
ret,camera_list
print("======3")
'''''   
"""
recognize_options = {'extractOptions': FaceMe.FR_FEATURE_OPTION_FEATURE_LANDMARK
            | FaceMe.FR_FEATURE_OPTION_BOUNDING_BOX
            | FaceMe.FR_FEATURE_OPTION_POSE}
print("======1")
ret=f.create_camera_recognizer_and_initialize()
print("======2")
ret, recognize_results = f.recognize_faces(faceMe_images, recognize_options)
print("======3")
#ret= f.initialize(faceMe_images,license_key, app_bundle_path, app_cache_path, app_data_path, options=None)
print(recognize_results)
"""

while(cap.isOpened()): 
    ret,image  = cap.read()
    if FC.FR_SUCC(ret):
        ret,im = f.convert_opencvMat__to_faceMe_image(image)
        print('convert to faceMe succ',im)
        cv2.imshow("result_img",image)  
        if cv2.waitKey(1) & 0xFF == 27:
            break
        print(image)
    
    '''''''''''
    ret,fram_count = f.__recognizer.RecognizeFace(image,argd)
    image_size = len(image)
    print(image_size)
    
    ''''''''''''
    
        





cap.release()   
cv2.destroyAllWindows()



'''''
"""
if FC.FR_FAILED(ret):
    print("Initialize failed, return:", ret)
ret, camera_list = f.list_cameras()
if FC.FR_FAILED(ret):
    print("list_cameras failed, return: ", ret)
print(camera_list)
"""
