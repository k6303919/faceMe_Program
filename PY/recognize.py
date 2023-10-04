import FaceMe.FaceMePython3SDK as FaceMe
from FaceMe.FaceMeSDK import FaceMeSDK, FR_FAILED, FR_SUCC
import FaceMePythonSample

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

command_args = ParseArg()

ret = FaceMePythonSample.parse_command_line(['recognizeface','-photo', '0820_KEVIN.jpg'], command_args)
command_function = FaceMePythonSample.switch_command(command_args)

if command_function is None:
    FaceMePythonSample.wrong_command()
ret = FaceMePythonSample.initialize_SDK(command_args)

if FR_FAILED(ret):
    print("initialize SDK failed")

FaceMePythonSample.recognize_face(command_args)