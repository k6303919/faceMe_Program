import json
import locale

default_langage = r'''
    {
        "en_US": {
            "setting-complete": "Save completed, please restart application to apply the new settings.",
            "credential-not-found": "can not found credential.",
            "credential-match-failed": "credential match failed.",
            "prompt-input-key": "please input the key",
            "title-of-radio-metry-cam": "radio metry",
            "title-of-usb-cam": "video",
            "degree-C": "degree C",
            "degree-F": "degree F",
            "message-normal-temperature": "Normal",
            "message-normal-label-detecting-1": "Getheermal",
            "message-normal-label-detecting-2": "Working",
            "message-alert-temperature": "Warning",
            "message-alert-label-detecting-1": "Please wait",
            "message-alert-label-detecting-2": "Second detection",
            "control-label-device-id": "Device ID",
            "control-label-temperature-alert-highest": "Alert level",
            "control-label-temperature-background-reference": "Temp. fix reference",
            "control-label-upload-url": "Server url",
            "control-label-temperature-valid-range": "Valid temp. detection",
            "control-label-temperature-unit": "Temp. Unit",
            "control-label-temperature-unit-c": "Celsius",
            "control-label-temperature-unit-f": "Fahrenheit",
            "control-label-cv-ratio": "Cascade ratio",
            "control-label-cascade-crop": "Cascade Crop",
            "control-label-cascade-crop-left": "Left",
            "control-label-cascade-crop-top": "Top",
            "control-label-cascade-crop-right": "Right",
            "control-label-cascade-crop-bottom": "Bottom",
            "control-label-cascade-detective-area": "Cascade area",
            "control-label-cascade-detective-area-edit-tip": "Editor",
            "control-label-gethermal-crop": "Gethermal Crop",
            "control-label-gethermal-crop-left": "Left",
            "control-label-gethermal-crop-top": "Top",
            "control-label-gethermal-crop-right": "Right",
            "control-label-gethermal-crop-bottom": "Bottom",
            "control-label-video-test": "Video test",
            "control-label-video-test-facecascade": "Face-cascade",
            "control-label-video-test-radiometry": "Radio-metry",
            "control-label-temperature-fix": "Temp. Fixed",
            "control-label-temperature-formula-used": "Formula",
            "control-label-temperature-formula-used-distribution": "Distribution",
            "control-label-temperature-formula-used-distance": "Distance",
            "control-button-close": "Close",
            "control-button-save": "Save",
            "message-setting-face-cascade-crop-tip": "The area cropped is in the screen 'Face-Cascade' (Left)",
            "message-setting-radio-metry-crop-tip": "The area cropped was in the screen 'Radio-Metry' (Right)",
            "message-setting-facecascade-area-tip": "Click & Drag your mouse to apply detection area\n\nNotice that:\n  It will apply with 'the face crop values'\n\nHotkey:\n  [Ctrl+s] Save and quit\n  [Ctrl+z] Remove last area\n  [Q] Quit without save\n  [Mouse-Left] Drag to select area\n  [Mouse-Right] Remove the selected area"
        }
    }
    '''
lc = locale.getlocale()[0]  # tuple of locale withoout encoding

try:
    with (open("language.json", "r", encoding='UTF-8')) as fo:
        text = fo.read()
        language_pack = json.loads(text)[lc]
except Exception as e:
    lc = 'en_US'
    language_pack = json.loads(default_langage)[lc]


def getText(key):
    return language_pack[key]
