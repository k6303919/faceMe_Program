from flask import Blueprint, Response
from flask_cors import cross_origin
import cv2

try:
    from queue import Queue, Empty, Full
except ImportError:
    from Queue import Queue, Empty, Full

q_frame = Queue(10)


def feed(frame):
    # feeding
    print("feeding ..")
    try:
        q_frame.put(frame, timeout=0.1)
    except Full:
        # print(".. queue full ..")
        pass


def gen_frames():
    while True:
        try:
            # querying
            print(".. quering")
            frame = q_frame.get(timeout=0.1)
            if getattr(frame, "any", None) != None and frame.any():
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    # encode failed
                    pass
            else:
                # no any frame
                pass
        except Empty as e:
            # preparing
            # print("..preparing frame..")
            pass
        except Exception as e:
            print("unknown error occured ......")
            pass


app = Blueprint('app_stream', __name__)


@app.route('/')
def index():
    return "stream route"


@app.route('/mjpeg')
@cross_origin()
def feed_mjpeg():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
