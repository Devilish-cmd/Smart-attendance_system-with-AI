from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import threading
import atexit
from face_engine import load_known_faces, recognize_face
from liveness import detect_liveness
from utils import mark_attendance
from chatbot import chatbot_response

app = Flask(__name__)

# --- Camera setup with safe release on shutdown ---
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
atexit.register(lambda: camera.release())

# --- Thread-safe prev_frame handling ---
prev_frame = None
frame_lock = threading.Lock()

load_known_faces()


def generate_frames():
    global prev_frame

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            faces, names = recognize_face(frame)

            # Thread-safe access to prev_frame
            with frame_lock:
                current_prev = prev_frame.copy() if prev_frame is not None else None

            live = detect_liveness(frame, current_prev)

            for (top, right, bottom, left), name in zip(faces, names):
                if not live:
                    label = "Fake Face"
                elif name == "Unknown":
                    label = "Unknown Alert"
                else:
                    status = mark_attendance(name)
                    label = f"{name} - {status}"

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Update prev_frame safely
            with frame_lock:
                prev_frame = frame.copy()

            # Fix: use separate variable instead of overwriting frame
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

    except GeneratorExit:
        # Client disconnected — exit cleanly
        pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()

    # Fix: validate input before accessing keys
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    response = chatbot_response(data['query'])
    return jsonify({"response": response})


if __name__ == "__main__":
    # Fix: use environment variable instead of hardcoded debug=True
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode)