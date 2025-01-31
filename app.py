from flask import Flask, Response, request, redirect, url_for, render_template, send_from_directory
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    annotated_frame = results.render()[0]
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    return annotated_frame

def generate_frames():
    # Return an error or fallback content if no camera is available
    if not os.path.exists('/dev/video0'):
        return b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'Camera not available\r\n'
    
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        detected_frame = detect_objects(frame)
        resized_frame = cv2.resize(detected_frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                image = cv2.imread(file_path)
                detected_image = detect_objects(image)
                detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
                cv2.imwrite(detected_image_path, detected_image)
                
                return render_template('index.html', image_url=url_for('uploaded_file', filename='detected_' + filename), camera=False)
        elif 'start_camera' in request.form:
            # Only start the camera if we're not in a Render environment
            if os.path.exists('/dev/video0'):
                return redirect(url_for('index', camera=True))
            else:
                return render_template('index.html', camera=False, error="Camera not available.")
        elif 'clear_camera' in request.form:
            return redirect(url_for('index', camera=False))
    return render_template('index.html', camera=request.args.get('camera'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, threaded=True)
