from flask import Flask, Response, request, redirect, url_for, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load pre-trained YOLOv5 model using ultralytics
model = YOLO('yolov5s.pt')  # Ensure you have the YOLOv5 weights file locally

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
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform object detection
        detected_frame = detect_objects(frame)
        
        # Resize frame to handle performance
        resized_frame = cv2.resize(detected_frame, (640, 480))
        
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', resized_frame)
        frame = buffer.tobytes()
        
        # Yield the frame in MJPEG stream format
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
        if 'start_camera' in request.form:
            return render_template('index.html', camera=True)
        elif 'clear_camera' in request.form:
            return redirect(url_for('index'))
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image = cv2.imread(file_path)
                detected_image = detect_objects(image)
                detected_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
                cv2.imwrite(detected_image_path, detected_image)
                return render_template('index.html', image_url=url_for('uploaded_file', filename='detected_' + filename))
    return render_template('index.html', camera=False)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
