from flask import Flask, Response, request, redirect, url_for, render_template, send_file
import cv2
import torch
import numpy as np
from werkzeug.utils import secure_filename
import io
from PIL import Image

app = Flask(__name__)

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_objects(frame):
    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(rgb_frame)
    
    # Render results on the frame
    annotated_frame = results.render()[0]
    
    # Convert RGB image back to BGR for correct color display
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
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                # Read the uploaded image directly from the file stream
                image = Image.open(file.stream).convert('RGB')
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Perform object detection on the uploaded image
                detected_image = detect_objects(image_np)
                
                # Convert the result image to JPEG format
                _, img_encoded = cv2.imencode('.jpg', detected_image)
                img_io = io.BytesIO(img_encoded)
                
                return send_file(img_io, mimetype='image/jpeg')
        elif 'start_camera' in request.form:
            return redirect(url_for('index', camera=True))
        elif 'clear_camera' in request.form:
            return redirect(url_for('index', camera=False))
    return render_template('index.html', camera=request.args.get('camera'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
