import cv2
from ultralytics import YOLO
import numpy as np
from flask import Flask, request, render_template, Response, jsonify
import base64

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 844)

app = Flask(__name__)
camera = cv2.VideoCapture(0)
model = YOLO("models/yolov8m-face.pt")

def detect_faces(image):
  results = model(image)
  boxes = []
  for result in results:
      for box in result.boxes:
          x1, y1, x2, y2 = map(int, box.xyxy[0])
          confidence = box.conf[0]
          class_id = int(box.cls[0])
          label = f'{model.names[class_id]}: {confidence:.2f}'
          boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label})
  return boxes

def gen_frames():
  while True:
    ret, frame = cap.read()
    
    if ret:
      results = model(frame)
      boxes = results[0].boxes
      # print("log")
      # print(results)
      
      for box in boxes:
        # print("log box")
        # print(box)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(
          frame,
          (x1, y1),
          (x2, y2),
          (50, 200, 129),
          2
        )
      ret, buffer = cv2.imencode('.jpg', frame)
      frame = buffer.tobytes()
      yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    data = request.json
    image_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes = detect_faces(frame)
    return jsonify({'boxes': boxes})
    # return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)