import os
import cv2
import torch
import shutil
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Folder to store uploaded videos and extracted frames
UPLOAD_FOLDER = 'uploads'
FRAMES_FOLDER = 'extracted_frames'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(FRAMES_FOLDER):
    os.makedirs(FRAMES_FOLDER)

# Function to extract frames from video
def extract_frames(video_path, output_folder, fps=1):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    if not cap.isOpened():
        return None, "Error opening video file."
    
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(frame_rate / fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_name = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_name, frame)
        frame_count += 1
    
    cap.release()
    return output_folder, None

# Detect objects in frames and generate event data
def detect_objects_in_frames(frames_folder, fps):
    event_data = []
    for frame_name in os.listdir(frames_folder):
        frame_path = os.path.join(frames_folder, frame_name)
        img = cv2.imread(frame_path)
        results = model(img)

        objects_in_frame = results.pandas().xyxy[0].name.tolist()
        timestamp = int(frame_name.split('_')[1].split('.')[0]) / fps
        event = {
            "time": f"{timestamp} seconds",
            "objects": objects_in_frame,
            "action": "moving" if "person" in objects_in_frame else "static"
        }
        event_data.append(event)
    return event_data

# Function to generate long-form summary
def generate_long_summary(events):
    summary = "The video begins with a quiet scene. "
    for event in events:
        objects = ", ".join(event["objects"])
        if not objects:
            continue
        summary += f"At {event['time']}, a {event['objects'][0]} is seen {event['action']}. "
        if len(event["objects"]) > 1:
            summary += f"Meanwhile, a {objects} also appears in the frame. "
    
    summary += "The video captures a series of everyday events, providing a glimpse of typical movements in the area."
    return summary

# Flask route to upload video
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        video_file = request.files['video']
        if video_file.filename == '':
            return redirect(request.url)

        if video_file:
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)

            # Extract frames from uploaded video
            frames_folder, error = extract_frames(video_path, FRAMES_FOLDER)
            if error:
                return f"Error processing video: {error}"

            # Detect objects and generate summary
            fps = 1  # You can adjust this to change how frequently frames are analyzed
            event_summary = detect_objects_in_frames(frames_folder, fps)
            long_summary = generate_long_summary(event_summary)

            # Clear extracted frames after summarization
            shutil.rmtree(FRAMES_FOLDER)
            os.makedirs(FRAMES_FOLDER)

            return render_template('summary.html', summary=long_summary)

    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)