from flask import Flask, render_template, Response, redirect, url_for, request, flash, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import time, os

app = Flask(__name__)
socketio = SocketIO(app)
cam = 1


@app.route('/')
def welcome():
    return render_template('fitWelcome.html')


@app.route('/explore')
def explore():
    return redirect(url_for('fit_home'))


@app.route('/home')
def fit_home():
    return render_template('homepageSenam.html')


@app.route('/aboutus')
def fit_aboutus():
    return render_template('aboutusSenam.html')


@app.route('/tutorial')
def fit_tutorial():
    return render_template('tutorialSenam.html')


@app.route('/contact')
def fit_contact():
    return render_template('contactSenam.html')


@app.route('/model')
def open():
    return render_template('modelSenam.html')


@app.route('/video1')
def video():
    return Response(start_model(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video2')
def video2():
    return Response(start_model2(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video3')
def video3():
    return Response(start_model3(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video4')
def video4():
    return Response(start_model4(cam), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/videoResult',  methods=['POST'])
def videoR():
    # Check if the request contains a file
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)

    # Get the uploaded video file
    video_file = request.files['video']

    # Specify the directory path to save the uploaded video
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the uploaded video to the specified directory
    video_path = os.path.join(upload_dir, 'temporary.mp4')
    video_file.save(video_path)

    # Pass the video path to the template
    return jsonify({'video_path': video_path})


@app.route('/displayVideo1/<video_name>')
def displayVideo1(video_name):
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    video_path = os.path.join(upload_dir, video_name)
    return Response(start_model(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/displayVideo2/<video_name>')
def displayVideo2(video_name):
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    video_path = os.path.join(upload_dir, video_name)
    return Response(start_model2(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/displayVideo3/<video_name>')
def displayVideo3(video_name):
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    video_path = os.path.join(upload_dir, video_name)
    return Response(start_model3(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/displayVideo4/<video_name>')
def displayVideo4(video_name):
    upload_dir = os.path.join(app.instance_path, 'uploaded_videos')
    video_path = os.path.join(upload_dir, video_name)
    return Response(start_model4(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_model')
def start_model(video_path):
    if video_path == 1:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print(video_path)
    else:
        cap = cv2.VideoCapture(video_path)
        print(video_path)

    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        # Display the countdown number on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), font, 4, (0, 0, 255), 2,
                    cv2.LINE_AA)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Wait for 1 second
        time.sleep(1)
    while True:
        squat_count = 0
        squat_count_error = 0
        expected_stages = [0, 1, 2, 1, 0]
        expected_stages_w = [0, 1, 3, 1, 0]
        current_stages = []
        tick = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        landmarks = []
        for val in range(1, 33 + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        model = joblib.load('squat_model.pkl')

        mean = np.load('mean.npy')
        std_dev = np.load('std_dev.npy')

        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std_dev

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )

                try:
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                    X = pd.DataFrame([row], columns=landmarks)

                    # Scale the input data
                    Xy = scaler.transform(X)
                    body_language_prob = model.predict_proba(Xy)
                    body_language_class = model.predict(Xy)
                    print(body_language_class)

                    current_stage = -1

                    if body_language_class == 0:
                        current_stage = 0  # Standing
                    elif body_language_class == 1:
                        current_stage = 1  # Descending/Ascending
                    elif body_language_class == 2:
                        current_stage = 2  # Squatting
                    elif body_language_class == 3:
                        current_stage = 3  # Squatting_Wrong

                    # Add the current stage to the list of stages
                    if current_stage == 0 and tick == 0:
                        current_stages.append(current_stage)
                        tick = 1
                    elif current_stage == 1 and tick == 1:
                        current_stages.append(current_stage)
                        tick = 2
                    elif current_stage == 2 and tick == 2:
                        current_stages.append(current_stage)
                        tick = 3
                    elif current_stage == 3 and tick == 2:
                        current_stages.append(current_stage)
                        tick = 3
                        # Draw the status box
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Squat too deep!', (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.putText(image, 'When going down, make sure the line between', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.putText(image, 'hip and knee is parallel to your feet.', (120, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                    elif current_stage == 3 and tick == 3:
                        current_stages.pop()
                        current_stages.append(current_stage)
                        tick = 3
                        # Draw the status box
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Squat too deep!', (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.putText(image, 'When going down, make sure the line between', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.putText(image, 'hip and knee is parallel to your feet.', (120, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                    elif current_stage == 1 and tick == 3:
                        current_stages.append(current_stage)
                        tick = 4
                    elif current_stage == 0 and tick == 4:
                        current_stages.append(current_stage)
                        tick = 5
                    # Trim the list if it exceeds 5 elements
                    if len(current_stages) > 5:
                        current_stages.pop(0)

                    # Check if the current stages match the expected sequence
                    if current_stages == expected_stages:
                        # Increment the squat count and reset the current stages list
                        squat_count += 1
                        current_stages = []
                        tick = 0
                    elif current_stages == expected_stages_w:
                        squat_count_error += 1
                        current_stages = []
                        tick = 0

                    class_names = {
                        0: 'standing up',
                        1: 'descending ascending',
                        2: 'squatting down',
                        3: 'squatting down incorrect'
                    }

                    # Send data to the front end
                    socketio.emit('update', {
                        'technique': 'Squat',
                        'class': class_names[np.argmax(body_language_prob)],
                        'classNo': int(body_language_class[0]),
                        'correct': squat_count,
                        'incorrect': squat_count_error
                    })

                except Exception as e:
                    print("Error:", e)

                # Encode frame as JPEG image
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_model2')
def start_model2(video_path):
    if video_path == 1:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        # Display the countdown number on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), font, 4, (0, 0, 255), 2,
                    cv2.LINE_AA)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Wait for 1 second
        time.sleep(1)
    while True:
        bicep_count = 0
        bicep_count_error = 0
        expected_stages = [0, 2, 0]
        expected_stages1 = [1, 3, 1]
        expected_stages_w = [0, 4, 0]
        expected_stages_w1 = [1, 5, 1]
        current_stages = []
        tick = 0
        tack = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        landmarks = []
        for val in range(1, 33 + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        model = joblib.load('bicep_model.pkl')

        # Load the mean and standard deviation from the files
        mean = np.load('mean2.npy')
        std_dev = np.load('std_dev2.npy')

        # Initialize StandardScaler with the loaded mean and standard deviation
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std_dev

        # Initiate holistic model
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Make Detection
                results = pose.process(image)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                 circle_radius=2)
                                          )

                try:
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                    results.pose_landmarks.landmark]).flatten()
                    X = pd.DataFrame([row], columns=landmarks)

                    # Scale the input data
                    Xy = scaler.transform(X)
                    body_language_prob = model.predict_proba(Xy)
                    body_language_class = model.predict(Xy)
                    print(body_language_class)

                    max_prob = np.max(body_language_prob)

                    current_stage = -1
                    if body_language_class == 0:
                        current_stage = 0  # Down left
                    elif body_language_class == 1:
                        current_stage = 1  # Down right
                    elif body_language_class == 2:
                        current_stage = 2  # Up left
                    elif body_language_class == 3:
                        current_stage = 3  # Up right
                    elif body_language_class == 4 and max_prob >= 0.9:
                        current_stage = 4  # Up incorrect left
                    elif body_language_class == 5 and max_prob >= 0.9:
                        current_stage = 5  # Up incorrect right

                    # left
                    if current_stage == 0 and tick == 0:
                        current_stages.append(current_stage)
                        tick = 1
                    elif current_stage == 2 and tick == 1:
                        current_stages.append(current_stage)
                        tick = 2
                    elif current_stage == 0 and tick == 2:
                        current_stages.append(current_stage)
                        tick = 10
                    elif current_stage == 4 and tick == 1:
                        current_stages.append(current_stage)
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        tick = 4
                    elif current_stage == 4 and tick == 2:
                        current_stages.pop()
                        current_stages.append(current_stage)
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        tick = 4
                    elif current_stage == 4:
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 0 and tick == 4:
                        current_stages.append(current_stage)
                        tick = 10

                    # right
                    if current_stage == 1 and tack == 0:
                        current_stages.append(current_stage)
                        tack = 1
                    elif current_stage == 3 and tack == 1:
                        current_stages.append(current_stage)
                        tack = 2
                    elif current_stage == 1 and tack == 2:
                        current_stages.append(current_stage)
                        tack = 10
                    elif current_stage == 5 and tack == 1:
                        current_stages.append(current_stage)
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        tack = 4
                    elif current_stage == 5 and tack == 2:
                        current_stages.pop()
                        current_stages.append(current_stage)
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        tack = 4
                    elif current_stage == 5:
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Elbow should not go up!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When lifting up, make sure the elbow', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'align with the shoulder.', (235, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 1 and tack == 4:
                        current_stages.append(current_stage)
                        tack = 10

                    # Trim the list if it exceeds 3 elements
                    if len(current_stages) > 3:
                        current_stages.pop(0)

                    # Check if the current stages match the expected sequence
                    if current_stages == expected_stages:
                        # Increment the squat count and reset the current stages list
                        bicep_count += 1
                        current_stages = []
                        tick = 0
                    elif current_stages == expected_stages_w:
                        bicep_count_error += 1
                        current_stages = []
                        tick = 0

                    if current_stages == expected_stages1:
                        # Increment the squat count and reset the current stages list
                        bicep_count += 1
                        current_stages = []
                        tack = 0
                    elif current_stages == expected_stages_w1:
                        bicep_count_error += 1
                        current_stages = []
                        tack = 0

                    # Display Class
                    class_names = {
                        0: 'down left side',
                        1: 'down right side',
                        2: 'up left side',
                        3: 'up right side',
                        4: 'incorrect up left side',
                        5: 'incorrect up right side'
                    }

                    # Send data to the front end
                    socketio.emit('update', {
                        'technique': 'Bicep Curl',
                        'class': class_names[current_stage],
                        'classNo': current_stage,
                        'correct': bicep_count,
                        'incorrect': bicep_count_error
                    })

                except Exception as e:
                    print("Error:", e)

                # Encode frame as JPEG image
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_model3')
def start_model3(video_path):
    if video_path == 1:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_path)
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        # Display the countdown number on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), font, 5, (0, 0, 255), 3,
                    cv2.LINE_AA)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Wait for 1 second
        time.sleep(1)
    while True:
        situp_count = 0
        situp_count_error = 0
        expected_stages = [0, 1, 0]
        expected_stages_w = [0, 2, 0]
        current_stages = []
        tick = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        landmarks = []
        for val in range(1, 33 + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        model = joblib.load('situp_model.pkl')

        # Load the mean and standard deviation from the files
        mean = np.load('mean3.npy')
        std_dev = np.load('std_dev3.npy')

        # Initialize StandardScaler with the loaded mean and standard deviation
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std_dev

        # Initiate holistic model
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detection
                results = pose.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                 circle_radius=2)
                                          )

                try:
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                    results.pose_landmarks.landmark]).flatten()
                    X = pd.DataFrame([row], columns=landmarks)

                    # Scale the input data
                    Xy = scaler.transform(X)
                    body_language_prob = model.predict_proba(Xy)
                    body_language_class = model.predict(Xy)
                    print(body_language_class)

                    current_stage = -1
                    max_prob = np.max(body_language_prob)

                    if body_language_class == 0:
                        current_stage = 0  # Down
                    elif body_language_class == 1:
                        current_stage = 1  # Up
                    elif body_language_class == 2 and max_prob >= 0.9:
                        current_stage = 2  # Up Incorrect

                    # Add the current stage to the list of stages
                    if current_stage == 0 and tick == 0:
                        current_stages.append(current_stage)
                        tick = 1
                    elif current_stage == 1 and tick == 1:
                        current_stages.append(current_stage)
                        tick = 2
                    elif current_stage == 1 and tick == 3:
                        tick = 4
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Feet must not float!', (245, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to fix your', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'feet to the ground.', (245, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 2 and tick == 1:
                        current_stages.append(current_stage)
                        tick = 3
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Feet must not float!', (245, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to fix your', (118, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'feet to the ground.', (245, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 0 and (tick == 2 or tick == 3 or tick == 4):
                        current_stages.append(current_stage)
                        tick = 5

                    if len(current_stages) > 3:
                        current_stages.pop(0)

                    # Check if the current stages match the expected sequence
                    if current_stages == expected_stages:
                        # Increment the squat count and reset the current stages list
                        situp_count += 1
                        current_stages = []
                        tick = 0
                    elif current_stages == expected_stages_w:
                        situp_count_error += 1
                        current_stages = []
                        tick = 0

                    # Display Class
                    class_names = {
                        0: 'down',
                        1: 'up',
                        2: 'up incorrect'
                    }

                    # Send data to the front end
                    socketio.emit('update', {
                        'technique': 'Sit Up',
                        'class': class_names[current_stage],
                        'classNo': current_stage,
                        'correct': situp_count,
                        'incorrect': situp_count_error
                    })

                except Exception as e:
                    print("Error:", e)

                # Encode frame as JPEG image
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_model4')
def start_model4(video_path):
    if video_path == 1:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_path)
    for i in range(5, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break

        # Display the countdown number on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, str(i), (int(frame.shape[1] / 2), int(frame.shape[0] / 2)), font, 4, (0, 0, 255), 2,
                    cv2.LINE_AA)

        # Encode frame as JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Wait for 1 second
        time.sleep(1)
    while True:
        pushup_count = 0
        pushup_count_error = 0
        expected_stages = [0, 1, 0]
        expected_stages_w = [0, 2, 0]
        current_stages = []
        tick = 0
        tick1 = 0
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        landmarks = []
        for val in range(1, 33 + 1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        model = joblib.load('pushup_model.pkl')

        # Load the mean and standard deviation from the files
        mean = np.load('mean4.npy')
        std_dev = np.load('std_dev4.npy')

        # Initialize StandardScaler with the loaded mean and standard deviation
        scaler = StandardScaler()
        scaler.mean_ = mean
        scaler.scale_ = std_dev

        # Initiate holistic model
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make Detection
                results = pose.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                 circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                 circle_radius=2)
                                          )

                try:
                    row = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                    results.pose_landmarks.landmark]).flatten()
                    X = pd.DataFrame([row], columns=landmarks)

                    # Scale the input data
                    Xy = scaler.transform(X)
                    body_language_prob = model.predict_proba(Xy)
                    body_language_class = model.predict(Xy)
                    print(body_language_class)

                    current_stage = -1

                    if body_language_class == 0:
                        current_stage = 0  # Up
                    elif body_language_class == 1:
                        current_stage = 1  # Down
                    elif body_language_class == 2:
                        current_stage = 2  # Ascend incorrect

                    # Add the current stage to the list of stages
                    if current_stage == 0 and tick == 0:
                        current_stages.append(current_stage)
                        tick = 1
                    elif current_stage == 1 and tick == 1:
                        current_stages.append(current_stage)
                        tick = 2
                        tick1 = 0
                    elif current_stage == 2 and tick == 2:
                        current_stages.pop()
                        current_stages.append(current_stage)
                        tick = 3
                        tick1 = 0
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Body must be straight!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to straighten your', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'posture from head to feet to stay aligned.', (72, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 2 and tick1 != 0:
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Body must be straight!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to straighten your', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'posture from head to feet to stay aligned.', (72, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        if tick1 == 1:
                            pushup_count_error += 1
                            if pushup_count != 0:
                                pushup_count -= 1
                        current_stages = []
                        tick = 0
                        tick1 = 0
                    elif current_stage == 2:
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Body must be straight!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to straighten your', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'posture from head to feet to stay aligned.', (72, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                    elif current_stage == 0 and tick == 2:
                        current_stages.append(current_stage)
                        tick = 3
                    elif current_stage == 0 and tick == 3:
                        current_stages.append(current_stage)
                        tick = 4
                        cv2.rectangle(image, (20, 10), (620, 90), (0, 0, 255), -1)
                        cv2.putText(image, 'Body must be straight!', (235, 35), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'When going up, make sure to straighten your', (70, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(image, 'posture from head to feet to stay aligned.', (72, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

                    # Trim the list if it exceeds 3 elements
                    if len(current_stages) > 3:
                        current_stages.pop(0)

                    # Check if the current stages match the expected sequence
                    if current_stages == expected_stages and tick1 == 0:
                        # Increment the squat count and reset the current stages list
                        pushup_count += 1
                        current_stages = []
                        tick = 0
                        tick1 = 1
                    elif current_stages == expected_stages_w and tick1 == 0:
                        pushup_count_error += 1
                        current_stages = []
                        tick = 0
                        tick1 = 2

                    # Display Class
                    class_names = {
                        0: 'up',
                        1: 'down',
                        2: 'ascend incorrect'
                    }

                    # Send data to the front end
                    socketio.emit('update', {
                        'technique': 'Push Up',
                        'class': class_names[np.argmax(body_language_prob)],
                        'classNo': int(body_language_class[0]),
                        'correct': pushup_count,
                        'incorrect': pushup_count_error
                    })

                except Exception as e:
                    print("Error:", e)

                # Encode frame as JPEG image
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
