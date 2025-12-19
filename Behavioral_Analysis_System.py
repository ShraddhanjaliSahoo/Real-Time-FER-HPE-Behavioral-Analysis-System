import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# Load emotion recognition model
model = load_model('model_file_64.h5')
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Load face detector for FER
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize MediaPipe for head pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    start = time.time()

    # FER: Face Detection and Emotion Prediction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_resized = cv2.resize(roi_gray, (64, 64))
        roi_normalized = roi_gray_resized / 255.0
        reshaped = np.reshape(roi_normalized, (1, 64, 64, 1))
        result = model.predict(reshaped)
        emotion_label = np.argmax(result[0])
        emotion_text = labels_dict[emotion_label]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    # HPE: Head Pose Estimation using MediaPipe
    #image_rgb = cv2.cvtColor(cv2.flip(frame.copy(), 1), cv2.COLOR_BGR2RGB)
    image_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:  # landmarks needed for head pose
                    x = int(lm.x * img_w)
                    y = int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z * 3000)

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4,1))

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues (rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3 (rmat)

            pitch, yaw, roll = [angle * 360 for angle in angles]

            # Interpret direction
            if yaw < -10:
                direction = "Looking Left"
            elif yaw > 10:
                direction = "Looking Right"
            elif pitch < -10:
                direction = "Looking Down"
            elif pitch > 10:
                direction = "Looking Up"
            else:
                direction = "Forward"

            # Project direction
            nose_end_point, _ = cv2.projectPoints(np.array([(nose_3d)]), rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + yaw * 10), int(nose_2d[1] - pitch * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Display data
            cv2.putText(image, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Pitch: {round(pitch, 2)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(image, f"Yaw: {round(yaw, 2)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(image, f"Roll: {round(roll, 2)}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    end = time.time()
    fps = 1 / (end - start) if (end - start) > 0 else 0
    cv2.putText(image, f"FPS: {int(fps)}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("FER + Head Pose", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
