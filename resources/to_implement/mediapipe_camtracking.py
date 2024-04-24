import cv2
import mediapipe as mp
import numpy as np

# Initialize video capture for webcam
cap = cv2.VideoCapture(0)

# Initialize face mesh, pose, and selfie segmentation outside the loop
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with face mesh, pose, and selfie segmentation
    face_results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    segmentation_results = selfie_segmentation.process(rgb_frame)

    annotated_frame = frame.copy()
    
    # Draw face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(annotated_frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)  # Corrected this line

    # Draw pose landmarks
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(annotated_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Apply segmentation mask to the frame
    if segmentation_results.segmentation_mask is not None:
        mask = segmentation_results.segmentation_mask
        mask = np.stack((mask,) * 3, axis=-1) > 0.15
        blurred_background = cv2.GaussianBlur(annotated_frame, (35, 35), 0)
        annotated_frame = np.where(mask, annotated_frame, blurred_background)

    # TODO: Calculate and define predicted_class here or remove the following line if not needed
    # Display the predicted gesture expression on the frame
    # cv2.putText(annotated_frame, f"Gesture: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Annotated Frame', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()