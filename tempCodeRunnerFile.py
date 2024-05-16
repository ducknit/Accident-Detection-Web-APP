import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def save_clip(frame, clip_number):
    clip_path = f"C:\\Users\\Nitish Maurya\\Downloads\\Accident-Detection-System-main\\Accident-Detection-System-main\\accident detected\\accident_{clip_number}.avi"
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(clip_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    
    # Save frames for 5 seconds (assuming 30 frames per second)
    for _ in range(150):
        out.write(frame)
    
    out.release()

def startapplication():
    video = cv2.VideoCapture(r"C:\Users\Nitish Maurya\Downloads\Accident-Detection-System-main\Accident-Detection-System-main\Demo2.mp4")
    clip_number = 1
    clips_saved = 0
    
    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if pred == "Accident":
            prob_percent = round(prob[0][0] * 100, 2)

            if prob_percent > 97 and clips_saved < 5:
                save_clip(frame, clip_number)
                clip_number += 1
                clips_saved += 1

                # To beep when alert:
                # os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob_percent}%", (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)

if __name__ == '__main__':
    startapplication()
