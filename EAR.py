import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import pygame
import math

pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing=mp.solutions.drawing_utils

left_eye=[33, 160, 158, 133, 153, 144] 
right_eye=[362, 385, 387, 263, 373, 380]
def eye_aspect(landmarks,eyePoint,frame_width,frame_height):
    p1=landmarks[eyePoint[0]]
    p2=landmarks[eyePoint[1]]
    p3=landmarks[eyePoint[2]]
    p4=landmarks[eyePoint[3]]
    p5=landmarks[eyePoint[4]]
    p6=landmarks[eyePoint[5]]

    p1=(int(p1.x*frame_width),int(p1.y*frame_height))
    p2=(int(p2.x*frame_width), int(p2.y*frame_height))
    p3=(int(p3.x*frame_width), int(p3.y*frame_height))
    p4=(int(p4.x*frame_width), int(p4.y*frame_height))
    p5=(int(p5.x*frame_width), int(p5.y*frame_height))
    p6=(int(p6.x*frame_width), int(p6.y*frame_height))

    vertical_left=math.dist(p2,p6)
    vertical_right=math.dist(p3,p5)
    horizontal=math.dist(p1,p4)
    ear=(vertical_left+vertical_right)/(2.0*horizontal)
    return ear
blink_count=0
count_consec_frame=0
blink_threshold=.21
consec_frame_threshold=2
eye_closed=False

capture=cv.VideoCapture(0)

while capture.isOpened():
    ret,frame=capture.read()
    if not ret:
        break

    rgbFrame= cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result=face_mesh.process(rgbFrame)
    frame_height,frame_width,_=frame.shape #since frame.shape return 3 value whick are height,width and the 3 channels

    if result.multi_face_landmarks:
        lm=result.multi_face_landmarks[0].landmark

        LEFTEAR= eye_aspect(lm,left_eye,frame_width,frame_height)
        RIGHTEAR=eye_aspect(lm,right_eye,frame_width,frame_height)
        EAR=(LEFTEAR+RIGHTEAR)/2.0

        if EAR < blink_threshold:
            count_consec_frame+=1
            if count_consec_frame >=45 and not eye_closed:
                cv.putText(frame, "Drowsy Alert!!!", (100, 200), cv.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play(-1)
                
        else:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            if not eye_closed and count_consec_frame>=consec_frame_threshold:
                blink_count+=1
                eye_closed=True
            count_consec_frame=0
            eye_closed=False

    cv.putText(frame,f"EAR: {EAR:.2f}",(70,70),cv.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
    cv.putText(frame,f"Blink Count: {blink_count}", (70,100), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
    cv.putText(frame,f"Eye Staus : {'Closed' if EAR< blink_threshold else 'Open'}", (70,130), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)

    print(f"ear:{EAR:.2f}, Blink:{blink_count}, closed_frames_count:{count_consec_frame}")

    cv.imshow("Drowsy Detection",frame)
    if(cv.waitKey(1) & 0xFF==27):
        break
capture.release()
cv.destroyAllWindows()

