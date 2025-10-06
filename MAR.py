import cv2 as cv
import numpy as np
import mediapipe as mp
import pygame
import math
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
mp_drawing=mp.solutions.drawing_utils

lips=[61,291,13,14]
def mar(landmarks,lips,frame_width,frame_height):
    p1=landmarks[lips[0]]
    p2=landmarks[lips[1]]
    p3=landmarks[lips[2]]
    p4=landmarks[lips[3]]
   
    p1=(int(p1.x*frame_width),int(p1.y*frame_height))
    p2=(int(p2.x*frame_width),int(p2.y*frame_height))
    p3=(int(p3.x*frame_width),int(p3.y*frame_height))
    p4=(int(p4.x*frame_width),int(p4.y*frame_height))
   
    vertical_left=math.dist(p3,p4)
   
    horizontal=math.dist(p1,p2)
    if(horizontal==0):
        return 0.0
    mar=(vertical_left)/(horizontal)
    return mar

yawn_count=0
yawn_threshold=0.65
consec_frame_count=0
frame_threshold=10
yawning=False
yawn_completed=True

capture=cv.VideoCapture(0)
while capture.isOpened():
    ret,frame=capture.read()
    if not ret:
        break

    rgbframe=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result=face_mesh.process(rgbframe)
    frame_height, frame_width, _=frame.shape

    if result.multi_face_landmarks:
        lm=result.multi_face_landmarks[0].landmark
        MAR=mar(lm, lips, frame_width, frame_height)

        if MAR>yawn_threshold:
            consec_frame_count+=1
            if consec_frame_count>=frame_threshold and not yawning and yawn_completed:
                cv.putText(frame,"Yawn Detected!!",(100,200),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                yawning=True
                yawn_completed=False
                yawn_count+=1
        else:
            if yawning:
                
                yawn_completed=True
                yawning=False
            consec_frame_count=0
        status=f"MAR: {MAR: .2f} , Yawn count: {yawn_count}"
        if yawning:
            status+="[Yawning]"
        else:
            status+="[Completed]"
        print(status)
        
        cv.putText(frame,f"MAR: {MAR:.2f}",(50,50),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
        cv.putText(frame,f"Yawns: {yawn_count}",(50,80),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
       
    cv.imshow("Yawn Detection",frame)
    if(cv.waitKey(1) & 0xFF==27):
        break
capture.release()
cv.destroyAllWindows()

