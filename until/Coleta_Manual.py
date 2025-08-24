# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""

import cv2
import pandas as pd
import numpy as np

# Botões
# Sair
exit_button = 'q'
# Avançar Frame
next_button = 'w'
# Voltar frame
previous_button = 's'

sheet_path = r"C:\Users\User\Desktop\Trajetórias\Piloto1_Drone1_0014_transformed_rastreio_117.xlsx"
video_path = r"C:\Users\User\Desktop\Trajetórias\Piloto1_Drone1_0014_transformed.avi"

# Ler planilha para coletar os frames
df = pd.read_excel(sheet_path)

# Abrir vídeo
cap = cv2.VideoCapture(video_path)
width  = cap.get(3)
height = cap.get(4)

resolution_resize = 0.6

count_frame = 0

# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, params):
   global frame_num,new_df,sheet_path
   
   # checking for left mouse clicks
   if event == cv2.EVENT_LBUTTONDOWN:
      
      x_corr = int(x/resolution_resize)
      y_corr = int(y/resolution_resize)
      print(frame_num,x_corr,y_corr)
      
      new_df = pd.read_excel(sheet_path)
      new_df.loc[new_df['frame']==frame_num,'x'] = x_corr
      new_df.loc[new_df['frame']==frame_num,'y'] = y_corr
      new_df.to_excel(sheet_path,index=False)

# display the image
while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, df['frame'].iloc[count_frame])
    ret, frame = cap.read()

    frame_num = df["frame"].iloc[count_frame]
    frame = cv2.resize(frame,(int(width*resolution_resize),int(height*resolution_resize)),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    cv2.putText(frame, f'{frame_num}', (10,450), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', click_event)
    
    if cv2.waitKey(1) == ord('q'):
        break
    if cv2.waitKey(0) == ord('w'):
        count_frame = count_frame + 1
    if cv2.waitKey(0) == ord('s'):
        count_frame = count_frame - 1 if count_frame > 0 else 0
cap.release()
cv2.destroyAllWindows()
