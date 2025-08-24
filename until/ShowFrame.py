import cv2

cap = cv2.VideoCapture(r"C:\Users\User\Desktop\10_A_1_transformed_rastreio.avi")

frame_number = int((13*60+9.9)*30)
print(frame_number)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
while ret:
    resized_frame = cv2.resize(frame, (int(1920*0.5), int(1080*0.5)))
    cv2.imshow('Frame', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


