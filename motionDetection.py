import cv2
import numpy as np

cap = cv2.VideoCapture(0)
previous_frame = None

while True:
    flag, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.GaussianBlur(frameGray, (15, 15), 0)

    if previous_frame is None:
        previous_frame = frameGray

    diff_frame = cv2.absdiff(src1=previous_frame, src2=frameGray)
    previous_frame = frameGray

    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    frame_contoured = frame.copy()
    thresh_frame = cv2.threshold(src=diff_frame, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=frame_contoured, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=2,
                     lineType=cv2.LINE_AA)

    mask = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_TOZERO)[1]
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    movementRed = frame + mask

    # размытие
    # cv2.imshow("blur", frameGray)
    # только движения
    # cv2.imshow("diff_frame",  diff_frame)
    # контуры движения
    # cv2.imshow("frame_contoured", frame_contoured)
    # Движения подсвечены красным
    cv2.imshow("movementRed", movementRed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
