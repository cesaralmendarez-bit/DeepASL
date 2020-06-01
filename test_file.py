import cv2

cap = cv2.VideoCapture(0)

while(True):

    inputter = input("Enter")

    ret, frame = cap.read()

    frame[0 : 800, 0 : 1300] = (0, 0, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame,
                inputter,
                (50, 50),
                font, 1,
                (255, 255, 255),
                2,
                cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
