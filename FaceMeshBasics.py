import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("C:/Users/Arthur/Documents/FaceDetection/FacialDetection/Videos/3.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

#Conexões faciais definidas manualmente
facial_connections = [[10, 234], [234, 127], [127, 162], [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
                      [10, 152], [152, 234], [152, 454], [454, 234], [234, 435], [10, 42], [42, 15], [15, 1], [1, 8], [8, 234],
                      [10, 284], [284, 332], [332, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356], [356, 454],
                      [454, 323], [323, 361], [361, 454], [10, 468], [468, 417], [417, 478], [478, 417], [417, 324], [324, 361],
                      [361, 454], [454, 323], [323, 361], [361, 454]]

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Desenhe os pontos de referência faciais
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)  # Desenhe um círculo nos pontos de referência

            # Desenhe as conexões faciais manualmente
            # for connection in facial_connections:
            #     if connection[0] < len(faceLms.landmark) and connection[1] < len(faceLms.landmark):
            #         start_point = (int(faceLms.landmark[connection[0]].x * iw), int(faceLms.landmark[connection[0]].y * ih))
            #         end_point = (int(faceLms.landmark[connection[1]].x * iw), int(faceLms.landmark[connection[1]].y * ih))
            #         cv2.line(img, start_point, end_point, (0, 255, 0), 1)  # Desenhe uma linha entre os pontos de referência
            #     else:
            #         print("Índices de pontos de referência fora dos limites!")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

