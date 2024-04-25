import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=2, min_detection_confidence=minDetectionCon)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                            self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                            self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))
                face = []
                for lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)  # Capturing from webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return
    
    pTime = 0
    detector = FaceMeshDetector()
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("Error: Unable to read frame from webcam.")
            break  # Exit the loop if no frame is read
        
        img, faces = detector.findFaceMesh(img)
        print(faces)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit the loop if 'q' is pressed
    
    # Release the webcam capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
