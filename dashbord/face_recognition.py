import cv2
import os
import numpy as np

from effects import resolve_cascade
try:
    import face_recognition as fr
except ImportError:
    print("[ERROR] face_recognition לא מותקן!")
    print("הרץ: pip install face_recognition")
    exit(1)


class FaceIdentifier:
    def __init__(self):
        # Same root cause as effects.py: the cascade was loaded ONLY from a
        # hardcoded local path. When absent this RAISED, which disabled the
        # whole FaceIdentifier. Fall back to the cascade bundled with OpenCV.
        cascade_path = resolve_cascade("haarcascade_frontalface_default.xml")

        self.face_cascade = cv2.CascadeClassifier(cascade_path) if cascade_path \
            else cv2.CascadeClassifier()
        self.model_loaded = not self.face_cascade.empty()

        self.known_face_encodings = []
        self.known_face_names = []

        if self.model_loaded:
            print(f"[FACE] Haar Cascade loaded from: {cascade_path}")
        else:
            raise RuntimeError("[FACE ERROR] Could not load Haar Cascade model")


    def detect_faces(self, frame):
        if not self.model_loaded:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return list(faces)


    def register(self, name, camera, num_samples=30):
        print(f"[FACE] Starting registration for {name}...")
        print(f"[FACE] Collecting {num_samples} samples...")

        collected = 0
        
        while collected < num_samples:
            frame = camera.read_frame()
            if frame is None:
                continue

            faces = self.detect_faces(frame)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                
                try:
                    face_encoding = fr.face_encodings(face_crop)
                    if len(face_encoding) > 0:
                        self.known_face_encodings.append(face_encoding[0])
                        collected += 1
                        print(f"[FACE] Collected {collected}/{num_samples}")
                except Exception as e:
                    print(f"[FACE] Error encoding face: {e}")
            
            cv2.putText(
                frame,
                f"Registering {name}: {collected}/{num_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("Registration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if collected > 0:
            self.known_face_names.append(name)
            print(f"[FACE] Registration complete for {name}")
        else:
            print(f"[FACE] Failed to collect samples for {name}")

        cv2.destroyAllWindows()


    def train(self):
        print(f"[FACE] Training complete. {len(self.known_face_names)} people registered.")


    def identify(self, frame, box):
        if len(self.known_face_encodings) == 0:
            return "Unknown", 0.0

        x1, y1, x2, y2 = box
        
        # בדוק שה-coordinates תקינים
        if y2 <= y1 or x2 <= x1 or y1 < 0 or x1 < 0:
            return "Unknown", 0.0
        
        face_crop = frame[y1:y2, x1:x2]

        try:
            face_encodings = fr.face_encodings(face_crop)
            if len(face_encodings) == 0:
                return "Unknown", 0.0

            face_encoding = face_encodings[0]
            
            distances = fr.face_distance(
                self.known_face_encodings,
                face_encoding
            )
            
            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                confidence = 1 - distances[best_match_index]
                
                if confidence > 0.6:
                    name = self.known_face_names[best_match_index]
                    return name, confidence
            
            return "Unknown", 0.0
            
        except Exception as e:
            return "Unknown", 0.0