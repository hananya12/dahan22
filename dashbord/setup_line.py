import cv2
import json

points = []


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"[LINE] Point selected: {x}, {y}")


        if len(points) == 2:
            save_line()


def save_line():
    data = {
        "start": points[0],
        "end": points[1]
    }

    with open("line_config.json", "w") as f:
        json.dump(data, f, indent=4)

    print("[LINE] Saved successfully!")
    

def main():

    print("[COREWISE] Starting line setup")

    # Iriun בדרך כלל נמצא אצלך על 1
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Camera not found")
        return


    cv2.namedWindow("Corewise Line Setup")

    cv2.setMouseCallback(
        "Corewise Line Setup",
        mouse_callback
    )


    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # מציג נקודות שנבחרו
        for p in points:
            cv2.circle(
                frame,
                p,
                6,
                (0,255,0),
                -1
            )


        # מציג קו אחרי 2 לחיצות
        if len(points) == 2:
            cv2.line(
                frame,
                points[0],
                points[1],
                (0,0,255),
                3
            )


        cv2.putText(
            frame,
            "Click 2 points for entrance line",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )


        cv2.imshow(
            "Corewise Line Setup",
            frame
        )


        key = cv2.waitKey(1)

        if key == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()