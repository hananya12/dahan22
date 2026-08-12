import cv2
import json
import numpy as np

from camera import CameraManager


points = []

zone_name = "TAM"


print("[COREWISE] Zone setup")
print("Click 4 points around the area")
print("Press Q to save")


camera = CameraManager()


try:
    camera.connect()

except Exception as e:
    print("[ERROR] Camera:", e)
    exit()



def mouse_callback(event, x, y, flags, param):

    global points

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            points.append([x, y])

            print(f"[ZONE] Point selected: {x}, {y}")



cv2.namedWindow("Zone Setup")

cv2.setMouseCallback(
    "Zone Setup",
    mouse_callback
)



while True:

    frame = camera.read_frame()


    if frame is None:
        continue


    display = frame.copy()



    # Draw points

    for p in points:

        cv2.circle(
            display,
            tuple(p),
            6,
            (0,0,255),
            -1
        )



    # Draw polygon while selecting

    if len(points) > 1:

        cv2.polylines(
            display,
            [np.array(points, dtype=np.int32)],
            False,
            (255,0,0),
            2
        )



    # Completed zone

    if len(points) == 4:

        cv2.polylines(
            display,
            [np.array(points, dtype=np.int32)],
            True,
            (0,255,0),
            3
        )



    cv2.imshow(
        "Zone Setup",
        display
    )


    key = cv2.waitKey(1)


    if key == ord("q"):

        break



camera.release()

cv2.destroyAllWindows()



if len(points) == 4:


    data = {

        zone_name: {

            "points": points

        }

    }



    with open(
        "zones.json",
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )


    print("[ZONE] Saved successfully!")

else:

    print("[ERROR] Need exactly 4 points")