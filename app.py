
import cv2
import time
import winsound
import csv
import os
import requests
from datetime import datetime

from src.drowsiness_detector import DrowsinessDetector


# ==================================================
# SETTINGS
# ==================================================

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.45

DROWSY_TIME = 2.0
YAWN_TIME = 1.5

ALARM_FREQUENCY = 1000
ALARM_DURATION = 500

BACKEND_URL = "http://127.0.0.1:8000/api/status"


# ==================================================
# LANDMARK POINTS
# ==================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

MOUTH_POINTS = [61, 81, 13, 291, 311, 14]


# ==================================================
# DETECTOR
# ==================================================

detector = DrowsinessDetector()


# ==================================================
# LOGGING
# ==================================================

LOG_FOLDER = "logs"
LOG_FILE = os.path.join(
    LOG_FOLDER,
    "drowsiness_log.csv"
)

os.makedirs(LOG_FOLDER, exist_ok=True)


if not os.path.exists(LOG_FILE):

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Event",
            "EAR",
            "MAR",
            "Drowsiness Score"
        ])


def log_event(event, ear, mar, score):

    now = datetime.now()

    with open(
        LOG_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            event,
            f"{ear:.2f}",
            f"{mar:.2f}",
            score
        ])


# ==================================================
# CAMERA
# ==================================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


# ==================================================
# VARIABLES
# ==================================================

session_start = time.time()

eyes_closed_start = None
yawn_start = None

drowsy_count = 0
yawn_count = 0

was_drowsy = False
was_yawning = False

last_alarm_time = 0


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    # --------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Could not read frame.")
        break


    # --------------------------------------------------
    # CHECK BACKEND MONITORING STATUS
    # --------------------------------------------------

    MONITORING = True

    try:

        response = requests.get(
            BACKEND_URL,
            timeout=0.1
        )

        backend_data = response.json()

        MONITORING = backend_data.get(
            "monitoring",
            True
        )

    except requests.exceptions.RequestException:

        MONITORING = True


    # --------------------------------------------------
    # CURRENT TIME
    # --------------------------------------------------

    current_time = time.time()

    timestamp_ms = int(
        current_time * 1000
    )


    # --------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------

    status = "NO FACE"

    ear = 0.0
    mar = 0.0

    closed_seconds = 0.0
    yawn_seconds = 0.0

    drowsiness_score = 0


    # --------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------

    face_landmarks = detector.detect_face(
        frame,
        timestamp_ms
    )


    # ==================================================
    # FACE DETECTED
    # ==================================================

    if face_landmarks:

        # --------------------------------------------------
        # EAR
        # --------------------------------------------------

        left_ear = detector.calculate_ear(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = detector.calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear +
            right_ear
        ) / 2


        # --------------------------------------------------
        # MAR
        # --------------------------------------------------

        mar = detector.calculate_mar(
            face_landmarks,
            MOUTH_POINTS
        )


        # ==================================================
        # DROWSINESS DETECTION
        # ==================================================

        if ear < EAR_THRESHOLD:

            if eyes_closed_start is None:

                eyes_closed_start = current_time

            closed_seconds = (
                current_time -
                eyes_closed_start
            )


            if closed_seconds >= DROWSY_TIME:

                status = "DROWSY!"

                if not was_drowsy:

                    drowsy_count += 1

                    was_drowsy = True

                    log_event(
                        "Drowsiness",
                        ear,
                        mar,
                        0
                    )

            else:

                status = "EYES CLOSED"

        else:

            status = "AWAKE"

            eyes_closed_start = None

            was_drowsy = False

            closed_seconds = 0.0


        # ==================================================
        # YAWN DETECTION
        # ==================================================

        if mar > MAR_THRESHOLD:

            if yawn_start is None:

                yawn_start = current_time

            yawn_seconds = (
                current_time -
                yawn_start
            )


            if yawn_seconds >= YAWN_TIME:

                if not was_yawning:

                    yawn_count += 1

                    was_yawning = True

                    log_event(
                        "Yawn",
                        ear,
                        mar,
                        0
                    )

        else:

            yawn_start = None

            was_yawning = False

            yawn_seconds = 0.0


        # ==================================================
        # DROWSINESS SCORE
        # ==================================================

        if closed_seconds >= 1:

            drowsiness_score += 30


        if closed_seconds >= 2:

            drowsiness_score += 30


        if closed_seconds >= 3:

            drowsiness_score += 20


        if ear < EAR_THRESHOLD:

            drowsiness_score += 20


        if mar > MAR_THRESHOLD:

            drowsiness_score += 20


        drowsiness_score = min(
            drowsiness_score,
            100
        )


        if status == "DROWSY!":

            drowsiness_score = max(
                drowsiness_score,
                80
            )


        # ==================================================
        # ALARM
        # ==================================================

        if status == "DROWSY!":

            if (
                current_time -
                last_alarm_time
                >= 1
            ):

                winsound.Beep(
                    ALARM_FREQUENCY,
                    ALARM_DURATION
                )

                last_alarm_time = current_time


        # ==================================================
        # DRAW FACE LANDMARKS
        # ==================================================

        h, w, _ = frame.shape

        for landmark in face_landmarks:

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )


    # ==================================================
    # MONITORING TIME
    # ==================================================

    monitoring_time = int(
        current_time -
        session_start
    )

    minutes = monitoring_time // 60

    seconds = monitoring_time % 60


    # ==================================================
    # DISPLAY STATUS
    # ==================================================

    if not MONITORING:

        cv2.putText(
            frame,
            "MONITORING STOPPED",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )


    # ==================================================
    # DISPLAY INFORMATION
    # ==================================================

    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Eyes Closed: {closed_seconds:.1f} s",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Drowsiness Score: {drowsiness_score}%",
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Drowsy Events: {drowsy_count}",
        (30, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Yawns: {yawn_count}",
        (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Monitoring Time: {minutes:02d}:{seconds:02d}",
        (30, 275),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ==================================================
    # YAWNING WARNING
    # ==================================================

    if yawn_seconds >= YAWN_TIME:

        cv2.putText(
            frame,
            "YAWNING DETECTED",
            (30, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            3
        )


    # ==================================================
    # DROWSINESS WARNING
    # ==================================================

    if status == "DROWSY!":

        cv2.putText(
            frame,
            "WARNING! WAKE UP!",
            (30, 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


    # ==================================================
    # SEND LIVE DATA TO BACKEND
    # ==================================================

    try:

        requests.post(
            BACKEND_URL,
            json={
                "status": status,
                "ear": ear,
                "mar": mar,
                "drowsiness_score": drowsiness_score,
                "drowsy_events": drowsy_count,
                "yawns": yawn_count
            },
            timeout=0.1
        )

    except requests.exceptions.RequestException:

        pass


    # ==================================================
    # SHOW WEBCAM
    # ==================================================

    cv2.imshow(
        "AI Drowsiness Detection",
        frame
    )


    # ==================================================
    # QUIT
    # ==================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==================================================
# CLEANUP
# ==================================================

camera.release()

cv2.destroyAllWindows()


# ==================================================
# SESSION SUMMARY
# ==================================================

monitoring_duration = int(
    time.time() -
    session_start
)

summary_minutes = (
    monitoring_duration // 60
)

summary_seconds = (
    monitoring_duration % 60
)


print("\n==============================")
print("      SESSION SUMMARY")
print("==============================")


print(
    f"Monitoring Time: "
    f"{summary_minutes:02d}:"
    f"{summary_seconds:02d}"
)


print(
    f"Drowsy Events: "
    f"{drowsy_count}"
)


print(
    f"Yawns: "
    f"{yawn_count}"
)


print(
    f"Log File: "
    f"{LOG_FILE}"
)


print("==============================")
print("Program stopped.")
