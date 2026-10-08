import cv2
import time
import winsound

from src.drowsiness_detector import DrowsinessDetector


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.45
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Webcam started.")
# Eyes must remain closed this long to be considered drowsy
DROWSY_TIME = 2.0

# Mouth must remain open this long to count as a yawn
YAWN_TIME = 2.0

# Alarm settings
ALARM_FREQUENCY = 1000
ALARM_DURATION = 500


# ============================================================
# FACE LANDMARK POINTS
# ============================================================

LEFT_EYE = [
    362, 385, 387, 263, 373, 380
]

RIGHT_EYE = [
    33, 160, 158, 133, 153, 144
]

MOUTH_POINTS = [
    61, 81, 13, 291, 311, 14
]


# ============================================================
# START DETECTOR
# ============================================================

detector = DrowsinessDetector()


# ============================================================
# START CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


if not camera.isOpened():

    print("Error: Could not open webcam")
    exit()


print("====================================")
print(" AI DROWSINESS DETECTION SYSTEM")
print("====================================")
print("Webcam started.")
print("Press Q to quit.")


# ============================================================
# VARIABLES
# ============================================================

session_start = time.time()

eyes_closed_start = None
yawn_start = None

drowsy_count = 0
yawn_count = 0

was_drowsy = False
was_yawning = False

last_alarm_time = 0


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------------

    ret, frame = camera.read()

    if not ret:

        print("Error: Could not read frame")
        break


    current_time = time.time()

    timestamp_ms = int(
        (current_time - session_start) * 1000
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_landmarks = detector.detect_face(
        frame,
        timestamp_ms
    )


    # --------------------------------------------------------
    # DEFAULT VALUES
    # --------------------------------------------------------

    status = "NO FACE"

    ear = 0.0
    mar = 0.0

    closed_seconds = 0.0
    yawn_seconds = 0.0

    drowsiness_score = 0


    # ========================================================
    # FACE FOUND
    # ========================================================

    if face_landmarks is not None:

        # ----------------------------------------------------
        # CALCULATE EAR
        # ----------------------------------------------------

        left_ear = detector.calculate_ear(
            face_landmarks,
            LEFT_EYE
        )

        right_ear = detector.calculate_ear(
            face_landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear + right_ear
        ) / 2.0


        # ----------------------------------------------------
        # CALCULATE MAR
        # ----------------------------------------------------

        mar = detector.calculate_mar(
            face_landmarks,
            MOUTH_POINTS
        )

        if mar is None:

            mar = 0.0


        # ----------------------------------------------------
        # DRAW FACE LANDMARKS
        # ----------------------------------------------------

        height, width, _ = frame.shape

        for landmark in face_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                1,
                (255, 255, 0),
                -1
            )


        # ====================================================
        # DROWSINESS DETECTION
        # ====================================================

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

            else:

                status = "EYES CLOSED"


        else:

            status = "AWAKE"

            eyes_closed_start = None

            was_drowsy = False

            closed_seconds = 0.0


        # ====================================================
        # YAWNING DETECTION
        # ====================================================

        # ---------------- YAWN DETECTION ----------------

if mar > MAR_THRESHOLD:

    if yawn_start is None:
        yawn_start = current_time

    yawn_seconds = current_time - yawn_start

    # Yawn must remain for at least 1.5 seconds
    if yawn_seconds >= 1.5:

        if not was_yawning:
            yawn_count += 1
            was_yawning = True

        cv2.putText(
            frame,
            "YAWNING DETECTED",
            (30, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            3
        )

else:

    yawn_start = None
    was_yawning = False
    yawn_seconds = 0.0

        # ====================================================
        # DROWSINESS SCORE
        # ====================================================

    drowsiness_score = 0


        # Eye closure contribution
    if closed_seconds >= 1:

            drowsiness_score += 30


    if closed_seconds >= 2:

            drowsiness_score += 30


    if closed_seconds >= 3:

            drowsiness_score += 20


        # Low EAR contribution
    if ear < EAR_THRESHOLD:

            drowsiness_score += 20


        # Yawning contribution
    if mar > MAR_THRESHOLD:

            drowsiness_score += 20


        # Maximum score = 100
    drowsiness_score = min(
            drowsiness_score,
            100
        )


        # ====================================================
        # SMART STATUS
        # ====================================================

    if status == "DROWSY!":

            drowsiness_score = max(
                drowsiness_score,
                80
            )


        # ====================================================
        # ALARM
        # ====================================================

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


    # ========================================================
    # MONITORING TIME
    # ========================================================

    session_seconds = (
        current_time -
        session_start
    )

    minutes = int(
        session_seconds // 60
    )

    seconds = int(
        session_seconds % 60
    )

    session_time = (
        f"{minutes:02d}:{seconds:02d}"
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "AI DROWSINESS DETECTION",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # STATUS COLOR
    # ========================================================

    if status == "AWAKE":

        status_color = (0, 255, 0)

    elif status == "EYES CLOSED":

        status_color = (0, 165, 255)

    elif status == "DROWSY!":

        status_color = (0, 0, 255)

    else:

        status_color = (255, 255, 255)


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        frame,
        f"Status: {status}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )


    # ========================================================
    # EAR
    # ========================================================

    cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (30, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MAR
    # ========================================================

    cv2.putText(
        frame,
        f"MAR: {mar:.2f}",
        (30, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # EYES CLOSED
    # ========================================================

    cv2.putText(
        frame,
        f"Eyes Closed: {closed_seconds:.1f}s",
        (30, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DROWSINESS SCORE
    # ========================================================

    cv2.putText(
        frame,
        f"Drowsiness Score: {drowsiness_score}%",
        (30, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DROWSY EVENTS
    # ========================================================

    cv2.putText(
        frame,
        f"Drowsy Events: {drowsy_count}",
        (30, 265),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # YAWNS
    # ========================================================

    cv2.putText(
        frame,
        f"Yawns: {yawn_count}",
        (30, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # MONITORING TIME
    # ========================================================

    cv2.putText(
        frame,
        f"Monitoring Time: {session_time}",
        (30, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # WARNING
    # ========================================================

    if status == "DROWSY!":

        cv2.putText(
            frame,
            "WARNING! WAKE UP!",
            (30, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    # ========================================================
    # SHOW CAMERA
    # ========================================================
while True:
    cv2.imshow(
        "AI Drowsiness Detection",
        frame
    )


    # ========================================================
    # PRESS Q TO EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("====================================")
print(" SESSION COMPLETED")
print("====================================")

print(
    f"Monitoring Time : {session_time}"
)

print(
    f"Drowsy Events   : {drowsy_count}"
)

print(
    f"Yawns           : {yawn_count}"
)

print("System stopped.")