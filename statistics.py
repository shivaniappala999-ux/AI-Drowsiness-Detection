import csv
import os
from collections import Counter
import matplotlib.pyplot as plt


# ---------------- LOG FILE ----------------

LOG_FILE = os.path.join(
    "logs",
    "drowsiness_log.csv"
)


# ---------------- CHECK FILE ----------------

if not os.path.exists(LOG_FILE):

    print("Log file not found!")
    print("Run app.py first and generate some events.")
    exit()


# ---------------- READ CSV ----------------

events = []

with open(
    LOG_FILE,
    "r",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        events.append(row)


# ---------------- CHECK DATA ----------------

if not events:

    print("No events found in the log file.")
    exit()


# ---------------- COUNT EVENTS ----------------

event_counter = Counter(
    row["Event"]
    for row in events
)

drowsiness_count = event_counter.get(
    "Drowsiness",
    0
)

yawn_count = event_counter.get(
    "Yawn",
    0
)


# ---------------- DISPLAY STATISTICS ----------------

print("\n================================")
print("       DROWSINESS STATISTICS")
print("================================")

print(
    f"Total Events      : {len(events)}"
)

print(
    f"Drowsiness Events : {drowsiness_count}"
)

print(
    f"Yawn Events       : {yawn_count}"
)

print("================================")


# ---------------- GRAPH ----------------

labels = [
    "Drowsiness",
    "Yawns"
]

values = [
    drowsiness_count,
    yawn_count
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    labels,
    values
)

plt.title(
    "Drowsiness Detection Statistics"
)

plt.xlabel(
    "Event Type"
)

plt.ylabel(
    "Number of Events"
)

plt.tight_layout()

plt.show()