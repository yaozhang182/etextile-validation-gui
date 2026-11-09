import serial
import csv
import os
import sys
import time

# Check this computer's clock against time.google.com before recording
# (clock_check.py, one folder up). Only measures and warns; never blocks.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from clock_check import check_clock
except ImportError:
    def check_clock():
        print("[Clock] clock_check.py not found; synchronise the clock by hand "
              "before recording (see docs/manual.md).")

# Port that Arduino is connected to
port = '/dev/tty.usbmodem141301'
# Name of the file that gets written
file_name = './results/test_sensor.csv'

check_clock()
ser = serial.Serial(port, 115200)

with open(file_name, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=';')
    # 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'
    # start file with column names (added Timestamp column)
    writer.writerow(['EpochTime', 'S1', 'S2', 'S3', 'S4'])

    while True:
        line = ser.readline().decode().strip()
        if line:
            row = line.split(",")
            row = row[:-1]
            # prepend epoch time (float, high precision)
            epoch_time = time.time()
            writer.writerow([epoch_time] + row)
            print([epoch_time] + row)

