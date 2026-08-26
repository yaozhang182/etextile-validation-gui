import serial
import csv
import time

# Port that Arduino is connected to
port = '/dev/tty.usbmodem141301'
# Name of the file that gets written
file_name = './results/test_sensor.csv'

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

