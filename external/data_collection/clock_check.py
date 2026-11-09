#!/usr/bin/env python3
"""
How far is this computer's clock from Google's time servers?

The toolchain pairs video frames with sensor readings by their Unix timestamps
(time.time()). When the video and the sensor are recorded on two computers, any
difference between the two clocks shifts every sensor reading against the video
by that amount. Run this on each computer before recording, or let the recording
scripts run it for you at start-up (they call check_clock()).

    python clock_check.py                 # query time.google.com
    python clock_check.py pool.ntp.org    # another server

Both computers must be synchronised against the SAME time service. Google's
servers smear leap seconds over a day, other servers step them, so mixing the
two can disagree by up to half a second around a leap second.

Only the standard library is used. Nothing on the computer is changed: this only
measures the offset. To actually correct the clock, see "Synchronise the clocks
before recording" in docs/manual.md.
"""

import socket
import struct
import sys
import time

NTP_SERVER = 'time.google.com'
WARN_SECONDS = 0.1          # above this, the alignment error is worth fixing
_NTP_EPOCH = 2208988800     # seconds from 1900-01-01 (NTP) to 1970-01-01 (Unix)


def _ntp_time(data, offset):
    seconds, fraction = struct.unpack('!II', data[offset:offset + 8])
    return seconds - _NTP_EPOCH + fraction / 2 ** 32


def ntp_offset(server=NTP_SERVER, samples=5, timeout=2.0, port=123):
    """
    Return (offset, delay) in seconds, using the sample with the shortest round
    trip. offset > 0 means this computer's clock is BEHIND the server.
    Raises OSError if the server cannot be reached.
    """
    best = None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        address = (socket.gethostbyname(server), port)
        for _ in range(samples):
            request = b'\x23' + 47 * b'\0'          # NTP v4, client mode
            t1 = time.time()
            sock.sendto(request, address)
            data, _ = sock.recvfrom(48)
            t4 = time.time()
            t2, t3 = _ntp_time(data, 32), _ntp_time(data, 40)
            offset = ((t2 - t1) + (t3 - t4)) / 2
            delay = (t4 - t1) - (t3 - t2)
            if best is None or delay < best[1]:
                best = (offset, delay)
            time.sleep(0.05)
    return best


def check_clock(server=NTP_SERVER, warn=WARN_SECONDS, log_file='clock_check.log'):
    """
    Measure and report the clock offset; never stops the recording.

    Prints the result, warns if it is larger than `warn` seconds, and appends a
    line to `log_file` (in the current folder) so the offset of each recording
    can be looked up later. Returns the offset in seconds, or None if the
    server could not be reached.
    """
    try:
        offset, delay = ntp_offset(server)
    except (OSError, struct.error, ValueError) as e:
        print(f"[Clock] Could not reach {server} ({e}). Recording anyway; "
              f"make sure this computer's clock was synchronised recently.")
        return None

    ahead = 'behind' if offset > 0 else 'ahead of'
    print(f"[Clock] This computer is {abs(offset) * 1000:.0f} ms {ahead} "
          f"{server} (round trip {delay * 1000:.0f} ms).")
    if abs(offset) > warn:
        print(f"[Clock] WARNING: more than {warn * 1000:.0f} ms. Synchronise the "
              f"clock before recording (docs/manual.md, 'Synchronise the clocks "
              f"before recording'), otherwise the sensor data will be shifted "
              f"against the video by this amount.")
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{time.time():.3f}\t"
                        f"{server}\toffset_s={offset:+.4f}\tdelay_s={delay:.4f}\n")
        except OSError:
            pass
    return offset


if __name__ == '__main__':
    check_clock(sys.argv[1] if len(sys.argv) > 1 else NTP_SERVER, log_file=None)
