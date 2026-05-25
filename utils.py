import math

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    a, b, c are tuples of (x, y)
    Angle is at point b
    """
    ax, ay = a
    bx, by = b
    cx, cy = c

    angle = math.degrees(
        math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    )
    angle = abs(angle)
    
    if angle > 180:
        angle = 360 - angle
    return angle