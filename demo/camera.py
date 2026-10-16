"""Finite camera poses in local meters: X right, Y forward, Z up."""

import math


def target_pose(value):
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("expected [x,y,z,yaw,pitch]")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("camera pose must be finite numbers")
    if max(abs(x) for x in value[:3]) > 100 or abs(value[4]) > math.pi * 0.44:
        raise ValueError("camera pose exceeds demo bounds")
    return [float(x) for x in value]
