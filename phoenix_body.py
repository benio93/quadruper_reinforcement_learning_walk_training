# phoenix_body.py
# Simplified body kinematics (ZYX) + Phoenix-like sign convention for R/L legs.

import numpy as np

def body_fk(pos, rot, foot, side):
    """
    pos: {'x','y','z} body translation
    rot: {'x','y','z} body rotation (rad)  yaw=rot['y']
    foot: desired local foot before body motion (x,y,z)
    side: 'R' or 'L'
    Returns transformed foot (x,y,z) to feed into IK.
    """
    cx, cy, cz = np.cos([rot['x'], rot['y'], rot['z']])
    sx, sy, sz = np.sin([rot['x'], rot['y'], rot['z']])

    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    R = Rz @ Ry @ Rx

    v = np.array([foot['x'], foot['y'], foot['z']], dtype=float)
    v = R @ v

    # Phoenix mixes signs differently for R/L sides. This choice works robustly with Z-up.
    if side == 'R':
        x = v[0] - pos['x']
        y = v[1] + pos['y']
        z = v[2] - pos['z']
    else:
        x = v[0] + pos['x']
        y = v[1] + pos['y']
        z = v[2] - pos['z']

    return {"x": float(x), "y": float(y), "z": float(z)}
