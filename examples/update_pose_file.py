"""Example atomic FK pose-file writer; integrate function write_pose with your robot's FK.

Call write_pose(np.array(T_base_tool0)) every ~50ms while online calibration is running.
Do not use the illustrative identity matrix for real calibration!
"""
import json
import os
from pathlib import Path
import time
import numpy as np


def write_pose(T_base_gripper, filename='examples/robot_pose.json'):
    dest = Path(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.json.tmp')
    payload = {'T_base_gripper': np.asarray(T_base_gripper, float).reshape(4, 4).tolist(),
               'stamp_unix_s': time.time()}
    temp.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(temp, dest)


if __name__ == '__main__':
    # Demonstration: update function, not a live robot FK provider.
    write_pose(np.eye(4))
    print('Wrote a DEMO identity transform. Replace with your real T_base_gripper before collecting.')
