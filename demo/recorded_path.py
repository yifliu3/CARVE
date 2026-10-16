"""Exact 24-fps camera samples, split into current 17-frame token blocks."""

import math

from .camera import target_pose
from .settings import CFG

FPS = 24
BLOCK = 17
MAX_FRAMES = 720


def validate_path(value, start):
    if not isinstance(value, dict) or value.get("fps") != FPS:
        raise ValueError("请刷新页面并录制 24 fps 的 proxy 轨迹")
    raw = value.get("poses")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_FRAMES:
        raise ValueError("请先操作 proxy 录制轨迹，每段最多 30 秒")
    poses = [target_pose(p) for p in raw]
    end = target_pose(value.get("end_pose"))
    if any(abs(a - b) > 1e-5 for a, b in zip(poses[0], start)):
        raise ValueError("录制起点与模型当前位置不同，请点击重录后再操作")
    for previous, current in zip(poses, poses[1:] + [end]):
        distance = math.dist(previous[:3], current[:3])
        yaw = (current[3] - previous[3] + math.pi) % (2 * math.pi) - math.pi
        pitch = current[4] - previous[4]
        if distance > CFG["translation_m_per_s"] / FPS + 1e-5:
            raise ValueError("轨迹移动步长异常，请重新录制")
        if max(abs(yaw), abs(pitch)) > math.radians(CFG["rotation_deg_per_s"]) / FPS + 1e-5:
            raise ValueError("轨迹转向步长异常，请重新录制")
    return {"fps": FPS, "poses": poses, "end_pose": end}


def plan_path(path, request_id):
    poses, end = path["poses"], path["end_pose"]
    result = []
    for offset in range(0, len(poses), BLOCK):
        part = poses[offset:offset + BLOCK]
        next_pose = poses[offset + BLOCK] if offset + BLOCK < len(poses) else end
        result.append({
            "poses": part + [end] * (BLOCK - len(part)),
            "next_pose": next_pose,
            "publication": {"request_id": request_id, "frame_start": offset,
                            "visible_frames": len(part), "poses": part, "end_pose": next_pose},
        })
    return result
