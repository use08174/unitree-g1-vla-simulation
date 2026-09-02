from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

class BaseG1Env:
    def __init__(self, xml_path, instruction, max_episode_steps=5000):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / xml_path))
        self.data = mujoco.MjData(self.model)
        self.instruction = instruction
        self.max_episode_steps = max_episode_steps
        self.np_random = np.random.default_rng()
        self.step_count = 0

        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.default_ctrl = np.zeros(self.model.nu)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.default_ctrl = np.zeros(self.model.nu)
        self.step_count = 0
        self._reset_task(options or {})
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _reset_task(self, options):
        del options

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera="head_camera")
        return {
            "instruction": self.instruction,
            "rgb_image": self.renderer.render(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape {(self.model.nu,)}, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("Action contains non-finite values")
        limited = self.model.actuator_ctrllimited.astype(bool)
        action = action.copy()
        action[limited] = np.clip(
            action[limited],
            self.model.actuator_ctrlrange[limited, 0],
            self.model.actuator_ctrlrange[limited, 1],
        )
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        metrics = self._task_metrics()
        success = self._check_success(metrics)
        time_limit_reached = self.step_count >= self.max_episode_steps
        info = {
            "success": success,
            "time_limit_reached": time_limit_reached,
            "metrics": metrics,
        }
        return obs, self._compute_reward(metrics), success or time_limit_reached, info

    def _task_metrics(self):
        return {}

    def _compute_reward(self, metrics):
        del metrics
        return 0.0

    def _check_success(self, metrics):
        del metrics
        return False

    def body_position(self, body_name):
        return self.data.body(body_name).xpos.copy()

    def body_speed(self, body_name):
        body_id = self.model.body(body_name).id
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return float(np.linalg.norm(velocity[3:]))

    def bodies_in_contact(self, first_body_names, second_body_names):
        first_ids = {self.model.body(name).id for name in first_body_names}
        second_ids = {self.model.body(name).id for name in second_body_names}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            first = self.model.geom_bodyid[contact.geom1]
            second = self.model.geom_bodyid[contact.geom2]
            if (first in first_ids and second in second_ids) or (
                first in second_ids and second in first_ids
            ):
                return True
        return False

    def close(self):
        self.renderer.close()