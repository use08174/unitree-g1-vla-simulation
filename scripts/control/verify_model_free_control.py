"""Run a scripted, model-free G1 arm motion and report physical interaction."""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from envs.tasks import G1Task1Env


ARM_ACTUATORS = (
    "waist_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
RIGHT_HAND_JOINTS = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

# These poses make a visible, repeatable right-arm reaching motion. They are
# torque targets; the mug remains a free physical object throughout the run.
class JointPdController:
    def __init__(self, model):
        self.model = model
        self.all_actuator_ids = np.arange(model.nu)
        all_joint_ids = model.actuator_trnid[:, 0]
        self.all_qpos_addresses = model.jnt_qposadr[all_joint_ids]
        self.all_dof_addresses = model.jnt_dofadr[all_joint_ids]
        actuator_names = {model.actuator(index).name for index in range(model.nu)}
        arm_names = [name for name in ARM_ACTUATORS if name in actuator_names]
        if len(arm_names) != len(ARM_ACTUATORS):
            arm_names = [f"{name}_joint" for name in ARM_ACTUATORS]
        self.actuator_ids = np.array([model.actuator(name).id for name in arm_names], dtype=int)
        joint_ids = model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = model.jnt_qposadr[joint_ids]
        self.dof_addresses = model.jnt_dofadr[joint_ids]
        self.right_hand_actuator_ids = np.array(
            [model.actuator(name).id for name in RIGHT_HAND_JOINTS if name in actuator_names],
            dtype=int,
        )
        hand_joint_ids = model.actuator_trnid[self.right_hand_actuator_ids, 0]
        self.right_hand_qpos_addresses = model.jnt_qposadr[hand_joint_ids]
        self.kp = np.full(model.nu, 12.0)
        self.kd = np.full(model.nu, 1.0)
        self.kp[self.actuator_ids] = [80.0, 60.0, 60.0, 50.0, 50.0, 20.0, 15.0, 15.0]
        self.kd[self.actuator_ids] = [12.0, 10.0, 10.0, 8.0, 8.0, 3.0, 3.0, 3.0]
        self.kp[self.right_hand_actuator_ids] = 8.0
        self.kd[self.right_hand_actuator_ids] = 0.25

    def torque_action(self, data, target):
        position_error = np.asarray(target) - data.qpos[self.all_qpos_addresses]
        velocity_error = -data.qvel[self.all_dof_addresses]
        bias_compensation = data.qfrc_bias[self.all_dof_addresses]
        torque = self.kp * position_error + self.kd * velocity_error + bias_compensation
        limited = self.model.actuator_ctrllimited.astype(bool)
        torque[limited] = np.clip(
            torque[limited],
            self.model.actuator_ctrlrange[limited, 0],
            self.model.actuator_ctrlrange[limited, 1],
        )
        joint_ids = self.model.actuator_trnid[:, 0]
        joint_limited = self.model.jnt_actfrclimited[joint_ids].astype(bool)
        torque[joint_limited] = np.clip(
            torque[joint_limited],
            self.model.jnt_actfrcrange[joint_ids[joint_limited], 0],
            self.model.jnt_actfrcrange[joint_ids[joint_limited], 1],
        )
        return torque

    def solve_wrist_target(self, data, target_position):
        return self.solve_position_target(data, target_position, body_name="right_wrist_yaw_link")

    def solve_position_target(
        self,
        data,
        target_position,
        body_name=None,
        geom_name=None,
        site_name=None,
        initial_joint_positions=None,
        target_rotation=None,
    ):
        if sum(name is not None for name in (body_name, geom_name, site_name)) != 1:
            raise ValueError("Provide exactly one of body_name, geom_name, or site_name")
        original_qpos = data.qpos.copy()
        original_qvel = data.qvel.copy()
        if initial_joint_positions is not None:
            data.qpos[self.qpos_addresses] = initial_joint_positions
        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        if body_name:
            object_id = self.model.body(body_name).id
        elif geom_name:
            object_id = self.model.geom(geom_name).id
        else:
            object_id = self.model.site(site_name).id
        for _ in range(100):
            mujoco.mj_forward(self.model, data)
            position_jacobian = np.zeros((3, self.model.nv))
            rotation_jacobian = np.zeros((3, self.model.nv))
            if body_name:
                mujoco.mj_jacBody(
                    self.model, data, position_jacobian, rotation_jacobian, object_id
                )
                current_position = data.body(object_id).xpos
                current_rotation = data.body(object_id).xmat
            elif geom_name:
                mujoco.mj_jacGeom(
                    self.model, data, position_jacobian, rotation_jacobian, object_id
                )
                current_position = data.geom_xpos[object_id]
                current_rotation = data.geom_xmat[object_id]
            else:
                mujoco.mj_jacSite(
                    self.model, data, position_jacobian, rotation_jacobian, object_id
                )
                current_position = data.site_xpos[object_id]
                current_rotation = data.site_xmat[object_id]
            jacobian = position_jacobian[:, self.dof_addresses]
            position_error = target_position - current_position
            error = position_error
            if target_rotation is not None:
                target_quaternion = np.empty(4)
                current_quaternion = np.empty(4)
                rotation_error = np.empty(3)
                mujoco.mju_mat2Quat(target_quaternion, np.asarray(target_rotation).reshape(9))
                mujoco.mju_mat2Quat(current_quaternion, current_rotation.reshape(9))
                mujoco.mju_subQuat(rotation_error, target_quaternion, current_quaternion)
                orientation_weight = 0.25
                jacobian = np.vstack(
                    (jacobian, orientation_weight * rotation_jacobian[:, self.dof_addresses])
                )
                error = np.concatenate((position_error, orientation_weight * rotation_error))
            joint_update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.01 * np.eye(len(error)), error
            )
            data.qpos[self.qpos_addresses] += 0.4 * joint_update
            data.qpos[self.qpos_addresses] = np.clip(
                data.qpos[self.qpos_addresses],
                self.model.jnt_range[joint_ids, 0],
                self.model.jnt_range[joint_ids, 1],
            )
        target = data.qpos[self.qpos_addresses].copy()
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(self.model, data)
        return target


def contact_involving(model, data, geom_id):
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if geom_id in (contact.geom1, contact.geom2):
            other_geom = contact.geom2 if contact.geom1 == geom_id else contact.geom1
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
    return None


def run(args):
    np.random.seed(args.seed)
    env = G1Task1Env()
    env.reset()
    controller = JointPdController(env.model)
    mug_geom_id = env.model.geom("mug_body").id
    hand_geom_id = env.model.geom("right_hand_collision").id
    mug_start = env.data.body("red_mug").xpos.copy()
    wrist_start = env.data.body("right_wrist_yaw_link").xpos.copy()
    mug_contacts = set()
    hand_contact_detected = False
    minimum_hand_mug_distance = float("inf")
    home_target = env.data.qpos[controller.all_qpos_addresses].copy()
    reach_target = home_target.copy()
    reach_target[controller.actuator_ids] = controller.solve_wrist_target(env.data, mug_start)
    phases = (
        ("home", 1.0, home_target),
        ("reach_mug", 3.0, reach_target),
        ("return", 1.5, home_target),
    )

    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.viewer else None
    try:
        for phase_name, duration, target in phases:
            step_count = int(duration / env.model.opt.timestep)
            for _ in range(step_count):
                action = controller.torque_action(env.data, target)
                env.step(action)
                hand_position = env.data.geom_xpos[hand_geom_id]
                mug_position = env.data.geom_xpos[mug_geom_id]
                minimum_hand_mug_distance = min(
                    minimum_hand_mug_distance, np.linalg.norm(hand_position - mug_position)
                )
                other_geom = contact_involving(env.model, env.data, mug_geom_id)
                if other_geom:
                    mug_contacts.add(other_geom)
                    hand_contact_detected |= other_geom == "right_hand_collision"
                if viewer:
                    viewer.sync()
                    time.sleep(env.model.opt.timestep)
            print(f"phase={phase_name} completed")
    finally:
        if viewer:
            viewer.close()

    mug_end = env.data.body("red_mug").xpos.copy()
    wrist_end = env.data.body("right_wrist_yaw_link").xpos.copy()
    mug_displacement = np.linalg.norm(mug_end - mug_start)
    wrist_displacement = np.linalg.norm(wrist_end - wrist_start)

    print("\nModel-free verification summary")
    print(f"seed: {args.seed}")
    print(f"right wrist displacement: {wrist_displacement:.3f} m")
    print(f"mug displacement: {mug_displacement:.3f} m")
    print(f"minimum hand-mug center distance: {minimum_hand_mug_distance:.3f} m")
    print("mug contact geoms:", ", ".join(sorted(mug_contacts)) or "none")
    print("kinematics verified:", wrist_displacement > 0.02)
    print("right hand-mug contact:", hand_contact_detected)
    print("physics/collision observed:", hand_contact_detected)

    if args.require_contact and not hand_contact_detected:
        raise SystemExit("No right hand-mug contact observed; adjust the scripted target pose.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--viewer", action="store_true", help="Show the MuJoCo viewer during the run.")
    parser.add_argument("--require-contact", action="store_true", help="Fail when the mug is not contacted or moved.")
    run(parser.parse_args())