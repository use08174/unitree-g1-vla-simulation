"""Collect model-free interaction trajectories from G1 manipulation environments."""

import argparse
import json
from pathlib import Path

import numpy as np

from envs.tasks import G1Task1Env, G1Task2Env, G1Task3Env
from scripts.control.verify_model_free_control import JointPdController


ENVIRONMENTS = {
    1: (G1Task1Env, "red_mug", "Pick up the red mug and place it on the black coaster."),
    2: (G1Task2Env, "apple", "Place the apple in the red target and the soda can in the blue target."),
    3: (G1Task3Env, "brush", "Grasp the brush and use it to sweep the block into the green target."),
}
MINIMUM_CONTACT_FRAMES = 3
TASK1_LOW_ARM_SEED = np.array(
    [0.49324, -0.69451, 0.39138, -0.24441, 0.14184, -1.01912, 1.14032, -0.22488]
)


def quintic_blend(progress):
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def cartesian_path_targets(
    controller,
    data,
    home_target,
    start_position,
    end_position,
    initial_arm_seed,
    hand_pose,
    waypoint_count,
):
    targets = []
    arm_seed = initial_arm_seed
    for progress in np.linspace(0.0, 1.0, waypoint_count + 1)[1:]:
        position = start_position + progress * (end_position - start_position)
        target = home_target.copy()
        target[controller.actuator_ids] = controller.solve_position_target(
            data,
            position,
            geom_name="right_hand_palm_collision",
            initial_joint_positions=arm_seed,
        )
        target[controller.right_hand_actuator_ids] = hand_pose
        targets.append(target)
        arm_seed = target[controller.actuator_ids]
    return targets


def target_contact(model, data, target_body_id, hand_geom_ids):
    target_geom_start = model.body_geomadr[target_body_id]
    target_geom_ids = set(
        range(target_geom_start, target_geom_start + model.body_geomnum[target_body_id])
    )
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if contact.geom1 in hand_geom_ids or contact.geom2 in hand_geom_ids:
            other_geom = contact.geom2 if contact.geom1 in hand_geom_ids else contact.geom1
            if other_geom in target_geom_ids:
                return True
    return False


def sustained_task_contact(contact_samples, phase_ids):
    run_length = 0
    for contact, phase_id in zip(contact_samples, phase_ids):
        run_length = run_length + 1 if contact and phase_id > 0 else 0
        if run_length >= MINIMUM_CONTACT_FRAMES:
            return True
    return False


def collect_episode(
    environment_id,
    seed,
    sample_every,
    variant=None,
    task1_behavior="grasp",
    task2_behavior="grasp",
    task3_behavior="grasp",
):
    np.random.seed(seed)
    environment_class, target_body_name, demonstration_instruction = ENVIRONMENTS[environment_id]
    env = environment_class()
    reset_options = {"variant": variant} if variant else None
    initial_observation = env.reset(seed=seed, options=reset_options)
    controller = JointPdController(env.model)
    if environment_id == 2 and task2_behavior == "grasp":
        target_body_name = "soda_can"
    target_body_id = env.model.body(target_body_name).id
    hand_geom_ids = {
        geom_id
        for geom_id, body_id in enumerate(env.model.geom_bodyid)
        if (env.model.body(body_id).name or "").startswith("right_hand_")
    }
    if not hand_geom_ids:
        hand_geom_ids = {env.model.geom("right_hand_collision").id}
    home_target = env.data.qpos[controller.all_qpos_addresses].copy()
    target_position = env.data.body(target_body_name).xpos.copy()
    if environment_id == 1 and task1_behavior == "grasp":
        demonstration_instruction = "Pick up the red mug and place it on the black coaster."
        full_close_hand = np.array([0.25, -0.7, -1.35, 1.25, 1.45, 1.25, 1.45])
        preshape_hand = 0.35 * full_close_hand
        closed_hand = 0.60 * full_close_hand
        open_hand = np.zeros(7)
        grasp_rotation = env.data.site("right_pinch_site").xmat.copy()
        grasp_position = target_position + np.array([-0.02, -0.01, 0.0])

        def grasp_target(position, hand_pose, arm_seed):
            target = home_target.copy()
            target[controller.actuator_ids] = controller.solve_position_target(
                env.data,
                position,
                site_name="right_pinch_site",
                initial_joint_positions=arm_seed,
                target_rotation=grasp_rotation,
            )
            target[controller.right_hand_actuator_ids] = hand_pose
            return target

        approach_target = grasp_target(
            grasp_position + np.array([0.0, 0.0, 0.12]),
            preshape_hand,
            TASK1_LOW_ARM_SEED,
        )
        grasp_pose_target = grasp_target(
            grasp_position, preshape_hand, approach_target[controller.actuator_ids]
        )
        close_target = grasp_pose_target.copy()
        close_target[controller.right_hand_actuator_ids] = closed_hand
        lift_targets = []
        arm_seed = grasp_pose_target[controller.actuator_ids]
        for height in np.linspace(0.02, 0.08, 4):
            lift_target = grasp_target(
                grasp_position + np.array([0.0, 0.0, height]),
                closed_hand,
                arm_seed,
            )
            lift_targets.append(lift_target)
            arm_seed = lift_target[controller.actuator_ids]
        coaster_position = env.data.body("black_coaster").xpos.copy()
        grasp_offset = grasp_position - target_position
        placement_position = coaster_position + grasp_offset
        placement_position[2] = grasp_position[2]
        transfer_target = grasp_target(
            placement_position + np.array([0.0, 0.0, 0.08]),
            closed_hand,
            arm_seed,
        )
        lower_target = grasp_target(
            placement_position,
            closed_hand,
            transfer_target[controller.actuator_ids],
        )
        release_target = lower_target.copy()
        release_target[controller.right_hand_actuator_ids] = open_hand
        retreat_target = grasp_target(
            placement_position + np.array([0.0, 0.0, 0.12]),
            open_hand,
            lower_target[controller.actuator_ids],
        )
        phases = [
            ("home", 0.75, home_target),
            ("approach", 1.5, approach_target),
            ("grasp", 1.5, grasp_pose_target),
            ("close", 1.25, close_target),
            ("hold", 0.75, close_target),
        ]
        phases.extend(("lift", 0.75, target) for target in lift_targets)
        phases.extend([
            ("lift_hold", 1.0, lift_targets[-1]),
            ("transfer", 2.0, transfer_target),
            ("lower", 1.5, lower_target),
            ("release", 1.0, release_target),
            ("retreat", 1.0, retreat_target),
            ("settle", 1.0, retreat_target),
        ])
    elif environment_id == 1:
        coaster_position = env.data.body("black_coaster").xpos.copy()
        push_hand = np.array([0.0, 0.4, -1.2, 1.2, 1.4, 1.2, 1.4])
        push_direction = coaster_position[:2] - target_position[:2]
        push_direction /= np.linalg.norm(push_direction)
        approach_position = np.concatenate(
            (target_position[:2] - 0.07 * push_direction, [0.76])
        )
        push_position = np.concatenate(
            (coaster_position[:2] - 0.02 * push_direction, [0.76])
        )
        lift_position = push_position + np.array([0.0, 0.0, 0.25])

        approach_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            approach_position,
            approach_position,
            TASK1_LOW_ARM_SEED,
            push_hand,
            1,
        )[0]
        push_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            approach_position,
            push_position,
            approach_target[controller.actuator_ids],
            push_hand,
            7,
        )
        lift_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            push_position,
            lift_position,
            push_targets[-1][controller.actuator_ids],
            push_hand,
            1,
        )[0]
        phases = [
            ("home", 0.75, home_target),
            ("approach", 1.5, approach_target),
        ]
        phases.extend(("push", 5.5 / len(push_targets), target) for target in push_targets)
        phases.extend((
            ("lift", 1.0, lift_target),
            ("settle", 1.0, lift_target),
        ))
    elif environment_id == 2 and task2_behavior == "grasp":
        demonstration_instruction = "Place the apple in the red target and the soda can in the blue target."
        full_close_hand = np.array([0.25, -0.7, -1.35, 1.25, 1.45, 1.25, 1.45])
        preshape_hand = 0.35 * full_close_hand
        closed_hand = 0.60 * full_close_hand
        open_hand = np.zeros(7)
        grasp_rotation = env.data.site("right_pinch_site").xmat.copy()
        grasp_position = target_position + np.array([-0.02, -0.01, 0.0])
        blue_position = env.data.body("blue_tray").xpos.copy()
        apple_position = env.data.body("apple").xpos.copy()
        red_position = env.data.body("red_tray").xpos.copy()

        apple_push_hand = np.array([0.0, 0.4, -1.2, 1.2, 1.4, 1.2, 1.4])
        apple_push_direction = red_position[:2] - apple_position[:2]
        apple_push_direction /= np.linalg.norm(apple_push_direction)
        apple_approach_position = np.concatenate(
            (apple_position[:2] - 0.07 * apple_push_direction, [0.76])
        )
        apple_finish_position = np.concatenate(
            (red_position[:2] - 0.02 * apple_push_direction, [0.76])
        )
        apple_approach_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_approach_position,
            apple_approach_position,
            TASK1_LOW_ARM_SEED,
            apple_push_hand,
            1,
        )[0]
        apple_push_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_approach_position,
            apple_finish_position,
            apple_approach_target[controller.actuator_ids],
            apple_push_hand,
            7,
        )
        apple_raise_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_finish_position,
            apple_finish_position + np.array([0.0, 0.0, 0.20]),
            apple_push_targets[-1][controller.actuator_ids],
            apple_push_hand,
            1,
        )[0]

        def pinch_target(position, hand_pose, arm_seed):
            target = home_target.copy()
            target[controller.actuator_ids] = controller.solve_position_target(
                env.data,
                position,
                site_name="right_pinch_site",
                initial_joint_positions=arm_seed,
                target_rotation=grasp_rotation,
            )
            target[controller.right_hand_actuator_ids] = hand_pose
            return target

        above_target = pinch_target(
            grasp_position + np.array([0.0, 0.0, 0.12]),
            preshape_hand,
            home_target[controller.actuator_ids],
        )
        grasp_target = pinch_target(
            grasp_position,
            preshape_hand,
            above_target[controller.actuator_ids],
        )
        close_target = grasp_target.copy()
        close_target[controller.right_hand_actuator_ids] = closed_hand
        first_lift_target = pinch_target(
            grasp_position + np.array([0.0, 0.0, 0.04]),
            closed_hand,
            close_target[controller.actuator_ids],
        )
        second_lift_target = pinch_target(
            grasp_position + np.array([0.0, 0.0, 0.08]),
            closed_hand,
            first_lift_target[controller.actuator_ids],
        )
        placement_position = blue_position + np.array([-0.02, -0.01, 0.049])
        transfer_target = pinch_target(
            placement_position + np.array([0.0, 0.0, 0.08]),
            closed_hand,
            second_lift_target[controller.actuator_ids],
        )
        lower_target = pinch_target(
            placement_position,
            closed_hand,
            transfer_target[controller.actuator_ids],
        )
        release_target = lower_target.copy()
        release_target[controller.right_hand_actuator_ids] = open_hand
        retreat_target = pinch_target(
            placement_position + np.array([0.0, 0.0, 0.12]),
            open_hand,
            lower_target[controller.actuator_ids],
        )
        phases = [("home", 0.75, home_target), ("apple_approach", 1.5, apple_approach_target)]
        phases.extend(
            ("apple_push", 4.0 / len(apple_push_targets), target)
            for target in apple_push_targets
        )
        phases.extend([
            ("raise", 1.0, apple_raise_target),
            ("can_above", 1.5, above_target),
            ("descend", 1.5, grasp_target),
            ("close", 1.25, close_target),
            ("hold", 0.75, close_target),
            ("lift", 1.5, first_lift_target),
            ("lift", 1.5, second_lift_target),
            ("lift_hold", 1.0, second_lift_target),
            ("transfer", 2.0, transfer_target),
            ("lower", 1.5, lower_target),
            ("release", 1.0, release_target),
            ("retreat", 1.0, retreat_target),
            ("settle", 1.0, retreat_target),
        ])
    elif environment_id == 2 and task2_behavior == "sort":
        demonstration_instruction = (
            "Move the apple to the red target and the soda can to the blue target."
        )
        push_hand = np.array([0.0, 0.4, -1.2, 1.2, 1.4, 1.2, 1.4])
        apple_position = env.data.body("apple").xpos.copy()
        soda_position = env.data.body("soda_can").xpos.copy()
        red_position = env.data.body("red_tray").xpos.copy()
        blue_position = env.data.body("blue_tray").xpos.copy()

        def push_positions(object_position, goal_position):
            direction = goal_position[:2] - object_position[:2]
            direction /= np.linalg.norm(direction)
            approach = np.concatenate(
                (object_position[:2] - 0.07 * direction, [0.76])
            )
            finish = np.concatenate((goal_position[:2] - 0.02 * direction, [0.76]))
            return approach, finish

        apple_approach, apple_finish = push_positions(apple_position, red_position)
        soda_approach, soda_finish = push_positions(soda_position, blue_position)
        apple_approach_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_approach,
            apple_approach,
            TASK1_LOW_ARM_SEED,
            push_hand,
            1,
        )[0]
        apple_push_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_approach,
            apple_finish,
            apple_approach_target[controller.actuator_ids],
            push_hand,
            7,
        )
        raised_apple_finish = apple_finish + np.array([0.0, 0.0, 0.20])
        raised_soda_approach = soda_approach + np.array([0.0, 0.0, 0.20])
        raise_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            apple_finish,
            raised_apple_finish,
            apple_push_targets[-1][controller.actuator_ids],
            push_hand,
            1,
        )[0]
        transfer_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            raised_apple_finish,
            raised_soda_approach,
            raise_target[controller.actuator_ids],
            push_hand,
            4,
        )
        soda_approach_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            raised_soda_approach,
            soda_approach,
            transfer_targets[-1][controller.actuator_ids],
            push_hand,
            1,
        )[0]
        soda_push_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            soda_approach,
            soda_finish,
            soda_approach_target[controller.actuator_ids],
            push_hand,
            7,
        )
        final_lift_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            soda_finish,
            soda_finish + np.array([0.0, 0.0, 0.20]),
            soda_push_targets[-1][controller.actuator_ids],
            push_hand,
            1,
        )[0]
        phases = [("home", 0.75, home_target), ("apple_approach", 1.5, apple_approach_target)]
        phases.extend(("apple_push", 4.0 / len(apple_push_targets), target) for target in apple_push_targets)
        phases.append(("raise", 1.0, raise_target))
        phases.extend(("transfer", 1.5 / len(transfer_targets), target) for target in transfer_targets)
        phases.append(("soda_approach", 1.0, soda_approach_target))
        phases.extend(("soda_push", 4.0 / len(soda_push_targets), target) for target in soda_push_targets)
        phases.extend((("lift", 1.0, final_lift_target), ("settle", 1.0, final_lift_target)))
    elif environment_id == 3 and task3_behavior == "grasp":
        demonstration_instruction = "Grasp the brush and use it to sweep the block into the green target."
        full_close_hand = np.array([0.25, -0.7, -1.35, 1.25, 1.45, 1.25, 1.45])
        preshape_hand = 0.35 * full_close_hand
        closed_hand = 0.65 * full_close_hand
        grasp_rotation = env.data.site("right_pinch_site").xmat.copy()
        grasp_position = target_position + np.array([-0.02, -0.01, 0.065])

        def brush_pinch_target(position, hand_pose, arm_seed):
            target = home_target.copy()
            target[controller.actuator_ids] = controller.solve_position_target(
                env.data,
                position,
                site_name="right_pinch_site",
                initial_joint_positions=arm_seed,
                target_rotation=grasp_rotation,
            )
            target[controller.right_hand_actuator_ids] = hand_pose
            return target

        brush_above_target = brush_pinch_target(
            grasp_position + np.array([0.0, 0.0, 0.12]),
            preshape_hand,
            home_target[controller.actuator_ids],
        )
        brush_grasp_target = brush_pinch_target(
            grasp_position,
            preshape_hand,
            brush_above_target[controller.actuator_ids],
        )
        brush_close_target = brush_grasp_target.copy()
        brush_close_target[controller.right_hand_actuator_ids] = closed_hand
        dustpan_position = env.data.body("dustpan").xpos.copy()
        sweep_distance = dustpan_position[0] - target_position[0] - 0.05
        brush_sweep_targets = []
        arm_seed = brush_close_target[controller.actuator_ids]
        for progress in np.linspace(0.0, 1.0, 10)[1:]:
            sweep_position = grasp_position + np.array(
                [progress * sweep_distance, 0.0, 0.0]
            )
            sweep_target = brush_pinch_target(
                sweep_position,
                closed_hand,
                arm_seed,
            )
            brush_sweep_targets.append(sweep_target)
            arm_seed = sweep_target[controller.actuator_ids]
        brush_release_target = brush_sweep_targets[-1].copy()
        brush_release_target[controller.right_hand_actuator_ids] = np.zeros(7)
        brush_retreat_target = brush_pinch_target(
            grasp_position + np.array([sweep_distance, 0.0, 0.12]),
            np.zeros(7),
            brush_sweep_targets[-1][controller.actuator_ids],
        )
        phases = [
            ("home", 0.75, home_target),
            ("brush_above", 1.5, brush_above_target),
            ("brush_descend", 1.5, brush_grasp_target),
            ("brush_close", 1.25, brush_close_target),
            ("brush_hold", 0.75, brush_close_target),
        ]
        phases.extend(
            ("brush_sweep", 6.0 / len(brush_sweep_targets), target)
            for target in brush_sweep_targets
        )
        phases.extend([
            ("brush_release", 1.0, brush_release_target),
            ("brush_retreat", 1.0, brush_retreat_target),
            ("settle", 1.0, brush_retreat_target),
        ])
    elif environment_id == 3 and task3_behavior == "sweep":
        demonstration_instruction = (
            "Use the right hand to push the brush and sweep the block into the green target."
        )
        push_hand = np.array([0.0, 0.4, -1.2, 1.2, 1.4, 1.2, 1.4])
        brush_position = env.data.body("brush").xpos.copy()
        dustpan_position = env.data.body("dustpan").xpos.copy()
        sweep_direction = dustpan_position[:2] - brush_position[:2]
        sweep_direction /= np.linalg.norm(sweep_direction)
        approach_position = np.concatenate(
            (brush_position[:2] - 0.08 * sweep_direction, [0.76])
        )
        sweep_position = np.concatenate(
            (dustpan_position[:2] - 0.05 * sweep_direction, [0.76])
        )
        approach_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            approach_position,
            approach_position,
            TASK1_LOW_ARM_SEED,
            push_hand,
            1,
        )[0]
        sweep_targets = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            approach_position,
            sweep_position,
            approach_target[controller.actuator_ids],
            push_hand,
            9,
        )
        lift_target = cartesian_path_targets(
            controller,
            env.data,
            home_target,
            sweep_position,
            sweep_position + np.array([0.0, 0.0, 0.20]),
            sweep_targets[-1][controller.actuator_ids],
            push_hand,
            1,
        )[0]
        phases = [("home", 0.75, home_target), ("approach", 1.5, approach_target)]
        phases.extend(("sweep", 6.0 / len(sweep_targets), target) for target in sweep_targets)
        phases.extend((("lift", 1.0, lift_target), ("settle", 1.0, lift_target)))
    else:
        hand_position = env.data.geom_xpos[next(iter(hand_geom_ids))].copy()
        approach_direction = hand_position - target_position
        approach_direction /= np.linalg.norm(approach_direction)
        reach_target = home_target.copy()
        contact_position = target_position + 0.06 * approach_direction
        reach_target[controller.actuator_ids] = controller.solve_position_target(
            env.data, contact_position, geom_name="right_hand_collision"
        )
        phases = (
            ("home", 1.0, home_target),
            ("reach_target", 3.0, reach_target),
            ("hold_target", 1.5, reach_target),
        )
    observations = []
    actions = []
    wrist_positions = []
    wrist_quaternions = []
    phase_ids = []
    hand_target_contacts = []
    brush_handle_grasps = []
    target_positions = []
    fixed_body_positions = {}
    if environment_id == 1:
        fixed_body_positions["black_coaster"] = env.model.body_pos[
            env.model.body("black_coaster").id
        ].tolist()

    previous_target = home_target
    for phase_id, (_, duration, target) in enumerate(phases):
        step_count = int(duration / env.model.opt.timestep)
        for step_index in range(step_count):
            progress = (step_index + 1) / step_count
            smooth_progress = quintic_blend(progress)
            control_target = previous_target + smooth_progress * (target - previous_target)
            action = controller.torque_action(env.data, control_target)
            observation, _, _, _ = env.step(action)
            hand_target_contact = target_contact(
                env.model, env.data, target_body_id, hand_geom_ids
            )
            if step_index % sample_every == 0:
                observations.append(observation)
                actions.append(action.copy())
                wrist_positions.append(env.data.body("right_wrist_yaw_link").xpos.copy())
                wrist_quaternions.append(env.data.body("right_wrist_yaw_link").xquat.copy())
                phase_ids.append(phase_id)
                hand_target_contacts.append(hand_target_contact)
                brush_handle_grasps.append(
                    environment_id == 3 and env._brush_handle_grasped()
                )
                target_positions.append(env.data.body(target_body_name).xpos.copy())
            previous_target = target

    if environment_id == 1 and task1_behavior == "grasp":
        task_success = env._check_success(env._task_metrics())
    elif environment_id == 2 and task2_behavior == "grasp":
        sampled_positions = np.stack(target_positions)
        lifted = sampled_positions[:, 2] > target_position[2] + 0.02
        lift_hold_phase_ids = {
            phase_id
            for phase_id, (phase_name, _, _) in enumerate(phases)
            if phase_name == "lift_hold"
        }
        retained_samples = sum(
            bool(contact and is_lifted and phase_id in lift_hold_phase_ids)
            for contact, is_lifted, phase_id in zip(hand_target_contacts, lifted, phase_ids)
        )
        task_success = retained_samples >= 4 and env._check_success(env._task_metrics())
    elif environment_id == 1:
        task_success = env._check_success(env._task_metrics())
    elif environment_id == 2 and task2_behavior == "sort":
        task_success = env._check_success(env._task_metrics())
    elif environment_id == 3 and task3_behavior in ("sweep", "grasp"):
        task_success = env._check_success(env._task_metrics())
    else:
        task_success = sustained_task_contact(hand_target_contacts, phase_ids)

    return {
        "rgb_images": np.stack([observation["rgb_image"] for observation in observations]),
        "qpos": np.stack([observation["qpos"] for observation in observations]),
        "qvel": np.stack([observation["qvel"] for observation in observations]),
        "actions": np.stack(actions),
        "right_wrist_positions": np.stack(wrist_positions),
        "right_wrist_quaternions": np.stack(wrist_quaternions),
        "phase_ids": np.asarray(phase_ids, dtype=np.int8),
        "hand_target_contacts": np.asarray(hand_target_contacts, dtype=bool),
        "brush_handle_grasps": np.asarray(brush_handle_grasps, dtype=bool),
        "target_positions": np.stack(target_positions),
        "task_success": np.asarray(task_success),
        "metadata": json.dumps(
            {
                "environment": f"Environment {environment_id}",
                "environment_instruction": initial_observation["instruction"],
                "instruction": demonstration_instruction,
                "target_body": target_body_name,
                "seed": seed,
                "variant": variant or "random",
                "task1_behavior": task1_behavior if environment_id == 1 else None,
                "task2_behavior": task2_behavior if environment_id == 2 else None,
                "task3_behavior": task3_behavior if environment_id == 3 else None,
                "sample_every": sample_every,
                "control_frequency_hz": 1.0 / (sample_every * env.model.opt.timestep),
                "action_representation": f"{env.model.nu}D MuJoCo actuator torque",
                "end_effector": (
                    "right_pinch_site"
                    if environment_id == 1 and task1_behavior == "grasp"
                    else "right_pinch_site"
                    if environment_id == 2 and task2_behavior == "grasp"
                    or environment_id == 3 and task3_behavior == "grasp"
                    else "right_hand_palm_collision"
                    if environment_id == 1
                    or environment_id == 2 and task2_behavior == "sort"
                    or environment_id == 3 and task3_behavior == "sweep"
                    else "right_wrist_yaw_link"
                ),
                "trajectory_generation": "Cartesian waypoints, warm-start Jacobian IK, quintic joint timing",
                "phases": [phase[0] for phase in phases],
                "fixed_body_positions": fixed_body_positions,
            }
        ),
    }


def main(args):
    environment_ids = (1, 2, 3) if args.environment == "all" else (int(args.environment),)
    for environment_id in environment_ids:
        output_directory = Path(args.output_dir) / f"environment{environment_id}"
        output_directory.mkdir(parents=True, exist_ok=True)
        for episode_index in range(args.episodes):
            seed = args.seed + episode_index
            episode = collect_episode(
                environment_id,
                seed,
                args.sample_every,
                args.variant,
                args.task1_behavior,
                args.task2_behavior,
                args.task3_behavior,
            )
            output_path = output_directory / f"episode_{episode_index:04d}.npz"
            np.savez_compressed(output_path, **episode)
            print(
                f"saved {output_path}: {len(episode['actions'])} samples, "
                f"task success={bool(episode['task_success'])}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/groot_source")
    parser.add_argument("--environment", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--variant", choices=("fixed", "random"), default="random")
    parser.add_argument(
        "--task1-behavior",
        choices=("push", "grasp"),
        default="grasp",
        help="Use the pick-and-place controller or the legacy push controller.",
    )
    parser.add_argument(
        "--task2-behavior",
        choices=("touch", "sort", "grasp"),
        default="grasp",
        help="Use the legacy touch probe, push sorting, or instruction-complete soda pick-and-place.",
    )
    parser.add_argument(
        "--task3-behavior",
        choices=("touch", "sweep", "grasp"),
        default="grasp",
        help="Use the legacy touch probe, palm-push sweeping, or instruction-complete brush grasp-and-sweep.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Physics steps between samples; 25 gives 20 Hz for the current 0.002 s timestep.",
    )
    main(parser.parse_args())