import numpy as np
from envs.base_env import BaseG1Env


TABLE_COLORS = np.asarray([
    [0.42, 0.28, 0.18, 1.0],
    [0.18, 0.22, 0.26, 1.0],
    [0.52, 0.44, 0.36, 1.0],
    [0.20, 0.32, 0.25, 1.0],
])
LEFT_HAND_BODIES = tuple(
    f"left_hand_{finger}_{index}_link"
    for finger, count in (("thumb", 3), ("middle", 2), ("index", 2))
    for index in range(count)
)
RIGHT_HAND_BODIES = tuple(name.replace("left_", "right_") for name in LEFT_HAND_BODIES)


class RandomizedTaskEnv(BaseG1Env):
    def _set_free_joint_pose(self, joint_name, position, yaw=0.0):
        self.data.joint(joint_name).qpos[:7] = [
            *position,
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ]

    def _randomize_table_color(self):
        geom_id = self.model.geom("table_top").id
        color_index = self.np_random.integers(len(TABLE_COLORS))
        self.model.geom_rgba[geom_id] = TABLE_COLORS[color_index]

    def _sample_separated_xy(self, x_range, y_range, occupied, minimum_distance):
        for _ in range(50):
            candidate = np.asarray([
                self.np_random.uniform(*x_range),
                self.np_random.uniform(*y_range),
            ])
            if all(np.linalg.norm(candidate - point) >= minimum_distance for point in occupied):
                return candidate
        raise RuntimeError("Could not sample a collision-free task layout")

# ==========================================
# Task 1: Single Object Pick and Place (Easy)
# ==========================================
class G1Task1Env(RandomizedTaskEnv):
    def __init__(self):
        super().__init__(
            xml_path="unitree_robots/g1/g1_task1_scene.xml",
            instruction="Pick up the red mug and place it on the black coaster."
        )
        self.initial_mug_position = np.asarray([0.0, 0.0, 0.80])
        self.max_mug_height = self.initial_mug_position[2]
        self.min_mug_upright = 1.0
        self.has_contacted = False
        self.has_lifted = False

    def _reset_task(self, options):
        if options.get("variant") == "fixed":
            mug_x, mug_y = 0.48, -0.04
            coaster_x, coaster_y = 0.58, -0.04
            yaw_angle = np.pi
        else:
            mug_x = self.np_random.uniform(0.48, 0.50)
            mug_y = self.np_random.uniform(-0.07, -0.01)
            coaster_x = mug_x + self.np_random.uniform(0.105, 0.12)
            coaster_y = mug_y + self.np_random.uniform(-0.02, 0.02)
            yaw_angle = self.np_random.uniform(0, 2 * np.pi)

        # Free joint qpos: [x, y, z, qw, qx, qy, qz]
        self._set_free_joint_pose("mug_joint", [mug_x, mug_y, 0.781], yaw_angle)
        coaster_body_id = self.model.body('black_coaster').id
        self.model.body_pos[coaster_body_id] = [coaster_x, coaster_y, 0.751]
        self.initial_mug_position = np.asarray([mug_x, mug_y, 0.781])
        self.max_mug_height = self.initial_mug_position[2]
        self.min_mug_upright = 1.0
        self.has_contacted = False
        self.has_lifted = False
        self._randomize_table_color()

    def _task_metrics(self):
        mug_position = self.body_position("red_mug")
        coaster_position = self.body_position("black_coaster")
        hand_contact = self.bodies_in_contact(RIGHT_HAND_BODIES, ("red_mug",))
        self.has_contacted = self.has_contacted or hand_contact
        self.has_lifted = self.has_lifted or mug_position[2] > self.initial_mug_position[2] + 0.02
        self.max_mug_height = max(self.max_mug_height, float(mug_position[2]))
        mug_upright = float(self.data.body("red_mug").xmat.reshape(3, 3)[2, 2])
        self.min_mug_upright = min(self.min_mug_upright, mug_upright)
        return {
            "mug_to_coaster_xy": float(np.linalg.norm(mug_position[:2] - coaster_position[:2])),
            "mug_height": float(mug_position[2]),
            "max_mug_height": self.max_mug_height,
            "mug_upright": mug_upright,
            "min_mug_upright": self.min_mug_upright,
            "mug_displacement": float(np.linalg.norm(mug_position[:2] - self.initial_mug_position[:2])),
            "has_contacted": self.has_contacted,
            "has_lifted": self.has_lifted,
            "mug_speed": self.body_speed("red_mug"),
            "hand_contact": hand_contact,
        }

    def _compute_reward(self, metrics):
        place_reward = np.exp(-8.0 * metrics["mug_to_coaster_xy"])
        move_reward = np.clip(metrics["mug_displacement"] / 0.12, 0.0, 1.0)
        return float(0.4 * place_reward + 0.3 * move_reward + 0.3 * self._check_success(metrics))

    def _check_success(self, metrics):
        return bool(
            metrics["has_contacted"]
            and metrics["has_lifted"]
            and metrics["mug_to_coaster_xy"] < 0.06
            and 0.78 < metrics["mug_height"] < 0.84
            and metrics["mug_upright"] > 0.9
            and metrics["mug_speed"] < 0.05
            and not metrics["hand_contact"]
        )


class G1Task2Env(RandomizedTaskEnv):
    def __init__(self):
        super().__init__(
            xml_path="unitree_robots/g1/g1_task2_scene.xml",
            instruction="Place the apple in the red tray and the soda can in the blue tray."
        )

    def _reset_task(self, options):
        if options.get("variant") == "fixed":
            apple_xy = np.asarray([0.48, -0.18])
            soda_xy = np.asarray([0.48, -0.04])
            sponge_xy = np.asarray([0.40, 0.10])
            red_xy = np.asarray([0.58, -0.18])
            blue_xy = np.asarray([0.58, -0.04])
            yaw_soda = yaw_sponge = 0.0
        else:
            apple_xy = np.asarray([
                self.np_random.uniform(0.47, 0.49),
                self.np_random.uniform(-0.21, -0.16),
            ])
            soda_xy = np.asarray([
                self.np_random.uniform(0.47, 0.475),
                self.np_random.uniform(-0.08, -0.02),
            ])
            red_xy = apple_xy + np.asarray([
                self.np_random.uniform(0.10, 0.12),
                self.np_random.uniform(-0.01, 0.01),
            ])
            blue_xy = soda_xy + np.asarray([
                self.np_random.uniform(0.10, 0.12),
                self.np_random.uniform(-0.01, 0.01),
            ])
            sponge_xy = np.asarray([
                self.np_random.uniform(0.38, 0.44),
                self.np_random.uniform(0.08, 0.14),
            ])
            yaw_soda = self.np_random.uniform(0, 2 * np.pi)
            yaw_sponge = self.np_random.uniform(0, 2 * np.pi)

        self._set_free_joint_pose("apple_joint", [*apple_xy, 0.79])
        self._set_free_joint_pose("soda_joint", [*soda_xy, 0.80], yaw_soda)
        self._set_free_joint_pose("sponge_joint", [*sponge_xy, 0.77], yaw_sponge)
        self.model.body_pos[self.model.body("red_tray").id] = [*red_xy, 0.751]
        self.model.body_pos[self.model.body("blue_tray").id] = [*blue_xy, 0.751]
        self._randomize_table_color()

    def _task_metrics(self):
        apple_position = self.body_position("apple")
        soda_position = self.body_position("soda_can")
        apple_distance = float(np.linalg.norm(apple_position[:2] - self.body_position("red_tray")[:2]))
        soda_distance = float(np.linalg.norm(soda_position[:2] - self.body_position("blue_tray")[:2]))
        return {
            "apple_to_red_tray_xy": apple_distance,
            "soda_to_blue_tray_xy": soda_distance,
            "apple_in_red_tray": apple_distance < 0.065 and apple_position[2] < 0.84,
            "soda_in_blue_tray": soda_distance < 0.065 and soda_position[2] < 0.84,
            "apple_speed": self.body_speed("apple"),
            "soda_speed": self.body_speed("soda_can"),
        }

    def _compute_reward(self, metrics):
        apple_reward = np.exp(-8.0 * metrics["apple_to_red_tray_xy"])
        soda_reward = np.exp(-8.0 * metrics["soda_to_blue_tray_xy"])
        return float(0.4 * apple_reward + 0.4 * soda_reward + 0.2 * self._check_success(metrics))

    def _check_success(self, metrics):
        return bool(
            metrics["apple_in_red_tray"]
            and metrics["soda_in_blue_tray"]
            and metrics["apple_speed"] < 0.05
            and metrics["soda_speed"] < 0.05
        )


class G1Task3Env(RandomizedTaskEnv):
    def __init__(self):
        super().__init__(
            xml_path="unitree_robots/g1/g1_task3_scene.xml",
            instruction="Grasp the brush handle with the right hand and sweep the block into the green target."
        )
        self.brush_touched_block = False
        self.hand_touched_brush = False
        self.brush_handle_grasp_steps = 0
        self.max_brush_handle_grasp_steps = 0

    def _reset_task(self, options):
        if options.get("variant") == "fixed":
            brush_x, brush_y = 0.42, -0.04
            block_x, block_y = 0.52, -0.04
            dustpan_x, dustpan_y = 0.62, -0.04
            dustpan_yaw = brush_yaw = block_yaw = 0.0
        else:
            brush_x = self.np_random.uniform(0.418, 0.425)
            brush_y = self.np_random.uniform(-0.05, -0.03)
            block_x = brush_x + self.np_random.uniform(0.09, 0.11)
            block_y = brush_y + self.np_random.uniform(-0.01, 0.01)
            dustpan_x = block_x + self.np_random.uniform(0.09, 0.11)
            dustpan_y = block_y + self.np_random.uniform(-0.01, 0.01)
            dustpan_yaw = 0.0
            brush_yaw = self.np_random.uniform(-0.12, 0.12)
            block_yaw = self.np_random.uniform(-0.25, 0.25)

        self._set_free_joint_pose("brush_joint", [brush_x, brush_y, 0.77], brush_yaw)
        self._set_free_joint_pose("block_joint", [block_x, block_y, 0.77], block_yaw)
        self.model.body_pos[self.model.body("dustpan").id] = [dustpan_x, dustpan_y, 0.751]
        self.brush_touched_block = False
        self.hand_touched_brush = False
        self.brush_handle_grasp_steps = 0
        self.max_brush_handle_grasp_steps = 0
        self._randomize_table_color()

    def _brush_handle_grasped(self):
        handle_geom_id = self.model.geom("brush_handle").id
        contacting_digits = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if handle_geom_id not in (contact.geom1, contact.geom2):
                continue
            hand_geom_id = contact.geom2 if contact.geom1 == handle_geom_id else contact.geom1
            body_name = self.model.body(self.model.geom_bodyid[hand_geom_id]).name or ""
            if not body_name.startswith("right_hand_"):
                continue
            for digit in ("thumb", "index", "middle"):
                if digit in body_name:
                    contacting_digits.add(digit)
        return bool(
            "thumb" in contacting_digits
            and ({"index", "middle"} & contacting_digits)
        )

    def _task_metrics(self):
        block_position = self.body_position("block")
        dustpan_position = self.body_position("dustpan")
        brush_contact = self.bodies_in_contact(("brush",), ("block",))
        hand_contact = self.bodies_in_contact(RIGHT_HAND_BODIES, ("brush",))
        brush_handle_grasped = self._brush_handle_grasped()
        if brush_handle_grasped:
            self.brush_handle_grasp_steps += 1
            self.max_brush_handle_grasp_steps = max(
                self.max_brush_handle_grasp_steps,
                self.brush_handle_grasp_steps,
            )
        else:
            self.brush_handle_grasp_steps = 0
        self.brush_touched_block = self.brush_touched_block or brush_contact
        self.hand_touched_brush = self.hand_touched_brush or hand_contact
        block_distance = float(np.linalg.norm(block_position[:2] - dustpan_position[:2]))
        return {
            "block_to_dustpan_xy": block_distance,
            "block_in_dustpan": block_distance < 0.05 and block_position[2] < 0.80,
            "brush_contact": brush_contact,
            "brush_touched_block": self.brush_touched_block,
            "hand_touched_brush": self.hand_touched_brush,
            "brush_handle_grasped": brush_handle_grasped,
            "max_brush_handle_grasp_steps": self.max_brush_handle_grasp_steps,
            "block_speed": self.body_speed("block"),
        }

    def _compute_reward(self, metrics):
        sweep_reward = np.exp(-10.0 * metrics["block_to_dustpan_xy"])
        return float(
            0.5 * sweep_reward
            + 0.2 * metrics["brush_touched_block"]
            + 0.1 * metrics["hand_touched_brush"]
            + 0.2 * self._check_success(metrics)
        )

    def _check_success(self, metrics):
        return bool(
            metrics["block_in_dustpan"]
            and metrics["brush_touched_block"]
            and metrics["hand_touched_brush"]
            and metrics["max_brush_handle_grasp_steps"] >= 250
            and metrics["block_speed"] < 0.05
        )