import argparse
import unittest

import mujoco
import numpy as np

from envs.tasks import (
    G1Task1Env,
    G1Task2Env,
    G1Task3Env,
    LEFT_HAND_BODIES,
    RIGHT_HAND_BODIES,
)


ENV_CLASSES = (G1Task1Env, G1Task2Env, G1Task3Env)


class EnvironmentTests(unittest.TestCase):
    def test_scenes_share_articulated_g1_contract(self):
        for env_class in ENV_CLASSES:
            with self.subTest(env=env_class.__name__):
                env = env_class()
                try:
                    obs = env.reset(seed=7)
                    self.assertEqual(env.model.nu, 43)
                    self.assertEqual(obs["rgb_image"].shape, (224, 224, 3))
                    self.assertEqual(obs["qpos"].shape, (env.model.nq,))
                    self.assertEqual(obs["qvel"].shape, (env.model.nv,))
                finally:
                    env.close()

    def test_seeded_resets_are_reproducible_and_varied(self):
        for env_class in ENV_CLASSES:
            with self.subTest(env=env_class.__name__):
                env = env_class()
                try:
                    first = env.reset(seed=13)["qpos"].copy()
                    repeated = env.reset(seed=13)["qpos"].copy()
                    different = env.reset(seed=14)["qpos"].copy()
                    np.testing.assert_allclose(first, repeated)
                    self.assertFalse(np.allclose(first, different))
                finally:
                    env.close()

    def test_step_reports_finite_task_metrics(self):
        for env_class in ENV_CLASSES:
            with self.subTest(env=env_class.__name__):
                env = env_class()
                try:
                    env.reset(seed=5)
                    obs, reward, done, info = env.step(env.default_ctrl)
                    self.assertTrue(np.isfinite(reward))
                    self.assertIsInstance(done, bool)
                    self.assertIsInstance(info["success"], bool)
                    self.assertTrue(info["metrics"])
                    self.assertEqual(obs["rgb_image"].shape, (224, 224, 3))
                finally:
                    env.close()

    def test_invalid_actions_are_rejected(self):
        env = G1Task1Env()
        try:
            env.reset(seed=0)
            with self.assertRaises(ValueError):
                env.step(np.zeros(env.model.nu - 1))
            invalid = np.zeros(env.model.nu)
            invalid[0] = np.nan
            with self.assertRaises(ValueError):
                env.step(invalid)
        finally:
            env.close()

    def test_objects_start_clear_and_settle_on_table(self):
        cases = (
            (G1Task1Env, ("red_mug",)),
            (G1Task2Env, ("apple", "soda_can", "sponge")),
            (G1Task3Env, ("dustpan", "brush", "block")),
        )
        hand_bodies = LEFT_HAND_BODIES + RIGHT_HAND_BODIES
        for env_class, object_names in cases:
            with self.subTest(env=env_class.__name__):
                env = env_class()
                try:
                    for seed in range(10):
                        env.reset(seed=seed)
                        self.assertFalse(env.bodies_in_contact(hand_bodies, object_names))
                        if env_class is G1Task1Env:
                            self.assertFalse(
                                env.bodies_in_contact(("red_mug",), ("black_coaster",))
                            )
                            table_geom = env.model.geom("table_top")
                            table_top = (
                                env.data.geom("table_top").xpos[2]
                                + env.model.geom_size[table_geom.id, 2]
                            )
                            for geom_name, vertical_size_index in (
                                ("mug_body", 1),
                                ("mug_handle", 2),
                                ("mug_weighted_base", 1),
                            ):
                                geom = env.model.geom(geom_name)
                                geom_bottom = (
                                    env.data.geom(geom_name).xpos[2]
                                    - env.model.geom_size[geom.id, vertical_size_index]
                                )
                                self.assertGreaterEqual(geom_bottom, table_top - 1e-4)
                        for _ in range(25):
                            mujoco.mj_step(env.model, env.data)
                        for object_name in object_names:
                            self.assertGreaterEqual(env.body_position(object_name)[2], 0.74)
                finally:
                    env.close()

    def test_success_requires_complete_task_conditions(self):
        task1 = G1Task1Env()
        task2 = G1Task2Env()
        task3 = G1Task3Env()
        try:
            task1_success = {
                "has_contacted": True,
                "has_lifted": True,
                "mug_displacement": 0.12,
                "mug_to_coaster_xy": 0.01,
                "mug_height": 0.80,
                "max_mug_height": 0.81,
                "mug_upright": 0.99,
                "min_mug_upright": 0.99,
                "mug_speed": 0.0,
                "hand_contact": False,
            }
            self.assertTrue(task1._check_success(task1_success))
            self.assertFalse(task1._check_success(dict(task1_success, has_contacted=False)))
            self.assertFalse(task1._check_success(dict(task1_success, has_lifted=False)))
            self.assertFalse(task1._check_success(dict(task1_success, mug_to_coaster_xy=0.07)))
            self.assertFalse(task1._check_success(dict(task1_success, mug_upright=0.8)))
            self.assertFalse(task1._check_success(dict(task1_success, hand_contact=True)))

            task2_success = {
                "apple_in_red_tray": True,
                "soda_in_blue_tray": True,
                "apple_speed": 0.0,
                "soda_speed": 0.0,
            }
            self.assertTrue(task2._check_success(task2_success))
            self.assertFalse(task2._check_success(dict(task2_success, soda_in_blue_tray=False)))

            task3_success = {
                "block_in_dustpan": True,
                "brush_touched_block": True,
                "hand_touched_brush": True,
                "max_brush_handle_grasp_steps": 250,
                "block_speed": 0.0,
            }
            self.assertTrue(task3._check_success(task3_success))
            self.assertFalse(task3._check_success(dict(task3_success, brush_touched_block=False)))
            self.assertFalse(task3._check_success(dict(task3_success, hand_touched_brush=False)))
            self.assertFalse(
                task3._check_success(dict(task3_success, max_brush_handle_grasp_steps=249))
            )
        finally:
            task1.close()
            task2.close()
            task3.close()


def run_interactive_preview():
    import mujoco.viewer
    from PIL import Image

    for env_class in ENV_CLASSES:
        env = env_class()
        task_name = env_class.__name__
        try:
            for episode in range(1, 4):
                obs = env.reset(seed=episode)
                image_path = f"{task_name}_ep{episode}_rgb.png"
                Image.fromarray(obs["rgb_image"]).save(image_path)
                print(f"Saved {image_path}")
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                    env.step(env.default_ctrl)
                    viewer.sync()
                    input("Press Enter for the next setup...")
        finally:
            env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args, unittest_args = parser.parse_known_args()
    if args.interactive:
        run_interactive_preview()
    else:
        unittest.main(argv=[__file__, *unittest_args])