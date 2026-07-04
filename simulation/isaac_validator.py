"""Isaac Gym validation setup adapted from DexGraspNet."""

from time import sleep

from isaacgym import gymapi


gym = gymapi.acquire_gym()


JOINT_NAMES = {
    "shadow_hand": [
        "robot0:THJ4", "robot0:THJ3", "robot0:THJ2", "robot0:THJ1", "robot0:THJ0",
        "robot0:FFJ3", "robot0:FFJ2", "robot0:FFJ1", "robot0:FFJ0",
        "robot0:MFJ3", "robot0:MFJ2", "robot0:MFJ1", "robot0:MFJ0",
        "robot0:RFJ3", "robot0:RFJ2", "robot0:RFJ1", "robot0:RFJ0",
        "robot0:LFJ4", "robot0:LFJ3", "robot0:LFJ2", "robot0:LFJ1", "robot0:LFJ0",
    ],
    "allegro_hand": [
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
    ],
    "barrett": [
        "a_palm_link1_joint",
        "a_link1_link2_joint",
        "a_link2_link3_joint",
        "b_palm_link1_joint",
        "b_link1_link2_joint",
        "b_link2_link3_joint",
        "c_palm_link2_joint_0",
        "c_link2_link3_joint",
    ],
    "robotiq_3f": [
        "palm_finger_1_joint",
        "finger_1_joint_1",
        "finger_1_joint_2",
        "finger_1_joint_3",
        "palm_finger_2_joint",
        "finger_2_joint_1",
        "finger_2_joint_2",
        "finger_2_joint_3",
        "finger_middle_joint_1",
        "finger_middle_joint_2",
        "finger_middle_joint_3",
    ],
    "human_hand": [
        "palm_index1_0_joint",
        "index1_0_index1_joint",
        "index1_index2_joint",
        "index2_index3_joint",
        "palm_mid1_0_joint",
        "mid1_0_mid1_joint",
        "mid1_mid2_joint",
        "mid2_mid3_joint",
        "palm_ring1_0_joint",
        "ring1_0_ring1_joint",
        "ring1_ring2_joint",
        "ring2_ring3_joint",
        "palm_pinky1_0_joint",
        "pinky1_0_pinky1_joint",
        "pinky1_pinky2_joint",
        "pinky2_pinky3_joint",
        "palm_thumb1_0_joint",
        "thumb1_0_thumb1_joint",
        "thumb1_thumb2_joint",
        "thumb2_thumb3_joint",
    ],
}


HAND_ALIASES = {
    "shadow": "shadow_hand",
    "shadow_hand": "shadow_hand",
    "allegro": "allegro_hand",
    "allegro_hand": "allegro_hand",
    "barrett": "barrett",
    "robotiq_3f": "robotiq_3f",
    "human_hand": "human_hand",
}


class IsaacValidator:
    """Isaac Gym validator used by the setup-1 simulation results.

    The simulator runs with gravity enabled and reports success after a
    physics rollout.
    """

    def __init__(
        self,
        mode="direct",
        hand_friction=7.0,
        obj_friction=7.0,
        sim_step=200,
        gpu=0,
        debug_interval=0.05,
        hand_type="shadow_hand",
    ):
        self.hand_type = HAND_ALIASES.get(hand_type, hand_type)
        if self.hand_type not in JOINT_NAMES:
            raise ValueError(f"Unsupported hand type: {hand_type}")

        self.joint_names = JOINT_NAMES[self.hand_type]
        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.sim_step = sim_step
        self.gpu = gpu
        self.debug_interval = debug_interval

        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.obj_asset_cache = {}
        self.hand_asset = None

        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 16
        self.sim_params.physx.num_velocity_iterations = 2
        self.sim_params.physx.max_gpu_contact_pairs = 1024 * 1024 * 40
        self.sim_params.use_gpu_pipeline = False

        compute_device = -1 if mode == "direct" else gpu
        self.sim = gym.create_sim(gpu, compute_device, gymapi.SIM_PHYSX, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 10
        plane_params.segmentation_id = 0
        plane_params.normal = gymapi.Vec3(0.0, -1.0, 0.0)
        gym.add_ground(self.sim, plane_params)

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = True

        self.viewer = None
        self.has_viewer = mode == "gui"
        if self.has_viewer:
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(
                self.viewer,
                None,
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 0, 0),
            )

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        if self.hand_type != "allegro_hand":
            self.hand_asset_options.collapse_fixed_joints = True
        if self.hand_type != "shadow_hand":
            self.hand_asset_options.use_physx_armature = True

        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 15
        if self.hand_type == "human_hand":
            self.obj_asset_options.vhacd_enabled = True
            self.obj_asset_options.vhacd_params = gymapi.VhacdParams()
            self.obj_asset_options.vhacd_params.resolution = 1000000

    def set_asset(self, hand_root, hand_file):
        self.hand_asset = gym.load_asset(
            self.sim,
            str(hand_root),
            str(hand_file),
            self.hand_asset_options,
        )

    def load_obj(self, obj_root, obj_file):
        object_code = f"{obj_root}/{obj_file}"
        if object_code not in self.obj_asset_cache:
            self.obj_asset_cache[object_code] = gym.load_asset(
                self.sim,
                str(obj_root),
                str(obj_file),
                self.obj_asset_options,
            )
        return self.obj_asset_cache[object_code]

    def add_env(
        self,
        hand_rotation,
        hand_translation,
        hand_qpos,
        obj_scale,
        obj_root,
        obj_file,
    ):
        env = gym.create_env(
            self.sim,
            gymapi.Vec3(-1, -1, -1),
            gymapi.Vec3(1, 1, 1),
            6,
        )
        self.envs.append(env)

        hand_pose = gymapi.Transform()
        hand_pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
        hand_pose.p = gymapi.Vec3(*hand_translation)
        hand_actor_handle = gym.create_actor(
            env,
            self.hand_asset,
            hand_pose,
            self.hand_type,
            0,
            -1,
        )
        self.hand_handles.append(hand_actor_handle)

        hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
        hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        hand_props["stiffness"].fill(10)
        hand_props["damping"].fill(0.1)
        hand_props["effort"].fill(100.0)

        if self.hand_type in {"allegro_hand", "barrett"}:
            hand_props["stiffness"].fill(20)
            hand_props["damping"].fill(0.1)
        elif self.hand_type == "human_hand":
            hand_props["stiffness"].fill(8000)
            hand_props["damping"].fill(0)
        elif self.hand_type == "robotiq_3f":
            hand_props["stiffness"].fill(10000)
            hand_props["damping"].fill(0)

        gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
        dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
        for i, joint in enumerate(self.joint_names):
            joint_idx = gym.find_actor_dof_index(
                env,
                hand_actor_handle,
                joint,
                gymapi.DOMAIN_ACTOR,
            )
            dof_states["pos"][joint_idx] = float(hand_qpos[i])
        gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_states["pos"])

        hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
        hand_rigid_body_set = set()
        for i in range(gym.get_actor_rigid_body_count(env, hand_actor_handle)):
            hand_rigid_body_set.add(
                gym.get_actor_rigid_body_index(
                    env,
                    hand_actor_handle,
                    i,
                    gymapi.DOMAIN_ENV,
                )
            )
        self.hand_rigid_body_sets.append(hand_rigid_body_set)
        for shape_prop in hand_shape_props:
            shape_prop.friction = self.hand_friction
        gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)

        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0, 0, 0)
        obj_pose.r = gymapi.Quat(0, 0, 0, 1)
        obj_actor_handle = gym.create_actor(
            env,
            self.load_obj(obj_root, obj_file),
            obj_pose,
            "obj",
            0,
            1,
        )
        self.obj_handles.append(obj_actor_handle)
        gym.set_actor_scale(env, obj_actor_handle, float(obj_scale))

        obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
        obj_rigid_body_set = set()
        for i in range(gym.get_actor_rigid_body_count(env, obj_actor_handle)):
            obj_rigid_body_set.add(
                gym.get_actor_rigid_body_index(
                    env,
                    obj_actor_handle,
                    i,
                    gymapi.DOMAIN_ENV,
                )
            )
        self.obj_rigid_body_sets.append(obj_rigid_body_set)
        for shape_prop in obj_shape_props:
            shape_prop.friction = self.obj_friction
        gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)

    def run_sim(self, debug=False):
        if debug:
            gym.step_graphics(self.sim)
            gym.draw_viewer(self.viewer, self.sim, True)
            for _ in range(self.sim_step * 2):
                gym.draw_viewer(self.viewer, self.sim, True)
                gym.step_graphics(self.sim)
                sleep(self.debug_interval)

        for _ in range(self.sim_step):
            gym.simulate(self.sim)
            if self.has_viewer:
                if gym.query_viewer_has_closed(self.viewer):
                    break
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, True)
                sleep(self.debug_interval)

        success = []
        for i, env in enumerate(self.envs):
            flag = False
            for contact in gym.get_env_rigid_contacts(env):
                hand_first = (
                    contact[2] in self.hand_rigid_body_sets[i]
                    and contact[3] in self.obj_rigid_body_sets[i]
                )
                obj_first = (
                    contact[3] in self.hand_rigid_body_sets[i]
                    and contact[2] in self.obj_rigid_body_sets[i]
                )
                if hand_first or obj_first:
                    flag = True
                    break
            success.append(flag)
        return success

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)
