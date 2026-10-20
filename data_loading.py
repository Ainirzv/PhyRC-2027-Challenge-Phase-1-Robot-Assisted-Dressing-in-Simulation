import sys
sys.path.insert(0, 'src/PhyRC_Sim')
from Policy.dataset_reader import PolicyDataset

path = 'output/policy_datasets/20261020T055633_377126Z_dc1b2c/policy.hdf5'
dataset = PolicyDataset(path, observation_keys=[
    'overview_rgb', 'robot_0_wrist_rgb', 'robot_1_wrist_rgb',
    'robot_0_head_rgb', 'robot_1_head_rgb',
    'joint_position', 'joint_velocity', 'previous_action',
])
obs, action = dataset[0]
# obs['overview_rgb']: uint8 (256,256,3)
# obs['joint_position']: float32 (26,)
# action: float32 (18,), robot_0's 9 values followed by robot_1's 9
print(len(dataset), action.shape)
dataset.close()