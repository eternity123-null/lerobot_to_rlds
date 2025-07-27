

from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
from PIL import Image
sys.path.append('.')
from .conversion_utils import MultiThreadedDatasetBuilder
import os
tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

# Agibot dataformat
# <KeysViewHDF5 ['action', 'hand_left', 'hand_right', 'head', 'language_instruction', 'state', 'timestamps']>
# <HDF5 dataset "action": shape (528, 36), type "<f8">
# <HDF5 dataset "hand_left": shape (528, 480, 640, 3), type "|u1">
# <HDF5 dataset "hand_right": shape (528, 480, 640, 3), type "|u1">
# <HDF5 dataset "head": shape (528, 480, 640, 3), type "|u1">
# <HDF5 dataset "language_instruction": shape (528,), type "|O">
# <HDF5 dataset "state": shape (528, 55), type "<f8">
# <HDF5 dataset "timestamps": shape (528,), type "<i8">

# Action and state meaning:
#  SELECTED_FEATS = [
#         # Action features (will be concatenated into 36-dimensional vector)
#         ["action", "effector", "position"],        # Output columns 0-1
#         ["action", "end", "orientation"],          # Columns 2-9
#         ["action", "end", "position"],            # Columns 10-15
#         ["action", "head", "position"],            # Columns 16-17
#         ["action", "joint", "position"],           # Columns 18-31
#         ["action", "robot", "velocity"],           # Columns 32-33
#         ["action", "waist", "position"],           # Columns 34-35
        
#         # State features (will be concatenated into 55-dimensional vector)
#         ["state", "effector", "position"],         # Output columns 0-1
#         ["state", "end", "orientation"],           # Columns 2-9
#         ["state", "end", "position"],              # Columns 10-15
#         ["state", "head", "position"],             # Columns 16-17
#         ["state", "joint", "current_value"],       # Columns 18-31
#         ["state", "joint", "position"],            # Columns 32-45
#         ["state", "robot", "orientation"],         # Columns 46-49
#         ["state", "robot", "position"],            # Columns 50-52
#         ["state", "waist", "position"]             # Columns 53-54
#     ]

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path):
        # Load raw data
        with h5py.File(episode_path, "r") as F:
            # breakpoint()
            actions = F['action'][:]
            states = F["state"][:]
            images = F["head"][:]  # Primary camera (top-down view) (528, 480, 640, 3) HWC uint8
            left_wrist_images = F["hand_left"][:]  # Left wrist camera
            right_wrist_images = F["hand_right"][:]  # Right wrist camera
            language_instruction = F["language_instruction"][:]

        frames = {
            'actions': actions.shape[0],
            'states': states.shape[0],
            'images': images.shape[0],
            'left_wrist_images': left_wrist_images.shape[0],
            'right_wrist_images': right_wrist_images.shape[0],
            'language_instruction': language_instruction.shape[0],
        }
        values = list(frames.values())
        first_value = values[0]
        all_equal = all(v == first_value for v in values)
        if not all_equal:
            print(episode_path," :数据的第一个维度不一致！", flush=True)
            for k, v in frames.items():
                print(f"{k}: {v}", flush=True)
            return None

        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            # state: [gripper_2, joint_pos_14, sin(yaw), cos(yaw), x , y , waist_2]
            # action: [gripper, joint_pos, linear_vel, yaw_rate, waist_2]
            episode.append({
                'observation': {
                    'image': np.array(Image.fromarray(images[i]).resize((256,256), Image.BILINEAR)),
                    'left_wrist_image': np.array(Image.fromarray(left_wrist_images[i]).resize((256,256), Image.BILINEAR)),
                    'right_wrist_image': np.array(Image.fromarray(right_wrist_images[i]).resize((256,256), Image.BILINEAR)), 
                    'state': np.concatenate((states[i][0:2], states[i][32:46], states[i][48:52],states[i][53:55]),axis=0).astype(np.float32),
                },
                'action': np.concatenate((actions[i][0:2],actions[i][18:36]),axis=0).astype(np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': str(language_instruction[i]),
            })

        # Create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # If you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # For smallish datasets, use single-thread parsing
    for sample in paths:
        ret = _parse_example(sample)
        yield ret


class agibot_pour(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 120           # number of parallel workers for data conversion #250
    MAX_PATHS_IN_MEMORY = 3500   # number of paths converted & stored in memory before writing to disk #3500
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            # shape=(96, 96, 3),
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'left_wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Left wrist camera RGB observation.',
                        ),
                        'right_wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Right wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(22,),
                            dtype=np.float32,
                            doc='Robot state [gripper_2, joint_pos_14, x,y, sin(yaw), cos(yaw), , waist_2].',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(20,),
                        dtype=np.float32,
                        doc='Robot gripper + arm action + linear_vel + yaw_rate + waist.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        data_root_path = "/inspire/hdd/global_user/xiaoyunxiao-240108120113/zcd/processed_data/Pour"
        all_files = [
            os.path.join(data_root_path, f) 
            for f in os.listdir(data_root_path) 
            if f.endswith('.hdf5')
        ]
        np.random.seed(2025)  # 固定随机种子以保证可重复性
        np.random.shuffle(all_files)
        split_idx = min(int(len(all_files) * 0.95), len(all_files)-2)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        # print("train_files:\n",train_files)
        return {
            "train": train_files,
            "val": val_files,
        }
