import h5py
import sys
import numpy as np
with h5py.File("/inspire/ssd/project/embodied-intelligence/public/jintian/WAIC/UniDomain/datasets/processed_data/agibot/Scoop/episode_0729367.hdf5", "r") as F:
    print(F.keys())
    print(F['hand_left'])
    print(F['head'])
    print(F['state'])
    print(F['action'])
    
    language_instruction = F['language_instruction'][:]
    
    # a=[0,1,2,3,4,5,6]
    # print(a[2:])
    print(type(str(language_instruction[0])))
    a=F['action'][:10]
    print(a)
    print("\n \n ")
    print(np.diff(np.array(a), axis=0))
    # print(language_instruction[:5].shape[0])
    # actions = F['frames']['action'][:]
    # states = F['frames']["state"][:]
    # images = F['frames']["observation_images_cam_high"][:]  # Primary camera (top-down view)
    # print("images shape:", images.shape)
    # left_wrist_images = F['frames']["observation_images_cam_left_wrist"][:]  # Left wrist camera
    # right_wrist_images = F['frames']["observation_images_cam_right_wrist"][:]  # Right wrist camera
    # low_cam_images = F['frames']["observation_images_cam_low"][:]  # Low third-person camera
    # # reward = F['frames']["reward"][:]
    # language_instruction = F['frames']["language_instruction"][:]
    # print("instruction:",language_instruction)
    # print(type(language_instruction))

sys.exit()

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
            language_instruction = str(F['frames']["language_instruction"][:])

        # Assemble episode: here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            
            episode.append({
                'observation': {
                    'image': images[i],
                    'image': np.array(Image.fromarray(images[i]).resize((256,256), Image.BILINEAR)),
                    'left_wrist_image': np.array(Image.fromarray(left_wrist_images[i]).resize((256,256), Image.BILINEAR)),
                    'right_wrist_image': np.array(Image.fromarray(right_wrist_images[i]).resize((256,256), Image.BILINEAR)), 
                    'state': np.asarray(states[i][18:32], np.float32),
                },
                'action': np.asarray(actions[i][18:32], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': language_instruction[i],
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


class lerobot_dataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 168            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 840   # number of paths converted & stored in memory before writing to disk
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
                            shape=(14,),
                            dtype=np.float32,
                            doc='Robot joint state (7D left arm + 7D right arm).',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(14,),
                        dtype=np.float32,
                        doc='Robot arm action.',
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
        data_root_path = "/inspire/ssd/project/embodied-intelligence/public/jintian/WAIC/UniDomain/datasets/processed_data/agibot/Scoop"
        all_files = [
            os.path.join(data_root_path, f) 
            for f in os.listdir(data_root_path) 
            if f.endswith('.h5')
        ]
        np.random.seed(2025)  # 固定随机种子以保证可重复性
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * 0.95)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]
        return {
            "train": train_files,
            "val": val_files,
        }
