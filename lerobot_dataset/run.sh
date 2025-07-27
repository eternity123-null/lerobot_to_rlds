CUDA_VISIBLE_DEVICE=-1 python convert_lerobot_to_rlds.py --repo-id AgiBot/alpha --output-dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tmp/rlds_test \
     --local-files-only --root /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/demo_data/alpha > log.log 2>&1


CUDA_VISIBLE_DEVICE=-1 python convert_lerobot_to_rlds.py --repo-id tape1 --output-dir /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tmp/rlds_test/tape_test \
     --local-files-only --root /inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/mobile_aloha_lerobot/aloha_static_tape > convert1.log 2>&1
     
TFDS_DATA_DIR=/inspire/hdd/project/embodied-intelligence/xiaoyunxiao-240108120113/datasets/tensorflow_datasets tfds build --overwrite > build.log 2>&1