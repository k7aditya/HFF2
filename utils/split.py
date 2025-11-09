import os
import random
import numpy as np
from utils import check_dir

def correct_path(path):
    return path.replace('\\', '/').replace('../dataset', './dataset')


def split(brain_dir, save_dir, n_split=1, seed=9418):
    """
    Splits the dataset into Train (80%), Validation (10%), and Test (10%)
    and saves their file paths into separate text files for each split.
    """

    # Get all patient folders under the dataset directory
    sample_dir = os.path.join(brain_dir)
    patient_ids = os.listdir(sample_dir)
    np.random.seed(seed)
    random.seed(seed)

    check_dir(save_dir)
    total = len(patient_ids)

    for split_index in range(n_split):
        # Shuffle patient order
        np.random.shuffle(patient_ids)

        # Compute split sizes
        n_train = int(total * 0.8)
        n_val = int(total * 0.1)
        n_test = total - n_train - n_val

        # Split the data
        train_ids = patient_ids[:n_train]
        val_ids = patient_ids[n_train:n_train + n_val]
        test_ids = patient_ids[n_train + n_val:]

        # Define output text files
        train_file_path = os.path.join(save_dir, f'{split_index}-train.txt')
        val_file_path = os.path.join(save_dir, f'{split_index}-val.txt')
        test_file_path = os.path.join(save_dir, f'{split_index}-test.txt')

        # Write train file
        with open(train_file_path, 'w') as f:
            for pid in train_ids:
                path = os.path.join(sample_dir, pid)
                f.write(f"{correct_path(path)}\n")

        # Write val file
        with open(val_file_path, 'w') as f:
            for pid in val_ids:
                path = os.path.join(sample_dir, pid)
                f.write(f"{correct_path(path)}\n")

        # Write test file
        with open(test_file_path, 'w') as f:
            for pid in test_ids:
                path = os.path.join(sample_dir, pid)
                f.write(f"{correct_path(path)}\n")

        print(f"✅ Split {split_index}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    print("\nAll splits created successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Split BraTS dataset into train/val/test")
    parser.add_argument('--brain_dir', type=str,
                        default='/teamspace/studios/this_studio/HFF/BraTS2020_TrainingData3/MICCAI_BraTS2020_TrainingData')
    parser.add_argument('--save_dir', type=str,
                        default='/teamspace/studios/this_studio/HFF/brats20')
    parser.add_argument('--n_split', type=int, default=1)
    parser.add_argument('--seed', type=int, default=9418)
    args = parser.parse_args()

    split(args.brain_dir, args.save_dir, n_split=args.n_split, seed=args.seed)
