import os
import argparse

import ml
import numpy as np
from ml.inference_utils import *
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train according to parameters in DIRECTORY")
    parser.add_argument("--experiment_dir",
                        nargs="*",
                        type=str,
                        metavar="DIRECTORY",
                        help="path of the directory containing parameters")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data",
                        metavar="DATA",
                        help="""path of the directory containing data (default:
                        data)""")
    parser.add_argument("--num_workers",
                        type=int,
                        default=4,
                        metavar="NUM",
                        help="number of workers for dataloader (default: 4)")
    parser.add_argument("--restore_name",
                        type=str,
                        default="last",
                        metavar="NAME",
                        help="name of checkpoint to restore (default: last)")
    parser.add_argument("--batch_size",
                        type=int,
                        metavar="BS",
                        help="override batch size")
    parser.add_argument("--manual_seed",
                        type=int,
                        metavar="SEED",
                        help="override manual seed")
    parser.add_argument("--model", type=str, help="override model")
    parser.add_argument("--calculate_stats",
                        dest="calculate_stats",
                        action="store_true",
                        default=None,
                        help="""recalculate  mean and std of data (default is to
                        calculate only when they don't exist in parameters)""")

    args = parser.parse_args()

    #probs = np.array([], dtype="float32")
    probs = []
    for path in args.experiment_dir:  # all models
        completed_file_path = os.path.join(path, "completed")
        if not os.path.exists(completed_file_path):
            print(f"Can't find completed model of {path}")

        exp = ml.loading.load_experiment(experiment_dir=path,
                                         data_dir=args.data_dir)
        out_probs = exp.evaluate("inference",
                        restore_file_name="last",
                        use_swa=exp.params.get("swa") is not None,
                        save_predictions=True)

        #probs = np.vstack((probs, out_probs.numpy()))
        probs.append(out_probs.numpy())
    probs = np.array(probs)
    top_3_indices = np.argsort(probs.mean(axis=0))[-3:]  # ascending
    binarized = np.zeros_like(probs.mean(axis=0))
    binarized[top_3_indices] = 1
    out_tags = []
    for i in np.flip(top_3_indices):
        out_tags.append(TAG_MAP[str(i)])
    print(out_tags)

# python inference.py --experiment_dir ./experiments/convs-m128-i224-wbce --data_dir ../emotion/audio/1.mp3