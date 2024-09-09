"""Main script to train and evaluate model

Launches training and evaluation according to parameters contained in file
parameters.json of given directory experiment_dir.

Parameters contained in parent directories are loaded recursively. Parameters
are overidden by command line arguments. Final parameters are written to
experiment_dir/parameters.json"""

import argparse
import os

import ml
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train according to parameters in DIRECTORY")
    parser.add_argument("--experiment_dir",
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

    completed_file_path = os.path.join(args.experiment_dir, "completed")
    if os.path.exists(completed_file_path):
        print(f"Training {args.experiment_dir} already completed")
        print(f"Delete {completed_file_path} to run again")

    exp = ml.loading.load_experiment(**args.__dict__)
    exp.evaluate("inference",
                    restore_file_name="last",
                    use_swa=exp.params.get("swa") is not None,
                    save_predictions=True)


# python inference.py --experiment_dir ./experiments/convs-m128-i224-wbce --data_dir ../emotion/audio/1.mp3