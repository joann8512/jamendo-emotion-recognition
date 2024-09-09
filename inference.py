import os
import argparse

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
    if not os.path.exists(completed_file_path):
        print(f"Can't find completed model of {args.experiment_dir}")

    exp = ml.loading.load_experiment(**args.__dict__)
    exp.evaluate("inference",
                    restore_file_name="last",
                    use_swa=exp.params.get("swa") is not None,
                    save_predictions=True)

# python inference.py --experiment_dir ./experiments/convs-m128-i224-wbce --data_dir ../emotion/audio/1.mp3