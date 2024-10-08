import argparse
import logging
import warnings

from autoimblearn import AutoImblearn
from runpipe import RunPipe

logging.basicConfig(filename='cvd.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #  parser.add_argument('--dataset', type=str, default="nhanes.csv")
    parser.add_argument('--dataset', type=str, default="pima-indians-diabetes-missing.csv")
    parser.add_argument('--target', default="Status", type=str)  # Set the name of the prediction target

    #
    # Model parameters
    #
    parser.add_argument('--T_model', default="lr",
                        choices=["SVM", 'LSVM', 'lr', 'rf', 'mlp', 's2sl', 's2sLR', 'ensemble', 'ada',
                                 'bst'])  # Traditional models

    parser.add_argument('--repeat', default=0, type=int)

    #
    # Pre-Processing
    #
    parser.add_argument('--aggregation', default="binary", choices=["categorical", "binary"])

    parser.add_argument('--missing', default='median', choices=['median', 'mean', 'dropna', 'knn', 'ii', 'gain', 'MIRACLE', 'MIWAE'],
                        type=str)  # Handle null values

    # K-Fold
    parser.add_argument('--n_splits', default=10, type=int)  # Number of split in for K-fold

    # Resample related
    parser.add_argument('--infor_method', default='normal', choices=['normal', 'nothing'])  # Choose how to handle AUDM

    parser.add_argument('--resampling', default=False, action="store_true")
    parser.add_argument('--resample_method', default="under",
                        choices=['under', 'over', 'combined', 'herding', 's2sl_mwmote', 'MWMOTE', "smote"])
    parser.add_argument('--samratio', default=0.4, type=float)  # target sample ratio

    # Feature Importance
    parser.add_argument('--feature_importance', default='NA', choices=['NA', 'lime', 'shap'],
                        type=str)  # Which model to use

    # GridSearchCV
    parser.add_argument('--grid', default=False, action="store_true")  # Use Grid search to find best hyper-parameter

    # top k feature
    parser.add_argument('--top_k', default=-1, type=int)  # The number of features to keep

    # Auto-Imblearn related
    parser.add_argument('--train_ratio', default=1.0, type=float)  # Only use certain ratio of dataset
    parser.add_argument('--metric', default='auroc', choices=['auroc', 'macro_f1'], type=str)  # Determine the metric
    # parser.add_argument('--rerun', default=False, action="store_true")  # Re-run the best pipeline found with 100% data
    parser.add_argument('--rerun', default=False, action="store_true")  # Re-run the best pipeline found with 100% data
    parser.add_argument('--exhaustive', default=False, action="store_true") # run exhaustive search instead of AutoImblearn

    args = parser.parse_args()

    logging.info("-------------------------")

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument {}: {}".format(arg, value))

    # Load the data
    run_pipe = RunPipe(args=args)
    run_pipe.loadData()

    # Run Auto-Imblearn to find best pipeline
    checked = {}
    automl = AutoImblearn(run_pipe, metric=args.metric)

    if args.exhaustive:
        print("exhaustive search")
        automl.exhaustive_search(checked=checked, train_ratio=args.train_ratio)

    else:
        best_pipe, counter, best_score = automl.find_best(checked=checked, train_ratio=args.train_ratio)

        print("Final result:", best_pipe, args.metric, counter, end=" ")
        if args.train_ratio != 1.0 and args.rerun:
            # Re-run the best pipeline with whole dataset to get the output score
            print("Re-running best pipeline")
            best_score = automl.run_best(best_pipe)

        print(best_score)
        best_pipe = list(best_pipe)
        logging.info("Final result. Best pipe: {}, {}, {}, counter: {}, best score: {}".format(best_pipe[0], best_pipe[1],
                                                                                               best_pipe[2], counter,
                                                                                               best_score))
