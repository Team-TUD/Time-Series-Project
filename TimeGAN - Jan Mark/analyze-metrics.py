import argparse
import json
import os
from metrics.visualization_metrics import plot_metrics


def main(args):
    # define the name of the directory to be created
    path = args.path + '/analyze'

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    predictive = list()
    discriminative = list()

    for filename in os.listdir(args.path + '/metrics/'):
        if filename.endswith(".json") and "itt" in filename:
            with open(args.path + '/metrics/' + filename) as w:
                x = json.load(w)
                discriminative.append((int(filename.split("-")[1].rstrip("itt")), x["discriminative"]))
                predictive.append((int(filename.split("-")[1].rstrip("itt")), x["predictive"]))
        else:
            continue

    plot_metrics(predictive, "predictive", path)
    plot_metrics(discriminative, "discriminative", path)


if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        default='results',
        type=str)

    args = parser.parse_args()
    main(args)


