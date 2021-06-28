"""
experiment_tf1.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 1. TimeGAN model
from timegan import TimeGAN
# TODO add alternatives approaches.
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization, plot_loss

import argparse
import numpy as np
import warnings
import json
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main (args):
    """Main function for experiments.

    Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - model: model used for this experiment
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation

    Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
    """

    # Data loading
    data = []
    if args.data_name in ['stock', 'energy']:
        data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        data = sine_data_generation(no, args.seq_len, dim)

    print(args.data_name + ' dataset is ready.')

    # Synthetic data generation
    # Set network parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations_embedding'] = args.iteration_embedding
    parameters['iterations_supervised'] = args.iteration_supervised
    parameters['iterations_joint'] = args.iteration_joint
    parameters['batch_size'] = args.batch_size

    timegan = TimeGAN(data, parameters)
    generated, losses, times = timegan.train()

    print('Finish Synthetic Data Generation')

    # Performance metrics
    plot_loss(losses, args.model + '-' + args.data_name + '-' + str(args.iteration_supervised) + 'superviseditt-' + str(args.iteration_joint) + 'jointitt')

    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list() 
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(data, generated)
        discriminative_score.append(temp_disc)

    metric_results['discriminative-mean'] = np.mean(discriminative_score)
    metric_results['discriminative-std'] = np.std(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for _ in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(data, generated)
        predictive_score.append(temp_pred)

    metric_results['predictive-mean'] = np.mean(predictive_score)
    metric_results['predictive-std'] = np.std(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(data, generated, 'pca', args.model + '-' + args.data_name + '-' + str(args.iteration_supervised) + 'superviseditt-' + str(args.iteration_joint) + 'jointitt')
    visualization(data, generated, 'tsne', args.model + '-' + args.data_name + '-' + str(args.iteration_supervised) + 'superviseditt-' + str(args.iteration_joint) + 'jointitt')

    # Print discriminative and predictive scores
    print(metric_results)

    with open('results/metrics/' + args.model + '-' + args.data_name + '-' + str(args.iteration_supervised) + 'superviseditt-' + str(args.iteration_joint) + 'jointitt-metrics.json', 'w') as json_file:
        json.dump(metric_results, json_file, cls=NumpyEncoder)

    with open('results/metrics/' + args.model + '-' + args.data_name + '-' + str(args.iteration_supervised) + 'superviseditt-' + str(args.iteration_joint) + 'jointitt-time-metrics.json', 'w') as json_file:
        json.dump(times, json_file, cls=NumpyEncoder)

    return data, generated, metric_results


if __name__ == '__main__':  

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['timegan'],
        default='timegan',
        type=str)
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy'],
        default='stock',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration_embedding',
        help='Training iterations for embedding phase (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--iteration_supervised',
        help='Training iterations for supervised phase (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--iteration_joint',
        help='Training iterations for joint phase (should be optimized)',
        default=50000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)

    args = parser.parse_args()

    # Calls main function
    ori_data, generated_data, metrics = main(args)
