import math
from os import path
import argparse
import numpy as np
import json

from alternative_approaches.timegan_ydata.src.ydata_synthetic.synthesizers.timeseries import TimeGAN
from data_loading import real_data_loading, sine_data_generation
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import alternative_visualization, visualization

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(args):

    seq_len = args.seq_len
    hidden_dim = args.hidden_dim
    scale_train = args.scale_train
    gamma = 1

    noise_dim = 32
    dim = 0
    batch_size = args.batch_size

    log_step = 100
    learning_rate = 5e-4

    data = list()
    if args.data_name in ['stock', 'energy']:
        if args.data_name in ['stock']:
            dim = 6
        else:
            dim = 28
        print('Using ' + args.data_name + 'data set.')
        data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        print('Using sine data set.')
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        data = sine_data_generation(no, args.seq_len, dim)

    gan_args = [batch_size, learning_rate, noise_dim, 24, 2, (0, 1), dim]

    if args.trained_model != 'none' and path.exists(args.trained_model):
        print('Using trained model.')
        synth = TimeGAN.load(args.trained_model)
        output(data, seq_len, synth, args.model + '-' + args.data_name, args.metric_iteration)
    else:
        synth = TimeGAN(model_parameters=gan_args, hidden_dim=hidden_dim, seq_len=seq_len, n_seq=dim, gamma=gamma, scale_train=scale_train)

        iterations = args.iteration
        save_every_iterations = args.save_itt

        if save_every_iterations > 0 and False:
            print('Saving model every ' + str(save_every_iterations) + ' iterations.')
            rounds = math.ceil(iterations / save_every_iterations)
            iterations = math.ceil(iterations / rounds)
        else:
            rounds = 1

        for r in range(rounds):
            itt = (r + 1) * iterations
            # synth.train(stock_data, train_steps=5)
            synth.train(data, train_steps=iterations)
            synth.save('results/model/' + args.model + '-' + args.data_name + '-' + str(itt) + 'itt.pkl')

            output(data, seq_len, synth, args.model + '-' + args.data_name + '-' + str(itt) + 'itt', args.metric_iteration)


def output(ori_data, seq_len, model, name, metric_iteration):
    synth_data = model.sample(len(ori_data))[:len(ori_data)]
    print(len(synth_data))

    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, synth_data)
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, synth_data)
        predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)
    print(metric_results)

    with open('results/metrics/' + name + '-metrics.json', 'w') as json_file:
        json.dump(metric_results, json_file, cls=NumpyEncoder)

    # 3. Visualization (PCA and t-SNE)
    visualization(ori_data, synth_data, 'pca', name)
    visualization(ori_data, synth_data, 'tsne', name)

    alternative_visualization(ori_data, synth_data, seq_len, name)

if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model',
      choices=['timegan'],
      default='timegan',
      type=str)
    parser.add_argument(
        '--save_itt',
        default=0,
        type=int)
    parser.add_argument(
        '--trained_model',
        default='none',
        type=str)
    parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy'],
      default='stock',
      type=str)
    parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
    parser.add_argument(
        '--scale_train',
        help='scaled training',
        default=1,
        type=int)
    parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
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
      '--iteration',
      help='Training iterations (should be optimized)',
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
    main(args)