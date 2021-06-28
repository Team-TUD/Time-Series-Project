"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_test():
  xs = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
  ysd = [0.243, 0.250, 0.352, 0.153, 0.341, 0.210, 0.339, 0.263, 0.265, 0.376]
  ysp = [0.109, 0.093, 0.083, 0.097, 0.081, 0.080, 0.082, 0.089, 0.089, 0.085]

  plt.title('Discriminative score plot')
  plt.xlabel('iterations')
  plt.ylabel('discriminative score')
  plt.plot(xs, ysd)
  plt.savefig('results/visualization/' + 'discriminative-score-plot.png', dpi=300)
  plt.close()

  plt.title('Predictive score plot')
  plt.xlabel('iterations')
  plt.ylabel('predictive score')
  plt.plot(xs, ysp)
  plt.savefig('results/visualization/' + 'predictive-score-plot.png', dpi=300)
  plt.close()

def plot_loss(losses, name):

  for key, value in losses.items():
    print('Plotting ' + key + ' loss')
    plot(key, value, name)
    print('Finish plotting ' + key + ' loss')

def plot(key, value, name):
  plt.title(key + ' plot')
  plt.xlabel('iterations')
  plt.ylabel(key)
  plt.plot(value)
  plt.savefig('results/visualization/' + name + '-' + key + '-loss-plot.png', dpi=300)
  plt.close()

def plot_metrics(metrics, name, path):

  xs = list()
  ys = list()

  metrics.sort(key=lambda tup: tup[0])
  for (key, value) in metrics:
    xs.append(key)
    ys.append(value)

  print('Plotting ' + name + ' metric')
  plt.title(name + ' plot')
  plt.xlabel('iterations')
  plt.ylabel('predictive score')
  plt.plot(xs, ys)
  plt.savefig(path + '/' + name + '-plot.png', dpi=300)
  plt.close()
  print('Finish plotting ' + name + ' metric')


   
def visualization (ori_data, generated_data, analysis, name):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':

    print('Performing PCA analysis for ' + name)

    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot for ' + name)
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.savefig('results/visualization/' + name + '-pca-plot.png', dpi=300)
    plt.close()

    print('Finish PCA analysis for ' + name)

  elif analysis == 'tsne':

    print('Performing t-SNE analysis for ' + name)
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot for ' + name)
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.savefig('results/visualization/' + name + '-tsne-plot.png', dpi=300)
    plt.close()

    print('Finish t-SNE analysis for ' + name)


def alternative_visualization(ori_data, generated_data, seq_len, name):
  sample_size = 250
  idx = np.random.permutation(len(ori_data))[:sample_size]

  real_sample = np.asarray(ori_data)[idx]
  synthetic_sample = np.asarray(generated_data)[idx]

  synth_data_reduced = real_sample.reshape(-1, seq_len)
  ori_data_reduced = np.asarray(synthetic_sample).reshape(-1, seq_len)

  n_components = 2
  pca = PCA(n_components=n_components)
  tsne = TSNE(n_components=n_components, n_iter=300)

  # The fit of the methods must be done only using the real sequential data
  pca.fit(ori_data_reduced)

  pca_real = pd.DataFrame(pca.transform(ori_data_reduced))
  pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

  data_reduced = np.concatenate((ori_data_reduced, synth_data_reduced), axis=0)
  tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

  # PCA plot

  fig = plt.figure(constrained_layout=True, figsize=(20, 10))
  spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

  # TSNE scatter plot
  ax = fig.add_subplot(spec[0, 0])
  ax.set_title('PCA results',
               fontsize=20,
               color='red',
               pad=10)

  # PCA scatter plot
  plt.scatter(pca_real.iloc[:, 0].values, pca_real.iloc[:, 1].values,
              c='black', alpha=0.2, label='Original')
  plt.scatter(pca_synth.iloc[:, 0], pca_synth.iloc[:, 1],
              c='red', alpha=0.2, label='Synthetic')
  ax.legend()

  ax2 = fig.add_subplot(spec[0, 1])
  ax2.set_title('TSNE results',
                fontsize=20,
                color='red',
                pad=10)

  plt.scatter(tsne_results.iloc[:sample_size, 0].values, tsne_results.iloc[:sample_size, 1].values,
              c='black', alpha=0.2, label='Original')
  plt.scatter(tsne_results.iloc[sample_size:, 0], tsne_results.iloc[sample_size:, 1],
              c='red', alpha=0.2, label='Synthetic')

  ax2.legend()
  plt.savefig('results/visualization/' + name + '-pca-tsne-plot.png', dpi=300)