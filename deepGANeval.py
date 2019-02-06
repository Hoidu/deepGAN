# import python libraries
import os, sys
import argparse as ap
import datetime as dt
import time as tm

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# import tensorboard libraries
from tensorboardX import SummaryWriter

# importing data science libraries
import pandas as pd
import numpy as np
from sklearn import cluster

# import plotting libraries
import matplotlib
matplotlib.use('Agg')

# import project libraries
from UtilsHandler import UtilsHandler
from NetworksHandler import EncoderLinear
from NetworksHandler import EncoderSigmoid
from NetworksHandler import EncoderReLU
from NetworksHandler import Decoder
from NetworksHandler import Discriminator
from VisualizationHandler import ChartPlots

# init utilities handler
uha = UtilsHandler.UtilsHandler()
vha = ChartPlots.ChartPlots(plot_dir='./')

########################################################################################################################
# parse and set model parameters
########################################################################################################################

# init and prepare argument parser
parser = ap.ArgumentParser()

parser.add_argument('--exp_timestamp', help='', nargs='?', type=str, default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('--experiment_dir', help='', nargs='?', type=str, default='./01_experiments/2019-02-06_15-14-07_deepGAN_exp_sd_1234_ep_5_gs_2_rd_0.8_sd_0.015_mb_128_eco_linear_gpu_False')
parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)
parser.add_argument('--no_epochs', help='', nargs='?', type=int, default=5)
parser.add_argument('--eval_epochs', help='', nargs='?', type=int, default=50)
parser.add_argument('--enc_output', help='', nargs='?', type=str, default='relu')
parser.add_argument('--eval_latent_epochs', help='', nargs='?', type=int, default=500)
parser.add_argument('--learning_rate_enc', help='', nargs='?', type=float, default=1e-4)
parser.add_argument('--learning_rate_dec', help='', nargs='?', type=float, default=1e-4)
parser.add_argument('--learning_rate_dis', help='', nargs='?', type=float, default=1e-6)
parser.add_argument('--mini_batch_size', help='', nargs='?', type=int, default=128)
parser.add_argument('--no_gauss', help='', nargs='?', type=int, default=2)
parser.add_argument('--radi_gauss', help='', nargs='?', type=float, default=0.8)
parser.add_argument('--stdv_gauss', help='', nargs='?', type=float, default=0.015)
parser.add_argument('--eval_batch', help='', nargs='?', type=int, default=100)
parser.add_argument('--use_anomalies', help='', nargs='?', type=bool, default=False)
parser.add_argument('--use_cuda', help='', nargs='?', type=bool, default=False)
parser.add_argument('--base_dir', help='', nargs='?', type=str, default='./01_experiments')

# parse script arguments
experiment_parameter = vars(parser.parse_args())

# parse float args as float
experiment_parameter['learning_rate_enc'] = float(experiment_parameter['learning_rate_enc'])
experiment_parameter['learning_rate_dec'] = float(experiment_parameter['learning_rate_dec'])
experiment_parameter['learning_rate_dis'] = float(experiment_parameter['learning_rate_dis'])
experiment_parameter['radi_gauss'] = float(experiment_parameter['radi_gauss'])
experiment_parameter['stdv_gauss'] = float(experiment_parameter['stdv_gauss'])

# parse integer args as integer
experiment_parameter['seed'] = int(experiment_parameter['seed'])
experiment_parameter['no_epochs'] = int(experiment_parameter['no_epochs'])
experiment_parameter['eval_epochs'] = int(experiment_parameter['eval_epochs'])
experiment_parameter['eval_latent_epochs'] = int(experiment_parameter['eval_latent_epochs'])
experiment_parameter['mini_batch_size'] = int(experiment_parameter['mini_batch_size'])
experiment_parameter['no_gauss'] = int(experiment_parameter['no_gauss'])
experiment_parameter['eval_batch'] = int(experiment_parameter['eval_batch'])

# parse boolean args as boolean
#experiment_parameter['enc_linear'] = uha.str2bool(experiment_parameter['enc_linear'])
experiment_parameter['use_cuda'] = uha.str2bool(experiment_parameter['use_cuda'])
experiment_parameter['use_anomalies'] = uha.str2bool(experiment_parameter['use_anomalies'])

# init deterministic seeds
seed_value = 1234
np.random.seed(seed_value) # set numpy seed
torch.manual_seed(seed_value) # set pytorch seed CPU
torch.cuda.manual_seed(seed_value) # set pytorch seed GPU

# autoencoder architecture
encoder_hidden = [256, 64, 16, 2]
decoder_hidden = [2, 16, 64, 256]

# discriminator architecture
d_input = 2
d_hidden = [256, 64, 16]
d_output = 1

# create the experiment directory
par_dir = os.path.join(experiment_parameter['experiment_dir'], '00_param')
sta_dir = os.path.join(experiment_parameter['experiment_dir'], '01_statistics')
enc_dir = os.path.join(experiment_parameter['experiment_dir'], '02_encodings')
plt_dir = os.path.join(experiment_parameter['experiment_dir'], '03_visuals')
mod_dir = os.path.join(experiment_parameter['experiment_dir'], '04_models')
prd_dir = os.path.join(experiment_parameter['experiment_dir'], '05_predictions')
evl_dir = os.path.join(experiment_parameter['experiment_dir'], '06_evaluations')

# determine if evaluation dir exists
if not os.path.exists(evl_dir):

    # crete evaluation directory
    os.makedirs(evl_dir)

# set plot directory
vha.set_plot_dir(evl_dir)

########################################################################################################################
# load and pre-process the training data
########################################################################################################################

# set the data dir
data_dir = './00_datasets/fraud_dataset_v2.csv'

# read the transactional data
transactions = pd.read_csv(data_dir, sep=',', encoding='utf-8')

# case: anomalies disabled
if experiment_parameter['use_anomalies'] is False:

    # remove anomalies, keep regular transactions only
    transactions = transactions[transactions['label'] == 'regular']

# remove the label column of the transactional data
y = transactions.pop('label')

# copy the transactional data
x = transactions.copy()

# filter transactions for company codes
bukrs = ['C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17']
transactions_filtered = x[x['BUKRS'].isin(bukrs)]

# reset index of filtered transactions
transactions_filtered.index = range(0, transactions_filtered.shape[0])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN::Data loading, transactional data of shape {} rows and {} columns successfully loaded.'.format(now, str(transactions.shape[0]), str(transactions.shape[1])))

###### pre-process categorical attributes

# determine the categorical attributes
cat_attr = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR',	'BSCHL', 'HKONT']

# extract categorical attributes
cat_transactions = transactions_filtered[cat_attr].copy()

# encode categorical attributs into one-hot
encoded_cat_transactions = pd.get_dummies(cat_transactions)

###### pre-process numerical attributes

# determine the numerical attributes
numeric_attr = ['DMBTR', 'WRBTR']

# extract the numerical attributes
num_transactions = transactions_filtered[numeric_attr].copy()

# log-transform the numerical attributes
encoded_num_transactions = (num_transactions + 0.0001).apply(np.log)

# normalized the numerical attributes
encoded_num_transactions = (encoded_num_transactions - encoded_num_transactions.min()) / (encoded_num_transactions.max() - encoded_num_transactions.min())

###### merge encoded categorical and numerical attributes

# stack column-wise one-hot form and normalized numeric form
enc_transactions = pd.concat([encoded_cat_transactions, encoded_num_transactions], axis=1).values.astype(np.float32)

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN::Data loading, encoded transactional data of shape {} rows and {} columns successfully created.'.format(now, str(enc_transactions.shape[0]), str(enc_transactions.shape[1])))

# convert encoded transactions to torch tensor
enc_transactions = torch.FloatTensor(enc_transactions)

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push encoded transactions to cuda
    enc_transactions = enc_transactions.cuda()

# init the network training data loader
dataloader = DataLoader(dataset=enc_transactions, batch_size=experiment_parameter['mini_batch_size'], shuffle=True)

# determine total number of batches
no_batches = int(enc_transactions.shape[0] / experiment_parameter['mini_batch_size'])

# determine encoded columns
encoded_columns = [str(r) for r in encoded_cat_transactions]
encoded_columns.extend(['DMBTR', 'WRBTR'])

# =================== help functions ============================
# sample Guassian distribution(s) of a particular shape
def get_target_distribution(mu, sigma, n, dim=2):

    # determine samples per gaussian
    samples_per_gaussian = int(n / len(mu))

    # determine sample delta
    samples_delta = n - (len(mu) * samples_per_gaussian)

    # iterate over all gaussians
    for i, mean in enumerate(mu):

        # case: first gaussian
        if i == 0:

            # randomly sample from gaussion distribution + samples delta
            z_samples_all = np.random.normal(mean, sigma, size=(samples_per_gaussian + samples_delta, dim))

        # case: non-first gaussian
        else:

            # randomly sample from gaussion distribution + samples delta
            z_samples = np.random.normal(mean, sigma, size=(samples_per_gaussian, dim))

            # stack new samples
            z_samples_all = np.vstack([z_samples_all, z_samples])

    # return sampled data
    return z_samples_all

# split matrix: one-hot and numerical parts
def split_2_dummy_numeric(m, slice_index):
  dummy_part = m[:, :slice_index]
  numeric_part = m[:, slice_index:]
  return dummy_part, numeric_part

# calculate the purity score of a cluster
def get_purity(cluster_labels, input_data):

  def purity_score(cluster):
    # given the cluster compute cluster mean
    cluster_mean = np.mean(cluster, axis=0)
    # average distance of the cluster points to the cluster mean
    avg_distance = np.mean(np.sqrt(np.sum((cluster - cluster_mean)**2, axis=1)))
    # return the distance penalized by the number of cluster points
    return (avg_distance / np.shape(cluster)[0])
    #return (avg_distance)# / np.shape(cluster)[0])

  purity = []
  cluster_points = []

  for cluster_label in range(experiment_parameter['no_gauss']):

    cluster = input_data[cluster_labels==cluster_label]
    # sometimes clusters might be empty

    if len(cluster) == 0:
        purity.append(0)
        cluster_points.append(0)

    else:
        purity.append(purity_score(cluster))
        cluster_points.append(len(cluster))

  return purity, cluster_points


########################################################################################################################
# init and load pre-trained encoder model
########################################################################################################################

# case: linear encoder enabled
if experiment_parameter['enc_output'] == 'linear':

    # init encoder model
    encoder_eval = EncoderLinear.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'sigmoid':

    # init encoder model
    encoder_eval = EncoderSigmoid.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'relu':

    # init encoder model
    encoder_eval = EncoderReLU.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

# determine encoder number of parameters
encoder_parameter = uha.get_network_parameter(net=encoder_eval)

# load trained encoder model file from disk
encoder_model_name = 'ep_{}_encoder_model.pkl'.format(str(experiment_parameter['no_epochs']).zfill(4))
encoder_eval.load_state_dict(torch.load(os.path.join(mod_dir, encoder_model_name), map_location=lambda storage, location: storage))

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Encoder architecture loaded: {}.'.format(now, str(encoder_eval)))
print('[INFO {}] DeepGAN:: No. of Encoder architecture parameters: {}.'.format(now, str(encoder_parameter)))

# init decoder model
decoder_eval = Decoder.Decoder(hidden_size=decoder_hidden, output_size=enc_transactions.shape[1])

# determine decoder number of parameters
decoder_parameter = uha.get_network_parameter(net=decoder_eval)

# load trained encoder model file from disk
decoder_model_name = 'ep_{}_decoder_model.pkl'.format(str(experiment_parameter['no_epochs']).zfill(4))
decoder_eval.load_state_dict(torch.load(os.path.join(mod_dir, decoder_model_name), map_location=lambda storage, location: storage))

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Decoder architecture established: {}.'.format(now, str(decoder_eval)))
print('[INFO {}] DeepGAN:: No. of Decoder architecture parameters: {}.'.format(now, str(decoder_parameter)))

########################################################################################################################
# evaluate latent space feature distribution (transactions)
########################################################################################################################

# set network in training mode
encoder_eval.eval()

# determine latent space representation of all transactions
z_enc_transactions = encoder_eval(enc_transactions)

# convert latent space representation to numpy
z_enc_transactions = z_enc_transactions.cpu().data.numpy()

# visualize latent space
title = 'Epoch {} Latent Space Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space(z_representation=z_enc_transactions, title=title, filename='01_latent_space_ep_{}_bt_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), color='C1'))

# convert encodings to pandas data frame
cols = ['x', 'y']
df_z_enc_transactions = pd.DataFrame(z_enc_transactions, columns=cols)

# merge embeddings with actual transactions
embedded_transactions = pd.concat([transactions_filtered, df_z_enc_transactions], axis=1, sort=False)

# save encodings data frame to file directory
filename = '01_embedded_eval_ep_{}.csv'.format(str(experiment_parameter['no_epochs']).zfill(4))
embedded_transactions.to_csv(os.path.join(evl_dir, filename), sep=';', index=False, encoding='utf-8')

# visualize latent space
feature = 'KTOSL'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

# visualize latent space
feature = 'WAERS'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

# visualize latent space
feature = 'BUKRS'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

# visualize latent space
feature = 'PRCTR'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

# visualize latent space
feature = 'BSCHL'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

# visualize latent space
feature = 'HKONT'
title = 'Epoch {} Latent Space Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename='01_latent_space_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature)))

########################################################################################################################
# evaluate latent space feature distribution (entire space)
########################################################################################################################

# set network in training mode
decoder_eval.eval()

# generate equi-distant samples in the latent space
x_coord = np.arange(0.0, 1.0, 0.001)
y_coord = np.arange(0.0, 1.0, 0.001)

# generate equi-distant mesh grid
x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
m_encodings = np.round(np.array([x_mesh.flatten(), y_mesh.flatten()]).T, 2)

# convert embeddings transactions to torch tensor
m_encodings_torch = torch.FloatTensor(m_encodings)

# determine encoded transactions
m_enc_transactions = decoder_eval(m_encodings_torch)

# convert to pandas dataframe
m_enc_transactions = pd.DataFrame(m_enc_transactions.detach().numpy(), columns=encoded_columns)

# set latent space meshgrid
#x = np.arange(0.0, 1.0, 0.1)
#y = np.arange(0.0, 1.0, 0.1)
#xx, yy = np.meshgrid(x_coordinates, y_coordinates, sparse=True)

# set categorical features
cat_features = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR',	'BSCHL', 'HKONT']

# iterate over all categorical features
for feature in cat_features:

	# determine all columns of actual feature
	feature_columns = [col for col in m_enc_transactions.columns if feature in col]

	# extract corresponding encodings
	feature_encodings = m_enc_transactions[feature_columns]

	# determine max value per row
	feature_encodings['MAX_FEATURE'] = feature_encodings.idxmax(axis=1)

	# concatenate encoding with latent space coordinates
	feature_encodings = pd.concat([feature_encodings, pd.DataFrame(m_encodings, columns=['X', 'Y'])], axis=1)

	# subset max feature value and latent space coordinates
	feature_encodings_final = feature_encodings[['MAX_FEATURE', 'X', 'Y']]

	# left strip feature names of encoded feature
	feature_encodings_final['MAX_FEATURE'] = feature_encodings_final['MAX_FEATURE'].map(lambda x: x.lstrip(str(feature) + '_'))

	# determine determine numerical representation of categorical features
	feature_encodings_final['Z'] = feature_encodings_final['MAX_FEATURE'].astype("category").cat.codes

	# visualize sampled latent space
	title = 'Epoch {} Latent Space Sampling Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
	filename = '02_latent_space_sampling_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
	vha.visualize_z_space_sampling(feature=feature, feature_encodings=feature_encodings_final, x_coord=x_coord, y_coord=y_coord, filename=filename, title=title)

	# visualize sampled latent space
	title = 'Epoch {} Latent Space Sampling Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
	filename = '03_latent_space_sampling_ep_{}_bt_{}_ft_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
	vha.visualize_z_space_sampling_and_transactions(feature=feature, z_representation=embedded_transactions, feature_encodings=feature_encodings_final, x_coord=x_coord,  y_coord=y_coord, filename=filename, title=title)



#print(waers_encodings_final)




#print(list(df.columns))

'''
# case: linear encoder enabled
if experiment_parameter['enc_output'] == 'linear':

    # ixxnit encoder model
    encoder = EncoderLinear.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'sigmoid':

    # init encoder model
    encoder = EncoderSigmoid.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'relu':

    # init encoder model
    encoder = EncoderReLU.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)


# init encoder optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=experiment_parameter['learning_rate_enc'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Encoder architecture established: {}.'.format(now, str(encoder)))
print('[INFO {}] DeepGAN:: No. of Encoder architecture parameters: {}.'.format(now, str(encoder_parameter)))

# init decoder model
decoder = Decoder.Decoder(hidden_size=decoder_hidden, output_size=enc_transactions.shape[1])

# determine decoder number of parameters
decoder_parameter = uha.get_network_parameter(net=decoder)

# init decoder optimizer
decoder_optimizer = optim.Adam(decoder.parameters(), lr=experiment_parameter['learning_rate_dec'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Decoder architecture established: {}.'.format(now, str(decoder)))
print('[INFO {}] DeepGAN:: No. of Decoder architecture parameters: {}.'.format(now, str(decoder_parameter)))

# init discriminator model
discriminator = Discriminator.Discriminator(input_size=d_input, hidden_size=d_hidden, output_size=d_output)

# determine discriminator number of parameters
discriminator_parameter = uha.get_network_parameter(net=discriminator)

# init discriminator optimizer
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=experiment_parameter['learning_rate_dis'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Discriminator architecture established: {}.'.format(now, str(discriminator)))
print('[INFO {}] DeepGAN:: No. of Discriminator architecture parameters: {}.'.format(now, str(discriminator_parameter)))

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push models to cuda
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    discriminator = discriminator.cuda()

# init autoencoder losses
reconstruction_criterion_categorical = nn.BCELoss()
reconstruction_criterion_numeric = nn.MSELoss()

# init the discriminator losses
criterion = nn.BCELoss()

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push losses to cuda
    reconstruction_criterion_categorical.cuda()
    reconstruction_criterion_numeric.cuda()
    criterion.cuda()

# =================== START TRAINING ============================

# init all train results
all_train_results = np.array([])

# iterate over distinct training epochs
for epoch in range(experiment_parameter['no_epochs']):

    # log configuration processing
    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
    print('[INFO {}] DeepGAN:: Start model training of epoch: {}.'.format(now, str(epoch)))

    # case: evaluation epoch
    #if epoch % experiment_parameter['eval_latent_epochs'] == 0:

        # define variables for collecting scores
        #z_space_all_epochs = np.zeros(shape=(experiment_parameter['eval_latent_epochs'], np.shape(trX)[0], 2), dtype='float16')
        #cluster_labels_all_epochs = np.zeros(shape=(experiment_parameter['eval_latent_epochs'], len(trY)), dtype='int16')
        #purity_all_epochs = np.zeros(shape=(experiment_parameter['eval_latent_epochs'], len(z_mean)))
        #cluster_balance_all_epochs = np.zeros(shape=(experiment_parameter['eval_latent_epochs'], len(z_mean)))

    # iterate over distinct minibatches
    for i, batch in enumerate(dataloader):

        # profile batch start time
        batch_start = tm.time()

        # randomly sample minibatch of target distribution
        # todo: maybe include in corresponding data loader
        z_target_batch = z_target[np.random.randint(0, np.shape(z_target)[0]-1, batch.shape[0])]

        # convert to pytorch tensor
        # z_target_batch = torch.FloatTensor(z_target_batch)

        # case: GPU computing enabled
        if experiment_parameter['use_cuda'] is True:

            # push data to cuda
            z_target_batch = z_target_batch.cuda()

        ###### 1. conduct autoencoder training

        # set network in training mode
        encoder.train()
        decoder.train()

        # reset encoder and decoder gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # conduct forward pass
        z = encoder(batch)
        rec_batch = decoder(z)

        # categorical numeric split
        batch_cat, batch_num = split_2_dummy_numeric(batch, encoded_cat_transactions.shape[1])
        rec_batch_cat, rec_batch_num = split_2_dummy_numeric(rec_batch, encoded_cat_transactions.shape[1])

        # backward pass + gradients update
        rec_error_cat = reconstruction_criterion_categorical(input=rec_batch_cat, target=batch_cat)  # one-hot attr error
        rec_error_num = reconstruction_criterion_numeric(input=rec_batch_num, target=batch_num)  # numeric attr error

        # combine both reconstruction errors
        rec_error = rec_error_cat + rec_error_num

        # run error back-propagation
        rec_error.backward()

        # optimize encoder and decoder parameters
        decoder_optimizer.step()
        encoder_optimizer.step()

        ###### 2. conduct discriminator training

        # set network in training mode
        discriminator.train()

        # reset discriminator gradients
        discriminator.zero_grad()

        #  2a. forward pass on fake transactional mb data (sampled)

        # determine discriminator decision
        d_fake_decision = discriminator(z_target_batch)

        # determine discriminator target
        d_fake_target = torch.FloatTensor(torch.ones(d_fake_decision.shape))

        # case: GPU computing enabled
        if experiment_parameter['use_cuda'] is True:

            # push data to cuda
            d_fake_target = d_fake_target.cuda()

        # determine discriminator error
        # the target distribution should look like ones, target = 1
        d_fake_error = criterion(input=d_fake_decision, target=d_fake_target) #.cuda())

        #  2b. forward pass on real transactional mb data (non-sampled)

        # encode real mini-batch data to derive latent representation z
        z_real_batch = encoder(batch)

        # determine discriminator decision
        d_real_decision = discriminator(z_real_batch)

        # determine discriminator target
        d_real_target_zeros = torch.FloatTensor(torch.zeros(d_real_decision.shape))

        # case: GPU computing enabled
        if experiment_parameter['use_cuda'] is True:

            # push data to cuda
            d_real_target_zeros = d_real_target_zeros.cuda()

        # determine discriminator error
        # the real distribution should look like zeros, real = 0
        d_real_error = criterion(input=d_real_decision, target=d_real_target_zeros) #.cuda())  # zeros = fake

        # combine both decision errors
        d_error = d_fake_error + d_real_error

        # run error back-propagation
        d_error.backward()

        # optimize discriminator parameters
        discriminator_optimizer.step()

        ###### 3. conduct encoder-discriminator training

        # 3. Train encoder on D's response (but DO NOT train D on these labels)

        # set network in training mode
        encoder.train()
        discriminator.train()

        # reset encoder gradients
        encoder.zero_grad()

        # reset discriminator gradients
        #discriminator.zero_grad()

        # encode real mini-batch data to derive latent representation z
        z_real_batch = encoder(batch)

        # determine discriminator decision
        d_real_decision = discriminator(z_real_batch)

        # determine discriminator target
        d_real_target_ones = torch.FloatTensor(torch.ones(d_real_decision.shape))

        # case: GPU computing enabled
        if experiment_parameter['use_cuda'] is True:

            # push data to cuda
            d_real_target_ones = d_real_target_ones.cuda()

        # determine discriminator error
        # the real distribution should look like ones, real = 1
        d_real_error_fool = criterion(input=d_real_decision, target=d_real_target_ones) #.cuda())  # ones = true: here we try to fool D

        # run error back-propagation
        d_real_error_fool.backward()

        # optimize the encoder parameters
        encoder_optimizer.step()  # only optimizes encoder's parameters

        # profile batch start time
        batch_end = tm.time()

        # case: evaluation batch
        if i % experiment_parameter['eval_batch'] == 0:

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepGAN:: Completed model training of epoch {}/{} and batch {}/{} in {} sec.'.format(now, str(epoch), str(experiment_parameter['no_epochs']), str(i), str(no_batches), str(np.round(batch_end - batch_start, 4))))

            # log configuration processing
            rec_error_cat = np.round(rec_error_cat.item(), 4)
            rec_error_num = np.round(rec_error_cat.item(), 4)
            rec_error = np.round(rec_error.item(), 4)
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepGAN:: AE reconstruction error, categorical (BCE): {}, numeric (MSE): {}, total: {}.'.format(now, str(rec_error_cat), str(rec_error_num), str(rec_error)))

            # log configuration processing
            d_real_error = np.round(d_real_error.item(), 4)
            d_fake_error = np.round(d_fake_error.item(), 4)
            d_error = np.round(d_error.item(), 4)
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepGAN:: DS classification error, real (BCE): {}, fake (BCE): {}, total: {}.'.format(now, str(d_real_error), str(d_fake_error), str(d_error.item())))

            # log configuration processing
            d_real_error_fool = np.round(d_real_error_fool.item(), 4)
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepGAN:: DS classification error, fool (BCE): {}.'.format(now, str(d_real_error_fool.item())))

            # collect training results
            batch_train_results = np.array([epoch, i, rec_error_cat, rec_error_num, rec_error, d_real_error, d_fake_error, d_error, d_real_error_fool])

            # case: initial batch
            if len(all_train_results) == 0:

                # set all training results to current batch
                all_train_results = batch_train_results

            # case non-initial batch
            else:

                # add training results to all results
                all_train_results = np.vstack([all_train_results, batch_train_results])

    ##### save training_results

    # convert to pandas data frame
    cols = ['epoch', 'mb', 'rec_error_cat', 'rec_error_num', 'rec_error', 'd_real_error', 'd_fake_error', 'd_error', 'd_real_error_fool']
    df_all_train_results = pd.DataFrame(all_train_results, columns=cols)

    # save results data frame to file directory
    filename = '01_training_results.csv'
    df_all_train_results.to_csv(os.path.join(sta_dir, filename), sep=';', index=False, encoding='utf-8')

    ##### analyze latent representation

    if epoch % experiment_parameter['eval_epochs'] == 0:

        # determine latent space representation of all transactions
        z_enc_transactions = encoder(enc_transactions)

        # convert latent space representation to numpy
        z_enc_transactions = z_enc_transactions.cpu().data.numpy()

        # visualize latent space
        title = 'Epoch {} Latent Space Distribtion $Z$'.format(str(epoch))
        vha.visualize_z_space(z_representation=z_enc_transactions, title=title, filename='01_latent_space_ep_{}_bt_{}'.format(str(epoch).zfill(4), str(i).zfill(4)), color='C1')

        # convert encodings to pandas data frame
        cols = ['x', 'y']
        df_z_enc_transactions = pd.DataFrame(z_enc_transactions, columns=cols)

        # save encodings data frame to file directory
        filename = '01_encodings_ep_{}.csv'.format(str(epoch))
        df_z_enc_transactions.to_csv(os.path.join(enc_dir, filename), sep=';', index=False, encoding='utf-8')

    ##### save trained models

    if epoch % experiment_parameter['eval_epochs'] == 0:

        # save trained encoder model file to disk
        encoder_model_name = 'ep_{}_encoder_model.pkl'.format(str(epoch + 1).zfill(4))
        torch.save(encoder.state_dict(), os.path.join(mod_dir, encoder_model_name))

        # save trained decoder model file to disk
        decoder_model_name = 'ep_{}_decoder_model.pkl'.format(str(epoch + 1).zfill(4))
        torch.save(decoder.state_dict(), os.path.join(mod_dir, decoder_model_name))

        # save trained discriminator model file to disk
        discriminator_model_name = 'ep_{}_discriminator_model.pkl'.format(str(epoch + 1).zfill(4))
        torch.save(discriminator.state_dict(), os.path.join(mod_dir, discriminator_model_name))

    #if epoch % plot_interval == 0:
    # =================== INFERENCE and collection of scores ============================

    encoder_cpu = encoder.cpu()
    # get reconstruction of all samples
    x_all = Variable(torch.Tensor(trX))
    z_space = encoder_cpu(x_all).data.numpy()

    # compute kmeans clustering on the latent variables
    kmeans_model = cluster.KMeans(n_clusters=len(z_mean), init=z_mean)
    kmeans_model.fit(z_space)
    kmeans_labels = kmeans_model.labels_

    # compute purity score on the input data based on the labels from kmeans
    scores = get_purity(kmeans_labels, trX)

    # collect intermediate results
    z_space_all_epochs[epoch%experiment_parameter['eval_latent_epochs']] = z_space
    purity_all_epochs[epoch%experiment_parameter['eval_latent_epochs']] = scores[0]
    cluster_balance_all_epochs[epoch%experiment_parameter['eval_latent_epochs']] = scores[1]
    cluster_labels_all_epochs[epoch%experiment_parameter['eval_latent_epochs']] = kmeans_labels

    if epoch % experiment_parameter['eval_epochs'] == 0:
        print("Purity score: %s" % purity_all_epochs[epoch%experiment_parameter['eval_latent_epochs']])
        print("Clusters balance: %s" % cluster_balance_all_epochs[epoch%experiment_parameter['eval_latent_epochs']])
        print("Clusters balance fraction: %s" % np.round((cluster_balance_all_epochs[epoch%experiment_parameter['eval_latent_epochs']] / n_samples), 2))

    if (epoch+1) % experiment_parameter['eval_latent_epochs'] == 0:
        np.save(sta_dir + 'z_space_ep_' + str(epoch+1) + '.npy', z_space_all_epochs)
        np.save(sta_dir + 'purity_ep_' + str(epoch+1) + '.npy', purity_all_epochs)
        np.save(sta_dir + 'cluster_balance_ep_' + str(epoch+1) + '.npy', cluster_balance_all_epochs)
        np.save(sta_dir + 'cluster_labels_ep_' + str(epoch+1) + '.npy', cluster_labels_all_epochs)

        #os.system('tar cvzf 8gauss_2d_scores.tar.gz ' + scores_folder)
        #os.system('rm -R ' + scores_folder)
np.save(sta_dir + 'trY.npy', trY)
'''


