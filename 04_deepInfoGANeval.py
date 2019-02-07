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
parser.add_argument('--experiment_dir', help='', nargs='?', type=str, default='./01_experiments/2019-02-07_16-19-30_deepGAN_exp_sd_1234_ep_400_gs_8_rd_0.8_sd_0.03_mb_128_eco_linear_gpu_False')
parser.add_argument('--target_feature', help='', nargs='?', type=str, default='BUKRS')
parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)
parser.add_argument('--no_epochs', help='', nargs='?', type=int, default=100)
parser.add_argument('--eval_epochs', help='', nargs='?', type=int, default=50)
parser.add_argument('--enc_output', help='', nargs='?', type=str, default='relu')
parser.add_argument('--eval_latent_epochs', help='', nargs='?', type=int, default=500)
parser.add_argument('--learning_rate_enc', help='', nargs='?', type=float, default=1e-4)
parser.add_argument('--learning_rate_dec', help='', nargs='?', type=float, default=1e-4)
parser.add_argument('--learning_rate_dis', help='', nargs='?', type=float, default=1e-6)
parser.add_argument('--mini_batch_size', help='', nargs='?', type=int, default=128)
parser.add_argument('--no_gauss', help='', nargs='?', type=int, default=8)
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
# experiment_parameter['enc_linear'] = uha.str2bool(experiment_parameter['enc_linear'])
experiment_parameter['use_cuda'] = uha.str2bool(experiment_parameter['use_cuda'])
experiment_parameter['use_anomalies'] = uha.str2bool(experiment_parameter['use_anomalies'])

# init deterministic seeds
seed_value = 1234
np.random.seed(seed_value)  # set numpy seed
torch.manual_seed(seed_value)  # set pytorch seed CPU
torch.cuda.manual_seed(seed_value)  # set pytorch seed GPU

# init latent space dimensionality
latent_space_dim = experiment_parameter['no_gauss'] + 2

# autoencoder architecture
encoder_hidden = [256, 64, 16, latent_space_dim]
decoder_hidden = [latent_space_dim, 16, 64, 256]

# discriminator architecture
d_input = latent_space_dim
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

#transactions_filtered = x

# reset index of filtered transactions
transactions_filtered.index = range(0, transactions_filtered.shape[0])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN::Data loading, transactional data of shape {} rows and {} columns successfully loaded.'.format(
    now, str(transactions.shape[0]), str(transactions.shape[1])))

###### pre-process categorical attributes

# determine the categorical attributes
cat_attr = ['BUKRS', 'WAERS', 'KTOSL', 'PRCTR',	'BSCHL', 'HKONT']

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

# convert encodings to pandas data frame
cols = [str(x) for x in range(experiment_parameter['no_gauss'])]
cols.extend(['x', 'y'])
df_z_enc_transactions = pd.DataFrame(z_enc_transactions, columns=cols)

# visualize latent space
title = 'Epoch {} Latent Space Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
vha.visualize_z_space(z_representation=df_z_enc_transactions, title=title, filename='01_latent_space_ep_{}_bt_{}'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), color='C1'))

# merge embeddings with actual transactions
embedded_transactions = pd.concat([transactions_filtered, df_z_enc_transactions], axis=1, sort=False)

# save encodings data frame to file directory
filename = '01_embedded_eval_ep_{}.csv'.format(str(experiment_parameter['no_epochs']).zfill(4))
embedded_transactions.to_csv(os.path.join(evl_dir, filename), sep=';', index=False, encoding='utf-8')

# visualize latent space
feature = 'KTOSL'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name, limits=False)

# visualize latent space
feature = 'WAERS'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name, limits=False)

# visualize latent space
feature = 'BUKRS'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name, limits=False)

# visualize latent space
feature = 'PRCTR'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name, limits=False)

# visualize latent space
feature = 'BSCHL'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name, limits=False)

# visualize latent space
feature = 'HKONT'
title = 'Training Epoch {} Latent Space {} Feature Distribution $Z$'.format(str(experiment_parameter['no_epochs']), str(feature))
file_name = '01_latent_space_ep_{}_bt_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(feature))
vha.visualize_z_space_feature(z_representation=embedded_transactions, feature=feature, title=title, filename=file_name)

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Completed latent space feature distribution plots of data samples.'.format(now))

########################################################################################################################
# evaluate latent space feature distribution (entire space)
########################################################################################################################

# set network in training mode
decoder_eval.eval()

# generate equi-distant samples in the latent space
x_coord = np.arange(0.0, 1.0, 0.01)
y_coord = np.arange(0.0, 1.0, 0.01)

# generate equi-distant mesh grid
x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)
m_encodings_numeric = np.round(np.array([x_mesh.flatten(), y_mesh.flatten()]).T, 2)

# determine all encoded columns of target feature
target_feature_columns = [col for col in encoded_cat_transactions.columns if experiment_parameter['target_feature'] in col]

# determine all encoded values of target feature
encoded_cat_transactions_target = encoded_cat_transactions[target_feature_columns]

# iterate number of gaussians
for i in range(0, experiment_parameter['no_gauss']):

    # create artificial categorical encodings
    me_encodings_catgeorical = np.zeros((m_encodings_numeric.shape[0], experiment_parameter['no_gauss']))
    me_encodings_catgeorical[:, i] = 1.0

    # merge categorical and numerical encodings
    m_encodings = np.column_stack((me_encodings_catgeorical, m_encodings_numeric))

    # log configuration processing
    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
    print('[INFO {}] DeepGAN:: Successfully sampled {} equi-distant samples in the latent space.'.format(now, str(m_encodings.size)))

    # convert embeddings transactions to torch tensor
    m_encodings_torch = torch.FloatTensor(m_encodings)

    # determine encoded transactions
    m_enc_transactions = decoder_eval(m_encodings_torch)

    # convert to pandas dataframe
    m_enc_transactions = pd.DataFrame(m_enc_transactions.detach().numpy(), columns=encoded_columns)

    # convert mesh encodings to pandas
    cols = [str(x) for x in range(experiment_parameter['no_gauss'])]
    cols.extend(['X', 'Y'])
    df_m_encodings = pd.DataFrame(m_encodings, columns=cols)

    # determine embedded transactions of actual target value
    embedded_transactions_actual = embedded_transactions[encoded_cat_transactions_target.iloc[:, i] == 1.0]

    # set categorical features
    cat_features = ['KTOSL', 'WAERS', 'BUKRS', 'PRCTR', 'BSCHL', 'HKONT']

    # iterate over all categorical features
    for feature in cat_features:

        # determine all columns of actual feature
        feature_columns = [col for col in m_enc_transactions.columns if feature in col]

        # extract corresponding encodings
        feature_encodings = m_enc_transactions[feature_columns]

        # determine max value per row
        feature_encodings['MAX_FEATURE'] = feature_encodings.idxmax(axis=1)

        # concatenate encoding with latent space coordinates
        feature_encodings = pd.concat([feature_encodings, df_m_encodings], axis=1)

        # subset max feature value and latent space coordinates
        feature_encodings_final = feature_encodings[['MAX_FEATURE', 'X', 'Y']]

        # left strip feature names of encoded feature
        feature_encodings_final['MAX_FEATURE'] = feature_encodings_final['MAX_FEATURE'].map(lambda x: x.lstrip(str(feature) + '_'))

        # determine determine numerical representation of categorical features
        feature_encodings_final['Z'] = feature_encodings_final['MAX_FEATURE'].astype("category").cat.codes

        # visualize sampled latent space
        title = 'Training Epoch {} Latent Space Sampling Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
        file_name = '02a_latent_space_sampling_ep_{}_bt_{}_gs_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(i), str(feature))
        vha.visualize_z_space_sampling(feature_encodings=feature_encodings_final, x_coord=x_coord, y_coord=y_coord, filename=file_name, title=title)

        # visualize sampled latent space
        title = 'Epoch {} Latent Space Sampling Distribution $Z$'.format(str(experiment_parameter['no_epochs']))
        file_name = '02b_latent_space_sampling_ep_{}_bt_{}_gs_{}_ft_{}.png'.format(str(experiment_parameter['no_epochs']).zfill(4), str('eval'), str(i), str(feature))
        vha.visualize_z_space_sampling_and_transactions(feature=feature, z_representation=embedded_transactions_actual, feature_encodings=feature_encodings_final, x_coord=x_coord, y_coord=y_coord, filename=file_name, title=title)

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DeepGAN:: Completed latent space feature area plots of feature {}.'.format(now, str(feature)))
