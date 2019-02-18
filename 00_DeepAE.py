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
# from tensorboardX import SummaryWriter

# importing data science libraries
import pandas as pd
import numpy as np
from sklearn import cluster

# import plotting libraries
import matplotlib

matplotlib.use('Agg')

# import project libraries
from UtilsHandler import UtilsHandler
from DataHandler import DataHandler
from NetworksHandler import EncoderLinear
from NetworksHandler import EncoderSigmoid
from NetworksHandler import EncoderReLU
from NetworksHandler import Decoder
from NetworksHandler import Discriminator
from VisualizationHandler import ChartPlots

# init utilities handler
uha = UtilsHandler.UtilsHandler()
dha = DataHandler.DataHandler()
vha = ChartPlots.ChartPlots(plot_dir='./')

########################################################################################################################
# parse and set model parameters
########################################################################################################################

# init and prepare argument parser
parser = ap.ArgumentParser()

parser.add_argument('--exp_timestamp', help='', nargs='?', type=str, default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('--dataset', help='', nargs='?', type=str, default='sap')
parser.add_argument('--seed', help='', nargs='?', type=int, default=1234)
parser.add_argument('--no_epochs', help='', nargs='?', type=int, default=1000)
parser.add_argument('--eval_epochs', help='', nargs='?', type=int, default=1)
parser.add_argument('--enc_output', help='', nargs='?', type=str, default='linear')
parser.add_argument('--eval_latent_epochs', help='', nargs='?', type=int, default=500)
parser.add_argument('--learning_rate_enc', help='', nargs='?', type=float, default=1e-4)
parser.add_argument('--learning_rate_dec', help='', nargs='?', type=float, default=1e-4)
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

# autoencoder architecture
encoder_hidden = [128, 64, 16, 2]
decoder_hidden = [2, 16, 64, 128]

#encoder_hidden = [1024, 512, 256, 2]
#decoder_hidden = [2, 256, 512, 1024]

# create the experiment directory
experiment_parameter['experiment_dir'], par_dir, sta_dir, enc_dir, plt_dir, mod_dir, prd_dir = uha.create_experiment_directory(param=experiment_parameter, parent_dir=experiment_parameter['base_dir'], architecture='aue')

# save experiment parameters
uha.save_experiment_parameter(param=experiment_parameter, parameter_dir=par_dir)

# set plot directory
vha.set_plot_dir(plt_dir)

########################################################################################################################
# load and pre-process the training data
########################################################################################################################

# case: load SAP data
if experiment_parameter['dataset'] == 'sap':

    # set the data dir
    data_dir = './00_datasets/fraud_dataset_v2.csv'

    enc_transactions, enc_transactions_labels, enc_cat_transactions, enc_con_transactions = dha.get_SAP_data(experiment_parameter=experiment_parameter, data_dir=data_dir)

# case: load mnist data
elif experiment_parameter['dataset'] == 'mnist':

    enc_transactions, enc_transactions_labels, _, _ = dha.get_MNIST_data()

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

########################################################################################################################
# create the latent space target data
########################################################################################################################

# determine the radius and theta of the target mixture of gaussians
radius = experiment_parameter['radi_gauss']
theta = np.linspace(0, 2 * np.pi, experiment_parameter['no_gauss'], endpoint=False)

# determine x and y coordinates of the target mixture of gaussians
x_centroid = (radius * np.sin(theta) + 1) / 2
y_centroid = (radius * np.cos(theta) + 1) / 2

# determine gaussians mean (centroids) and standard deviation
z_mean = np.vstack([x_centroid, y_centroid]).T
z_stdv = experiment_parameter['stdv_gauss']

# determine latent space target data
z_target = uha.get_target_distribution(mu=z_mean, sigma=z_stdv, n=enc_transactions.shape[0], dim=2)

# convert to pandas data frame
cols = ['x', 'y']
df_z_target = pd.DataFrame(z_target, columns=cols)

# visualize target distribution
title = 'Gaussian All Target Latent Space Distribution $Z$'
file_name = '00_latent_space_target_gauss_all.png'
vha.visualize_z_space(z_representation=df_z_target, title=title, filename=file_name, color='C0')

# convert target distribution to torch tensor
z_target = torch.FloatTensor(z_target)

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push data to cuda
    z_target.cuda()

########################################################################################################################
# init adversarial autoencoder architecture and loss functions
########################################################################################################################

# case: linear encoder enabled
if experiment_parameter['enc_output'] == 'linear':

    # init encoder model
    encoder = EncoderLinear.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'sigmoid':

    # init encoder model
    encoder = EncoderSigmoid.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

elif experiment_parameter['enc_output'] == 'relu':

    # init encoder model
    encoder = EncoderReLU.Encoder(input_size=enc_transactions.shape[1], hidden_size=encoder_hidden)

# determine encoder number of parameters
encoder_parameter = uha.get_network_parameter(net=encoder)

# init encoder optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=experiment_parameter['learning_rate_enc'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepAE:: Encoder architecture established: {}.'.format(now, str(encoder)))
print('[INFO {}] DeepAE:: No. of Encoder architecture parameters: {}.'.format(now, str(encoder_parameter)))

# init decoder model
decoder = Decoder.Decoder(hidden_size=decoder_hidden, output_size=enc_transactions.shape[1])

# determine decoder number of parameters
decoder_parameter = uha.get_network_parameter(net=decoder)

# init decoder optimizer
decoder_optimizer = optim.Adam(decoder.parameters(), lr=experiment_parameter['learning_rate_dec'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepAE:: Decoder architecture established: {}.'.format(now, str(decoder)))
print('[INFO {}] DeepAE:: No. of Decoder architecture parameters: {}.'.format(now, str(decoder_parameter)))

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push models to cuda
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# init autoencoder losses
reconstruction_criterion_categorical = nn.BCELoss()
reconstruction_criterion_numeric = nn.MSELoss()

# case: GPU computing enabled
if experiment_parameter['use_cuda'] is True:

    # push losses to cuda
    reconstruction_criterion_categorical.cuda()
    reconstruction_criterion_numeric.cuda()

# =================== START TRAINING ============================

# init all train results
all_train_results = np.array([])

# iterate over distinct training epochs
for epoch in range(experiment_parameter['no_epochs']):

    # log configuration processing
    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
    print('[INFO {}] DeepAE:: Start model training of epoch: {}.'.format(now, str(epoch)))

    # iterate over distinct minibatches
    for i, batch in enumerate(dataloader):

        # profile batch start time
        batch_start = tm.time()

        # randomly sample minibatch of target distribution
        # todo: maybe include in corresponding data loader
        z_target_batch = z_target[np.random.randint(0, np.shape(z_target)[0] - 1, batch.shape[0])]

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

        if experiment_parameter['dataset'] == 'sap':

            # categorical numeric split
            batch_cat, batch_num = uha.split_2_dummy_numeric(batch, enc_cat_transactions.shape[1])
            rec_batch_cat, rec_batch_num = uha.split_2_dummy_numeric(rec_batch, enc_cat_transactions.shape[1])

            # determine reconstruction errors
            rec_error_cat = reconstruction_criterion_categorical(input=rec_batch_cat, target=batch_cat)  # one-hot attr error
            rec_error_num = reconstruction_criterion_numeric(input=rec_batch_num, target=batch_num)  # numeric attr error

            # combine both reconstruction errors
            rec_error = rec_error_cat + rec_error_num

        elif experiment_parameter['dataset'] == 'mnist':

            # determine reconstruction error
            rec_error = reconstruction_criterion_numeric(input=rec_batch, target=batch)  # numeric attr error

        # run error back-propagation
        rec_error.backward()

        # optimize encoder and decoder parameters
        decoder_optimizer.step()
        encoder_optimizer.step()

        # profile batch start time
        batch_end = tm.time()

        # case: evaluation batch
        if i % experiment_parameter['eval_batch'] == 0:

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepAE:: Completed model training of epoch {}/{} and batch {}/{} in {} sec.'.format(now, str(epoch), str(experiment_parameter['no_epochs']), str(i), str(no_batches), str(np.round(batch_end - batch_start, 4))))

            if experiment_parameter['dataset'] == 'sap':

                # log configuration processing
                rec_error_cat = np.round(rec_error_cat.item(), 4)
                rec_error_num = np.round(rec_error_cat.item(), 4)
                rec_error = np.round(rec_error.item(), 4)

            elif experiment_parameter['dataset'] == 'mnist':

                # log configuration processing
                rec_error_cat = 0.0
                rec_error_num = np.round(rec_error.item(), 4)
                rec_error = np.round(rec_error.item(), 4)

            rec_error = np.round(rec_error.item(), 4)
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DeepAE:: AE reconstruction error, categorical (BCE): {}, numeric (MSE): {}, total: {}.'.format(now, str(rec_error_cat), str(rec_error_num), str(rec_error)))

            # collect training results
            batch_train_results = np.array([epoch, i, rec_error_cat, rec_error_num, rec_error])

            # case: initial batch
            if len(all_train_results) == 0:

                # set all training results to current batch
                all_train_results = batch_train_results

            # case non-initial batch
            else:

                # add training results to all results
                all_train_results = np.vstack([all_train_results, batch_train_results])

    ##### save training results

    if epoch % experiment_parameter['eval_epochs'] == 0 and epoch != 0:

        # convert to pandas data frame
        cols = ['epoch', 'mb', 'rec_error_cat', 'rec_error_num', 'rec_error']
        df_all_train_results = pd.DataFrame(all_train_results, columns=cols)

        # aggregate distinct losses per training epoch
        df_all_train_results_agg = df_all_train_results.groupby(['epoch']).sum()

        # save results data frame to file directory
        filename = '01_training_results.csv'
        df_all_train_results_agg.to_csv(os.path.join(sta_dir, filename), sep=';', index=True, encoding='utf-8')

        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DeepAE:: ###########')
        print('[INFO {}] DeepAE:: AE epoch reconstruction error, categorical (BCE): {}, numeric (MSE): {}, total: {}.'.format(now, str(df_all_train_results_agg.iloc[epoch]['rec_error_cat']), str(df_all_train_results_agg.iloc[epoch]['rec_error_num']), str(df_all_train_results_agg.iloc[epoch]['rec_error'])))
        print('[INFO {}] DeepAE:: ###########')

        # plot training losses
        title = 'AE Training losses'.format(str(epoch))
        file_name = '01_training_results.png'
        vha.plot_AE_training_losses_all(losses=df_all_train_results_agg, filename=file_name, title=title)

    ##### analyze latent representation

    if epoch % experiment_parameter['eval_epochs'] == 0 and epoch != 0:

        # determine latent space representation of all transactions
        z_enc_transactions = encoder(enc_transactions)

        # convert latent space representation to numpy
        z_enc_transactions = z_enc_transactions.cpu().data.numpy()

        # convert encodings to pandas data frame
        cols = ['x', 'y']
        df_z_enc_transactions = pd.DataFrame(z_enc_transactions, columns=cols)

        # visualize latent space
        title = 'Epoch {} Latent Space Distribtion $Z$'.format(str(epoch))
        file_name = '01_latent_space_ep_{}_bt_{}'.format(str(epoch).zfill(4), str(i).zfill(4))
        vha.visualize_z_space(z_representation=df_z_enc_transactions, title=title, filename=file_name, color='C1')

        # convert labels to data frame
        df_enc_transactions_labels = pd.DataFrame(enc_transactions_labels, columns=['label'])

        # merge latent space and
        df_z_enc_transactions_labels = pd.concat([df_z_enc_transactions, df_enc_transactions_labels], axis=1)

        # visualize latent space
        title = 'Epoch {} Latent Space Distribtion $Z$'.format(str(epoch))
        file_name = '01_latent_space_ep_{}_bt_{}_label'.format(str(epoch).zfill(4), str(i).zfill(4))
        vha.visualize_z_space_label(z_representation=df_z_enc_transactions_labels, title=title, filename=file_name)

        # save encodings data frame to file directory
        filename = '01_encodings_ep_{}.csv'.format(str(epoch))
        #df_z_enc_transactions.to_csv(os.path.join(enc_dir, filename), sep=';', index=False, encoding='utf-8')

    ##### save trained models

    if epoch % experiment_parameter['eval_epochs'] == 0 and epoch != 0:

        # save trained encoder model file to disk
        encoder_model_name = 'ep_{}_encoder_model.pkl'.format(str(epoch + 1).zfill(4))
        torch.save(encoder.state_dict(), os.path.join(mod_dir, encoder_model_name))

        # save trained decoder model file to disk
        decoder_model_name = 'ep_{}_decoder_model.pkl'.format(str(epoch + 1).zfill(4))
        torch.save(decoder.state_dict(), os.path.join(mod_dir, decoder_model_name))

