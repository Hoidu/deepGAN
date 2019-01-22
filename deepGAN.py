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

parser.add_argument('-exp_timestamp', help='', nargs='?', const=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'), default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
parser.add_argument('-seed', help='', nargs='?', const=1234, default=1234)
parser.add_argument('-no_epochs', help='', nargs='?', const=201, default=201)
parser.add_argument('-eval_epochs', help='', nargs='?', const=10, default=10)
parser.add_argument('-enc_output', help='', nargs='?', const='linear', default='linear')
parser.add_argument('-eval_latent_epochs', help='', nargs='?', const=500, default=500)
parser.add_argument('-learning_rate_enc', help='', nargs='?', const=1e-4, default=1e-4)
parser.add_argument('-learning_rate_dec', help='', nargs='?', const=1e-4, default=1e-4)
parser.add_argument('-learning_rate_dis', help='', nargs='?', const=1e-6, default=1e-6)
parser.add_argument('-mini_batch_size', help='', nargs='?', const=128, default=128)
parser.add_argument('-no_gauss', help='', nargs='?', const=2, default=2)
parser.add_argument('-radi_gauss', help='', nargs='?', const=0.8, default=0.8)
parser.add_argument('-stdv_gauss', help='', nargs='?', const=0.015, default=0.015)
parser.add_argument('-eval_batch', help='', nargs='?', const=100, default=100)
parser.add_argument('-use_anomalies', help='', nargs='?', const='False', default='False')
parser.add_argument('-use_cuda', help='', nargs='?', const='False', default='False')
parser.add_argument('-base_dir', help='', nargs='?', const='./01_experiments', default='./01_experiments')

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
experiment_parameter['experiment_dir'], par_dir, sta_dir, enc_dir, plt_dir, mod_dir, prd_dir = uha.create_experiment_directory(param=experiment_parameter, parent_dir=experiment_parameter['base_dir'], architecture='gan')

# save experiment parameters
uha.save_experiment_parameter(param=experiment_parameter, parameter_dir=par_dir)

# set plot directory
vha.set_plot_dir(plt_dir)

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

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN::Data loading, transactional data of shape {} rows and {} columns successfully loaded.'.format(now, str(transactions.shape[0]), str(transactions.shape[1])))

###### pre-process categorical attributes

# determine the categorical attributes
cat_attr = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR',	'BSCHL', 'HKONT']

# extract categorical attributes
cat_transactions = x[cat_attr].copy()

# encode categorical attributs into one-hot
encoded_cat_transactions = pd.get_dummies(cat_transactions)

###### pre-process numerical attributes

# determine the numerical attributes
numeric_attr = ['DMBTR', 'WRBTR']

# extract the numerical attributes
num_transactions = x[numeric_attr].copy()

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

#trY = y.replace({'regular': 1, 'local': 2, 'global': 3}).values

# indices of one-hot encoded attributes
#dummy_mapping = x_categorical.nunique().cumsum()

# max and min of numeric attributes
#numeric_mapping_max = x_numeric.max()
#numeric_mapping_min = x_numeric.min()

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

# training params
#n_samples = int(533000 / experiment_parameter['no_gauss']) * experiment_parameter['no_gauss']

# determine latent space target data
z_target = get_target_distribution(mu=z_mean, sigma=z_stdv, n=transactions.shape[0], dim=2)

# visualize target distribution
title = 'Target Latent Space Distribtion $Z$'
vha.visualize_z_space(z_representation=z_target, title=title, filename='00_latent_space_target.png', color='C0')

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

# init encoder optimizer
encoder_optimizer = optim.Adam(encoder.parameters(), lr=experiment_parameter['learning_rate_enc'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Encoder architecture established: {}.'.format(now, str(encoder)))

# init decoder model
decoder = Decoder.Decoder(hidden_size=decoder_hidden, output_size=enc_transactions.shape[1])

# init decoder optimizer
decoder_optimizer = optim.Adam(decoder.parameters(), lr=experiment_parameter['learning_rate_dec'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Decoder architecture established: {}.'.format(now, str(decoder)))

# init discriminator model
discriminator = Discriminator.Discriminator(input_size=d_input, hidden_size=d_hidden, output_size=d_output)

# init discriminator optimizer
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=experiment_parameter['learning_rate_dis'])

# log configuration processing
now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
print('[INFO {}] DeepGAN:: Discriminator architecture established: {}.'.format(now, str(discriminator)))

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

    # save trained encoder model file to disk
    encoder_model_name = "ep_{}_encoder_model.pth".format(str(epoch + 1).zfill(4))
    torch.save(encoder.state_dict(), os.path.join(mod_dir, encoder_model_name))

    # save trained decoder model file to disk
    decoder_model_name = "ep_{}_decoder_model.pth".format(str(epoch + 1).zfill(4))
    torch.save(decoder.state_dict(), os.path.join(mod_dir, decoder_model_name))

    # save trained discriminator model file to disk
    discriminator_model_name = "ep_{}_discriminator_model.pth".format(str(epoch + 1).zfill(4))
    torch.save(discriminator.state_dict(), os.path.join(mod_dir, discriminator_model_name))

'''
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


