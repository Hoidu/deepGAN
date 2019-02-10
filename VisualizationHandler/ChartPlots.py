# import class libraries
import matplotlib as mlt
mlt.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
import os as os
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

mpl.rcParams['agg.path.chunksize'] = 1000000
plt.rcParams['agg.path.chunksize'] = 1000000

class ChartPlots(object):

	# define plain class constructor
	def __init__(self, plot_dir):

		# set plotting directory
		self.plot_dir = plot_dir

	# set plot dir
	def set_plot_dir(self, plot_dir):

		# set plotting directory
		self.plot_dir = plot_dir

	def visualize_z_space(self, z_representation, title, filename, color='C1'):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# create train scatter plot
		ax.scatter(z_representation.x, z_representation.y, color=color, marker='o', edgecolors='w')

		# set axis limits
		#ax.set_xlim([0.0, 1.0])
		#ax.set_ylim([0.0, 1.0])

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_space_cat_feature(self, z_representation, feature, title, filename, limits=True):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine feature groups
		feature_values = z_representation.groupby(feature)

		# iterate over feature groups
		for feature_name, feature_value in feature_values:

			# create feature_value scatter plot
			ax.scatter(feature_value['x'], feature_value['y'], marker='o', edgecolors='w', label=str(feature_name))

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# case: x and y limits are true
		if limits is True:

			# set axis limits
			ax.set_xlim([0.0, 1.0])
			ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# add legend to plot
		ax.legend(loc='upper left')

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_space_con_feature(self, z_representation, feature, title, filename, limits=True):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# create feature value scatter plot
		scatter = ax.scatter(z_representation['x'], z_representation['y'], c=z_representation[feature], marker='o', edgecolors='w', cmap=plt.cm.coolwarm)

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# case: x and y limits are true
		if limits is True:

			# set axis limits
			ax.set_xlim([0.0, 1.0])
			ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# add colorbar to plot
		fmt = FuncFormatter(lambda x, p: format(int(x), ','))
		cbar = fig.colorbar(scatter, ax=ax, format=fmt)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_categorical_space_sampling(self, decoded_samples_and_embeddings, x_coord, y_coord, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine levels of encodings
		level_min = np.min(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())
		level_max = np.max(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())

		# determine visualization levels
		levels = np.arange(level_min - 0.5, level_max + 1.5, 1.0)

		# determine x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		# create contour plot of categorical attribute
		im = ax.contourf(x_mesh, y_mesh, np.array(decoded_samples_and_embeddings['MAX_FEATURE_CODE']).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=levels)

		# reformat colorbar
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())
		cbar.set_ticklabels(decoded_samples_and_embeddings['MAX_FEATURE'].unique())

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# set axis limits
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_numeric_space_sampling(self, decoded_samples_and_embeddings, feature, x_coord, y_coord, filename, title, levels=10):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine min and max of decoded numerical transaction feature
		level_min = np.min(decoded_samples_and_embeddings[feature])
		level_max = np.max(decoded_samples_and_embeddings[feature])

		# create equi-distant levels
		levels = np.linspace(level_min, level_max, levels)

		# digitize decoded feature values
		decoded_samples_and_embeddings.loc[:, 'level'] = np.digitize(decoded_samples_and_embeddings[feature], levels)

		# determine unique equi-distant levels of original feature value
		unique_levels = decoded_samples_and_embeddings.groupby('level')[feature].max()

		# add additional min level
		unique_levels = pd.Series([0.0]).append(unique_levels)

		# add additional max level
		unique_levels = unique_levels.append(pd.Series([1.0]))

		# determine x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		# create contour plot of numerical attribute
		im = ax.contourf(x_mesh, y_mesh, np.array(decoded_samples_and_embeddings[feature]).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=unique_levels)

		# reformat colorbar
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(unique_levels)

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# set axis limits
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_numeric_space_sampling_and_transactions(self, decoded_samples_and_embeddings, encoded_transactions_and_embeddings, feature, x_coord, y_coord, filename, title, levels=10):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# plot sampled embeddings

		# determine min and max of decoded numerical transaction feature
		level_min = np.min(decoded_samples_and_embeddings[feature])
		level_max = np.max(decoded_samples_and_embeddings[feature])

		# create equi-distant levels
		levels = np.linspace(level_min, level_max, levels)

		# digitize decoded sampled feature values
		decoded_samples_and_embeddings.loc[:, 'level'] = np.digitize(decoded_samples_and_embeddings[feature], levels)

		# determine unique equi-distant levels of original feature value
		unique_levels = decoded_samples_and_embeddings.groupby('level')[feature].max()

		# add additional min level
		unique_levels = pd.Series([0.0]).append(unique_levels)

		# add additional max level
		unique_levels = unique_levels.append(pd.Series([1.0]))

		# determine x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		# create contour plot of numerical attribute
		im = ax.contourf(x_mesh, y_mesh, np.array(decoded_samples_and_embeddings[feature]).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=unique_levels)

		# reformat colorbar
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(unique_levels)

		# plot transactional embeddings

		# digitize encoded transactional feature values
		encoded_transactions_and_embeddings.loc[:, 'level'] = np.digitize(encoded_transactions_and_embeddings.loc[:, feature], levels)

		# determine feature groups
		encoded_transactions_and_embeddings_levels = encoded_transactions_and_embeddings.groupby('level')

		# iterate over numerical feature levels
		for level_name, level in encoded_transactions_and_embeddings_levels:

			# create train scatter plot
			ax.scatter(level['x'], level['y'], marker='o', facecolors='none', edgecolors='w', label=str(level_name))

		# set axis labels
		ax.set_xlabel('$z_1$')
		ax.set_ylabel('$z_2$')

		# set axis limits
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def visualize_z_categorical_space_sampling_and_transactions(self, decoded_samples_and_embeddings, encoded_transactions_and_embeddings, feature, x_coord, y_coord, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine levels of encodings
		level_min = np.min(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())
		level_max = np.max(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())

		# determine equi-distant visualization levels
		levels = np.arange(level_min - 0.5, level_max + 1.5, 1.0)

		# determien x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		# create contour plot of categorical attribute
		im = ax.contourf(x_mesh, y_mesh, np.array(decoded_samples_and_embeddings['MAX_FEATURE_CODE']).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=levels)

		# reformat colorbar
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(decoded_samples_and_embeddings['MAX_FEATURE_CODE'].unique())
		cbar.set_ticklabels(decoded_samples_and_embeddings['MAX_FEATURE'].unique())

		# determine feature groups
		encoded_transactions_and_embeddings_levels = encoded_transactions_and_embeddings.groupby(feature)

		# iterate over feature groups
		for level_name, level in encoded_transactions_and_embeddings_levels:

			# create train scatter plot
			ax.scatter(level['x'], level['y'], marker='o', facecolors='none', edgecolors='w', label=str(level_name))

		# add legend to plot
		#ax.legend(loc='upper left')

		# set axis labels
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')

		# set axis limits
		ax.set_xlim([0.0, 1.0])
		ax.set_ylim([0.0, 1.0])

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def plot_training_losses(self, losses, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# plot the autoencoder reconstruction losses
		ax.plot(losses.index, losses['rec_error_cat'], label='ae_loss_cat ($BCE$)')
		ax.plot(losses.index, losses['rec_error_num'], label='ae_loss_num ($MSE$)')
		ax.plot(losses.index, losses['rec_error'], label='ae_loss')

		# plot the generative adversarial network losses - discriminator
		ax.plot(losses.index, losses['d_real_error'], label='d_loss_real')
		ax.plot(losses.index, losses['d_fake_error'], label='d_loss_fake')
		ax.plot(losses.index, losses['d_error'], label='d_loss')

		# plot the generative adversarial network losses - generator
		ax.plot(losses.index, losses['d_real_error_fool'], label='g_loss')

		# set axis labels
		ax.set_xlabel('$epoch$')
		ax.set_ylabel('$loss$')

		# add legend to plot
		ax.legend(loc='upper left')

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()

	def plot_training_losses_all(self, losses, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [16, 9]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots()

		# init first plot axis
		ax0 = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)

		# plot the autoencoder reconstruction losses
		ax0.plot(losses.index, losses['rec_error_cat'], label='ae_loss_cat ($BCE$)')
		ax0.plot(losses.index, losses['rec_error_num'], label='ae_loss_num ($MSE$)')
		ax0.plot(losses.index, losses['rec_error'], label='ae_loss ($BCE$ + $MSE$)')

		# set axis labels
		ax0.set_xlabel('$epoch$')
		ax0.set_ylabel('$loss$')

		# add legend to plot
		ax0.legend(loc='upper right')

		# set plot header
		ax0.set_title(str('AE - Reconstruction Losses '), fontsize=12)

		# init second plot axis
		ax1 = plt.subplot2grid((4, 1), (1, 0), colspan=1, rowspan=1)

		# plot the generative adversarial network losses - discriminator details
		ax1.plot(losses.index, losses['d_real_error'], label='d_loss_real ($BCE$)')
		ax1.plot(losses.index, losses['d_fake_error'], label='d_loss_fake ($BCE$)')

		# set axis labels
		ax1.set_xlabel('$epoch$')
		ax1.set_ylabel('$loss$')

		# add legend to plot
		ax1.legend(loc='upper right')

		# set plot header
		ax1.set_title(str('GAN - Discriminator Real vs. Discriminator Fake Loss'), fontsize=12)

		# init third plot axis
		ax2 = plt.subplot2grid((4, 1), (2, 0), colspan=1, rowspan=1)

		# plot the generative adversarial network losses - discriminator generator
		ax2.plot(losses.index, losses['d_error'], label='d_loss ($BCE$)')
		ax2.plot(losses.index, losses['d_real_error_fool'], label='g_loss ($BCE$)')

		# set axis labels
		ax2.set_xlabel('$epoch$')
		ax2.set_ylabel('$loss$')

		# add legend to plot
		ax2.legend(loc='upper right')

		# set plot header
		ax2.set_title(str('GAN - Discriminator Loss vs. Generator Loss'), fontsize=12)

		# init third plot axis
		ax3 = plt.subplot2grid((4, 1), (3, 0), colspan=1, rowspan=1)

		# plot the generative adversarial network losses - discriminator generator
		ax3.plot(losses.index, losses['rec_error_latent'], label='rec_loss ($MSE$)')

		# set axis labels
		ax3.set_xlabel('$epoch$')
		ax3.set_ylabel('$loss$')

		# add legend to plot
		ax3.legend(loc='upper right')

		# set plot header
		ax3.set_title(str('EN - Reconstruction Loss Latent Space'), fontsize=12)

		# set grid and tight plotting layout
		plt.grid(True)
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()




