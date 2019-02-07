# import class libraries
import matplotlib as mlt
mlt.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
import os as os
import numpy as np

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

	def visualize_z_space_feature(self, z_representation, feature, title, filename, limits=True):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [10, 10]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine feature groups
		groups = z_representation.groupby(feature)

		# iterate over feature groups
		for name, group in groups:

			# create train scatter plot
			ax.scatter(group.x, group.y, marker='o', edgecolors='w', label=str(name))

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

	def visualize_z_space_sampling(self, feature_encodings, x_coord, y_coord, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [20, 20]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine levels of encodings
		level_min = np.min(feature_encodings['Z'].unique())
		level_max = np.max(feature_encodings['Z'].unique())

		# determine visualization levels
		levels = np.arange(level_min - 0.5, level_max + 1.5, 1.0)

		# determien x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		im = ax.contourf(x_mesh, y_mesh, np.array(feature_encodings['Z']).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=levels)
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(feature_encodings['Z'].unique())
		cbar.set_ticklabels(feature_encodings['MAX_FEATURE'].unique())

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

	def visualize_z_space_sampling_and_transactions(self, feature, z_representation, feature_encodings, x_coord, y_coord, filename, title):

		# set plotting appearance
		plt.style.use('seaborn')
		plt.rcParams['figure.figsize'] = [20, 20]  # width * height
		plt.rcParams['agg.path.chunksize'] = 1000000
		fig, ax = plt.subplots(1, 1)

		# determine levels of encodings
		level_min = np.min(feature_encodings['Z'].unique())
		level_max = np.max(feature_encodings['Z'].unique())

		# determine visualization levels
		levels = np.arange(level_min - 0.5, level_max + 1.5, 1.0)

		# determien x and y meshgrid
		x_mesh, y_mesh = np.meshgrid(x_coord, y_coord)

		im = ax.contourf(x_mesh, y_mesh, np.array(feature_encodings['Z']).reshape(len(x_coord), len(y_coord)), cmap=plt.cm.coolwarm, levels=levels)
		cbar = fig.colorbar(im, ax=ax)
		cbar.set_ticks(feature_encodings['Z'].unique())
		cbar.set_ticklabels(feature_encodings['MAX_FEATURE'].unique())

		# determine feature groups
		groups = z_representation.groupby(feature)

		# iterate over feature groups
		for name, group in groups:

			# create train scatter plot
			ax.scatter(group.x, group.y, marker='o', edgecolors='w', label=str(name))

		# add legend to plot
		ax.legend(loc='upper left')

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




