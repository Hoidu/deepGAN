# import class libraries
import matplotlib as mlt
mlt.use('Agg')

import matplotlib as mpl
import matplotlib.pyplot as plt
import os as os


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
		ax.scatter(z_representation[:, 0], z_representation[:, 1], color=color, marker='o')

		# set axis limits
		#ax.set_xlim([0.0, 1.0])
		#ax.set_ylim([0.0, 1.0])

		# set axis labels
		ax.set_xlabel('$x$')
		ax.set_ylabel('$y$')

		# set plot header
		ax.set_title(str(title), fontsize=14)

		# set tight plotting layout
		plt.tight_layout()

		# save plot to plotting directory
		plt.savefig(os.path.join(self.plot_dir, filename), dpi=300)

		# close plot
		plt.close()



