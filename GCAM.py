import random
import torch
import torch.nn as nn
import numpy as np
import time
import seaborn as sns
from scipy.stats import mannwhitneyu
from utils import batchify, get_batch, repackage_hidden # self-designed modules; specific to project

class GCAM:
	def __init__(self, model_file, cuda, args):
		self.load_model(model_file)
		self.args = args

	# specific to project
	def load_model(self, model_file):
		model_structure = torch.load(open(model, 'rb'))
		self.model = model_structure[0]
		self.criterion = model_structure[1]
		self.optimizer = model_structure[2]
		self.model_classifier = model_structure[3]

	def hook_gradient(self, module, input, output):
		self.gradients.append(output[0].detach().data.cpu().numpy())

	def hook_activation(self, module, input, output):
		self.activations.append(output.detach().data.cpu().numpy())

	# specific to project	
	def add_to_samples(self, sample_gcam, mapping_data, chrs):
		for read_idx in range(len(sample_gcam)):
			read_score = sample_gcam[read_idx][0]
			chr_no, flag, read_pos = mapping_data[read_idx]
			if break_flag(flag)[2] == 0:
				if chr_no < 23:
					chr_no = 'chr'+str(chr_no)
				else:
					chr_no = chr_dic[chr_no]
				if chr_no not in chrs:
					chrs[chr_no] = [[],[]]
				chrs[chr_no][0].append(read_score)
				chrs[chr_no][1].append(read_pos)

	def train(self, data, progress_check_size = 0):
		print('Loading all samples')
		# Specific to project
		if self.args.model == 'QRNN': model.reset()
		lr, batch_size, batch_size_of_sample = self.args.lr, self.args.batch_size, self.args.batch_size_of_sample
		# Specific to project; hook to all convolutional layers
		self.model_classifier.conv1_r.register_forward_hook(hook_activation)
		self.model_classifier.conv2_r.register_forward_hook(hook_activation)
		self.model_classifier.conv3_r.register_forward_hook(hook_activation)
		self.model_classifier.conv4_r.register_forward_hook(hook_activation)
		self.model_classifier.conv5_r.register_forward_hook(hook_activation)

		self.model_classifier.conv1_r.register_backward_hook(hook_gradient)
		self.model_classifier.conv2_r.register_backward_hook(hook_gradient)
		self.model_classifier.conv3_r.register_backward_hook(hook_gradient)
		self.model_classifier.conv4_r.register_backward_hook(hook_gradient)
		self.model_classifier.conv5_r.register_backward_hook(hook_gradient)

		# Specific to project; find accuracy and seperate cancer and healthy samples
		total_accuracy = []
		no_of_cancer_samples, no_of_healthy_samples = 0, 0
		self.chrs_in_cancer_samples = {}
		self.chrs_in_healthy_samples = {}

		print('Calculating GCAM scores')
		no_of_batches = len(data)
		progress_check = random.sample(range(no_of_batches), min(progress_check_size, len(no_of_batches)))
		curr_time = time.time()
		for batch_no, batch_data in enumerate(data):
			if batch_no in progress_check:
				print('\n {}%: {} seconds'.format(str(progress_check.index(batch_no)*10), time.time() - curr_time))
				curr_time = time.time()
			self.optimizer.param_groups[0]['lr'] = lr2
			self.model.train()
			self.model_classifier.train()
			training_data = np.transpose(batch_data[0].view(-1,batch_data[0].size(2)))
			targets = batch_data[1].view(-1)
			if self.args.cuda:
				training_data, targets = training_data.cuda(), targets.cuda()

			hidden = self.model.init_hidden(batch_size, batch_size_of_sample)
			self.optimizer.zero_grad()

			# specific to QRNN structure designed in project
			output, hidden, rnn_hs, dropped_rnn_hs, class_output  = self.model(training_data, hidden, return_h=True, batch_size_of_sample, classification = True)
			class_output = self.repackage_hidden(class_output)
			output  = self.model_classifier(class_output, hidden, return_h=True, batch_size_of_sample = batch_size_of_sample, classification = True)
			predictions = output.max(1)[1]
			accuracy = (predictions == targets).float().mean().item()
			total_accuracy.append(accuracy)
			loss = self.criterion(output, targets)
			if alpha: loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
			if beta: loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])

			# Backpropagation
			loss.backward()
			self.optimizer.step()
			self.optimizer.param_groups[0]['lr'] = lr2

			# store metadata required for project
			self.metadata = ''

			gradients = []
			activations = []
			# Specific to project, data annotation
			batch_mapping_data = batch_data[3].data.cpu().numpy()

			temp_container = []
			for i in range(len(predictions)):
				if predictions[i] == targets[i]: # only record correctly classified samples
					sample_gcam = []
					# GCAM calculations
					for j in range(len(gradients)):
						layer_gcam = gradients[j][i]*activations[len(activations)-j-1][i]  
						layer_gcam = cv2.resize(layer_gcam, (1,4096))
						sample_gcam.append(layer_gcam)

					sample_gcam = np.array(sample_gcam)
					sample_gcam = np.sum(sample_gcam, axis=0)
					sample_gcam = np.maximum(sample_gcam, 0)
					sample_gcam -= np.min(sample_gcam)
					sample_gcam /= np.max(sample_gcam)
					# normalize to -1 and 1
					sample_gcam *= 2
					sample_gcam -= 1

					if predictions[i] == 1:
						self.add_to_samples(sample_gcam, batch_mapping_data[i], self.chrs_in_cancer_samples)
						no_of_cancer_samples += 1
					else:
						self.add_to_samples(sample_gcam, batch_mapping_data[i], self.chrs_in_healthy_samples)
						no_of_healthy_samples += 1

					if batch_no in progress_check:
						temp_container += sample_gcam

			# Plot distribution of scores, ensure scores are evenly distributed
			sns.distplot(temp_container, hist=False, rug=False)
			plt.xlabel('GCAM Score')
			plt.savefig('Distribution_of_GCAM_scores_in_' + class_interest+'class_batch_no_'+str(batch_no)+'.png')
			plt.clf()
			temp_container = []

			gradients = []
			activations = []

		self.metadata += "Accuracy = " + str(sum(total_accuracy)/len(total_accuracy))+ '\n'
		self.metadata += 'Number of correctly classified cancer samples: '+ str(no_of_cancer_samples) + '\n'
		self.metadata += 'Number of correctly classified healthy samples: '+ str(no_of_healthy_samples) + '\n'

	def helper(self, chrs, class_interest, random_x):
		print('Visualization of '+class_interest+' class results')

		wig_file = open(class_interest+'_class_GCAM_scores.wig', 'w+')
		norm_wig_file = open(class_interest+'_class_normalized_GCAM_scores.wig', 'w+')
		
		chrs_keys = chrs.keys()

		if test == True: chrs_keys = {'chr1'}
		percent_coverage = [0, 0]

		mapped_scores, unmapped_scores = []

		prev_time = time.time()

		for chr_no in chrs_keys:
			print('Saving '+chr_no+' GCAM scores in wig file')
			
			wig_file.write('variableStep chrom='+chr_no+'\n')
			norm_wig_file.write('variableStep chrom='+chr_no+'\n')

			read_scores = chrs[chr_no][0]
			read_pos = chrs[chr_no][1]

			scores, hits = [], []

			for i in range(len(read_scores)):
				pos_idx = read_pos[i]
				score = read_scores[i]
				if pos_idx+101 >= len(scores):
					scores += [0]*(pos_idx-len(scores)+102)
					hits += [0]*(pos_idx-len(hits)+102)
				for j in range(pos_idx, pos_idx+101):
					scores[j] += score
					if scores[j] < 0: scores[j] = 0
					hits[j] += 1

			for pos_idx in range(len(scores)):
				score = scores[pos_idx]
				hit = hits[pos_idx]
				if score != 0 and hit != 0:
					norm_const =  1-math.exp(-0.2*hit)
					norm_score = (norm_const*score)/hit
					wig_file.write(str(pos_idx)+' '+str(score)+'\n')
					norm_wig_file.write(str(pos_idx)+' '+str(norm_score)+'\n')
					percent_coverage[0] += 1
					mapped = False
					if chr_no in genes_loc:
						for gene in genes_loc[chr_no]:
							if gene_check(pos, gene[1:3]):
								mapped = True
					if mapped:
						mapped_scores.append(score)
					else:
						unmapped_scores.append(score)
				percent_coverage[1] += 1

			curr_time = time.time()
			print('Completed in {}s'.format(curr_time-prev_time))
			prev_time = curr_time

		self.metadata += 'Percentage coverage for {} class: {}%'.format(class_interest, 100*percent_coverage[0]/percent_coverage[1])+'\n'

		print('Performing Mann Whitney U Test')
		stat, p = mannwhitneyu(mapped_scores, unmapped_scores, alternative='greater')
		self.metadata += 'Statistics={}, p= '.format(stat)+str(p)+'\n'

		if p > alpha:
			self.metadata += 'Same distribution (fail to reject H0); Mapped scores same as unmapped scores\n'
		else:
			self.metadata += 'Different distribution (reject H0); Mapped scores greater than unmapped scores\n'

		print('Plotting distributions of scores')
		sns.distplot(mapped_scores, hist=False, rug=False, label='mapped')
		sns.distplot(unmapped_scores, hist=False, rug=False, label='unmapped')
		plt.xlabel('GCAM Score')
		plt.legend()
		plt.savefig('Distribution_of_GCAM_scores_in_' + class_interest+'_class.png')

		unmapped_scores += mapped_scores
		print('Performing random sampling for {} times'.format(random_x))
		sample_size = len(mapped_scores)
		diff_dist = 0
		for m in range(random_x):
			random.shuffle(unmapped_scores)
			sample_mapped_scores = unmapped_scores[:sample_size]
			sample_unmapped_scores = unmapped_scores[sample_size:]
			stat, p = mannwhitneyu(sample_mapped_scores, sample_unmapped_scores, alternative='greater')
			if p <= alpha:
				diff_dist += 1
			self.metadata += 'Statistics={}, p= '.format(stat)+str(p)+'\n'

		rate = 100*diff_dist/random_x
		self.metadata += 'Out of {} random sampling rounds, {} is/are different distributions\n'

		wig_file.close()
		norm_wig_file.close()

	def results_to_wig(self, random_x=10):
		self.helper(self.chrs_in_cancer_samples, 'Cancer', random_x)
		self.helper(self.chrs_in_healthy_samples, 'Healthy', random_x)

	def write_metadata(self):
		metadata_file = open('metadata.txt', 'w+')
		metadata_file.write(self.metadata)
		metadata_file.close()