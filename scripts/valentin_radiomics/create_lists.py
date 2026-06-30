#System modules
import os, sys, glob
import numpy as np
import pandas as pd
import pdb
from sklearn.utils import shuffle

sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio

#Constants
RND_SEED = 5

def main():

	out = []
	out_tr = []
	out_tst = []
	errors = []

	#Train and test percentage. The sum of both must be equal to 1. If both are 0 it creates a list of all the data
	tr_split = 0.7
	tst_split = 0.3

	if not (tr_split == 0 and tst_split == 0):
		if (tr_split+tst_split) != 1:
			print("**> ERROR: the training and test split must be equal to 1")
			exit()

	data_dct = pd.read_excel('/home/fauvev01/Bachir/SBRT/data/labels/SBRT Amine.xlsx')
	# labels = data_dct['Response']
	# labels = labels.replace(['Complete','Partial','Stable','Progression'],['Gp1','Gp1','Gp2','Gp2'])

	data_dir = '/home/fauvev01/Bachir/SBRT/data/Amine SBRT/'

	data = glob.glob(data_dir+'*')

	data = shuffle(data, random_state=RND_SEED)

	# Find matching name in Patient column of data_dct
	subjs_labs = find_labels(data_dct, data)

	tr_data = data[:int(len(data)*tr_split)]
	tst_data = data[int(len(data)*tr_split):]


	if tr_split != 0 or tst_split != 0:
		for group in [tr_data, tst_data]:

			if group == tr_data:
				group_name = 'train'
			else:
				group_name = 'test'

			#Loops over all the subjects and checks if they have the radiomics table 
			for i in group:

				if os.path.isfile(i):
					continue

				subject = os.path.basename(i)

				# replace , and . with _
				# subject = subject.replace(',', '')
				# subject = subject.replace('.', '')
				# subject = subject.replace(' ', '_')

				# subject_nb = subject.split('.')[1]

				#Gets the subject label
				sub_lab = subjs_labs[subject]

				if sub_lab == 'Complete' or sub_lab == 'Partial':
					label = 1
				elif sub_lab == 'Stable' or sub_lab == 'Progression':
					label = 0

				# pdb.set_trace()

				#Loops over the pre-treatment protocols and checks if they have the csv file
				for j in glob.glob(i+'/Pre-Tx/**/*.csv', recursive=True):
					
					# Checks if it is a file
					if os.path.isfile(j):

						# Gets the sequence name
						seq = j.split('/')[-2]
						# pdb.set_trace()
						out.append([subject, seq, j, label])

						
				#Loops over the post-treatment protocols
				for k in glob.glob(i+'/Post Tx/**/*.csv', recursive=True):
					
					# Checks if it is a file
					if os.path.isfile(k):

						# Gets the sequence name
						seq = k.split('/')[-2]
						# pdb.set_trace()
						out.append([subject, seq, k, label])

			# stores the output in train or test list
			if group_name == 'train':
				out_tr = out
			elif group_name == 'test':
				out_tst = out

			out = []

	else:
		pass

	# pdb.set_trace()
	#Patients 45, 50, 58 and 59 were excluded from the study

	if tr_split != 0 or tst_split != 0:
		#Saves the output
		out_labels = ['Subject', 'Sequence', 'Path', 'Label']
		bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/pre&post_tr_lst.csv', out_tr, out_labels)
		bio.write_csv('/home/fauvev01/Bachir/SBRT/data/lists/pre&post_tst_lst.csv', out_tst, out_labels)
	else:
		pass

	print('>>> DONE <<<')


def find_labels(data_dct, data):
	# pdb.set_trace()
	subjs_dct = {}
	subjs_labs = {}
	for i in data:
		subject = os.path.basename(i)

		# print(subject)

		# Create empty key in dct for subject
		subjs_dct[subject] = []
		
		# Checks if there is a similar name in the Patient column
		for j in range(len(data_dct['Patient.1'])):

			# Get index
			idx = data_dct['Patient.1'].index[j]

			# Get the name
			name = data_dct['Patient.1'][idx]

			# Gets the subject label
			label = data_dct['Response'][idx]

			if name in subject:

				# Checks if subject already in dict
				if subject in subjs_dct:
					# pdb.set_trace()
					# Add the name to the list
					subjs_dct[subject].append(name)
					subjs_labs[subject] = label
					break
				else:
					# Add the name to the dict
					subjs_dct[subject] = [name]
					subjs_labs[subject] = label
					break

	# # Get len of dct
	# len_dct = len(subjs_dct)

	# # Count how many keys are empty and how many have double names
	# empty_keys = 0
	# double_names = 0
	# for i in subjs_dct:
	# 	if subjs_dct[i] == []:
	# 		empty_keys += 1
	# 	elif len(subjs_dct[i]) > 1:
	# 		double_names += 1

	# # Get the values with double names
	# double_names_lst = []
	# for i in subjs_dct:
	# 	if len(subjs_dct[i]) > 1:
	# 		double_names_lst.append(i)
	
	# pdb.set_trace()



		# found = data_dct['Patient'] == subject_nb
		# idx = found[found].index[0]
		# subjs_dct[subject] = data_dct['Patient'][idx]

	return subjs_labs


if __name__ == "__main__":
	main()