import os, sys, glob
import numpy as np
import pandas as pd
import pdb

from scipy import stats
from scipy.stats import chi2_contingency

sys.path.append('/home/fauvev01/BMEII-Functions/')

#Custom modules
import bmeii_io as bio

def main():

	# pdb.set_trace()

	data_dir = '/home/fauvev01/Bachir/SBRT/data/labels/SBRT Amine.xlsx'
	dct_data_raw = pd.read_excel(data_dir)

	#Categorical data
	dct_ethn = dct_data_raw['Ethnicity']
	dct_sex = dct_data_raw['Sex']
	dct_eti = dct_data_raw['Etiology']

	#Continuous data
	dct_age = dct_data_raw['Age']
	dct_size = dct_data_raw['Lesion Size (On PVP)']
	dct_dose = dct_data_raw['Treatment Dose']
	dct_preafp = dct_data_raw['pre-treatment AFP']
	dct_postafp = dct_data_raw['post treatment AFP 1']

	# pdb.set_trace()

	#Means
	mean_age = dct_age.mean()
	mean_size = dct_size.mean()
	mean_dose = dct_dose.mean()
	mean_preafp = dct_preafp.mean()
	mean_postafp = dct_postafp.mean()

	#Replacing nan by mean
	dct_age = dct_age.fillna(mean_age)
	dct_size = dct_size.fillna(mean_size)
	dct_dose = dct_dose.fillna(mean_dose)
	dct_preafp = dct_preafp.fillna(mean_preafp)
	dct_postafp = dct_postafp.fillna(mean_postafp)

	#Response
	dct_res = dct_data_raw['Response']
	dct_res = dct_res.replace(['Complete','Partial','Stable','Progression'],['Gp1','Gp1','Gp2','Gp2'])

	# pdb.set_trace()

	# Count the number of patients in each group
	print(dct_res.value_counts())

	# Count the number of patients of different ethnicity grouped by response
	print(dct_ethn.groupby(dct_res).value_counts())

	# Count the number of patients of different etiology grouped by response
	print(dct_eti.groupby(dct_res).value_counts())

	# Count the number of patients of different sex grouped by response
	print(dct_sex.groupby(dct_res).value_counts())

	pdb.set_trace()



	# #Chi square test for Categorical data
	cross_ethn_res = pd.crosstab(dct_ethn, dct_res)
	chi2, p, dof, expected = stats.chi2_contingency(cross_ethn_res)
	
	cross_eti_res = pd.crosstab(dct_eti, dct_res)
	chi2, p, dof, expected = stats.chi2_contingency(cross_eti_res)
	
	cross_sex_res = pd.crosstab(dct_sex, dct_res)
	chi2, p, dof, expected = stats.chi2_contingency(cross_sex_res)

	pdb.set_trace()

	#Fisher exact test for each ethnicity
	# hisp_rest = dct_ethn.replace(['Asian','Caucasian','African America',''],['Rest','Rest','Rest','Rest'])
	# asia_rest = dct_ethn.replace(['Hispanic','Caucasian','African America',''],['Rest','Rest','Rest','Rest'])
	# cauc_rest = dct_ethn.replace(['Asian','Hispanic','African America',''],['Rest','Rest','Rest','Rest'])
	# afra_rest = dct_ethn.replace(['Asian','Caucasian','Hispanic',''],['Rest','Rest','Rest','Rest'])

	# cross_hisp_rest = pd.crosstab(hisp_rest, dct_res)
	# cross_asia_rest = pd.crosstab(asia_rest, dct_res)
	# cross_cauc_rest = pd.crosstab(cauc_rest, dct_res)
	# cross_afra_rest = pd.crosstab(afra_rest, dct_res)

	# pdb.set_trace()

	# oddsr, p = stats.fisher_exact(cross_hisp_rest)
	# oddsr, p = stats.fisher_exact(cross_asia_rest)
	# oddsr, p = stats.fisher_exact(cross_cauc_rest)
	# oddsr, p = stats.fisher_exact(cross_afra_rest)

	# pdb.set_trace()

	# cross_sex_res = pd.crosstab(dct_sex, dct_res)
	# chi2, p, dof, expected = stats.chi2_contingency(cross_sex_res)

	# pdb.set_trace()

	# #Some expected values for the etiology are lower than 5 then we will use Fisher's Exact Test
	# hcv_rest = dct_eti.replace(['HBV','Alcohol','NASH','AIH','HB/CV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	# hbv_rest = dct_eti.replace(['HCV','Alcohol','NASH','AIH','HB/CV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	# alc_rest = dct_eti.replace(['HBV','HCV','NASH','AIH','HB/CV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	# nash_rest = dct_eti.replace(['HBV','Alcohol','HCV','AIH','HB/CV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	# aih_rest = dct_eti.replace(['HBV','Alcohol','NASH','HCV','HB/CV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	# hbcv_rest = dct_eti.replace(['HBV','Alcohol','NASH','AIH','HCV',''],['Rest','Rest','Rest','Rest','Rest','Rest'])
	
	# cross_hcv_rest = pd.crosstab(hcv_rest, dct_res)
	# cross_hbv_rest = pd.crosstab(hbv_rest, dct_res)
	# cross_alc_rest = pd.crosstab(alc_rest, dct_res)
	# cross_nash_rest = pd.crosstab(nash_rest, dct_res)
	# cross_aih_rest = pd.crosstab(aih_rest, dct_res)
	# cross_hbcv_rest = pd.crosstab(hbcv_rest, dct_res)

	# pdb.set_trace()

	# oddsr, p = stats.fisher_exact(cross_hcv_rest)
	# oddsr, p = stats.fisher_exact(cross_hbv_rest)
	# oddsr, p = stats.fisher_exact(cross_alc_rest)
	# oddsr, p = stats.fisher_exact(cross_nash_rest)
	# oddsr, p = stats.fisher_exact(cross_aih_rest)
	# oddsr, p = stats.fisher_exact(cross_hbcv_rest)

	# pdb.set_trace()

	#T-test for continuous data 
	age_gp1 = dct_age[dct_res == 'Gp1']
	age_gp2 = dct_age[dct_res == 'Gp2']
	size_gp1 = dct_size[dct_res == 'Gp1']
	size_gp2 = dct_size[dct_res == 'Gp2']
	dose_gp1 = dct_dose[dct_res == 'Gp1']
	dose_gp2 = dct_dose[dct_res == 'Gp2']
	preafp_gp1 = dct_preafp[dct_res == 'Gp1']
	preafp_gp2 = dct_preafp[dct_res == 'Gp2']
	postafp_gp1 = dct_postafp[dct_res == 'Gp1']
	postafp_gp2 = dct_postafp[dct_res == 'Gp2']

	pre_min_post_gp1 = preafp_gp1 - postafp_gp1
	pre_min_post_gp2 = preafp_gp2 - postafp_gp2

	pdb.set_trace()

	tval, pval = stats.ttest_ind(age_gp1,age_gp2)
	tval, pval = stats.ttest_ind(size_gp1,size_gp2)
	tval, pval = stats.ttest_ind(dose_gp1,dose_gp2)
	tval, pval = stats.ttest_ind(preafp_gp1,preafp_gp2)
	tval, pval = stats.ttest_ind(postafp_gp1,postafp_gp2)

	tval, pval = stats.ttest_ind(pre_min_post_gp1,pre_min_post_gp2)

	pdb.set_trace()

	print('>>> DONE <<<')



if __name__ == '__main__':
	main()