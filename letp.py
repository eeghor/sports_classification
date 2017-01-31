import json
import re
import pandas as pd
import itertools
from collections import defaultdict, Counter
import time
import random
import math 
from sklearn.svm import SVC
from sklearn.linear_model import Lars
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
import numpy as np
import sys
from sklearn.decomposition import PCA
#import jellyfish

from sklearn.model_selection import train_test_split


class SportsClassifier(object):

	def __init__(self):

		self.dictionary = defaultdict(int)

		# USEFUL DIRECTORIES

		# the raw data and data directories
		self.DATA_DIR = "data/"
		self.RAW_DATA_DIR = "raw_data/"

		# raw data file
		self.unlabeled_data_csv = self.RAW_DATA_DIR + "all_events.csv"  # where the raw data is
		
		# expected column names in the raw data file
		self.RAW_COL_NAMES = ['event', 'venue', 'performance_time']

		# where to store the training and testing data sets
		self.train_data_csv = self.DATA_DIR + "training_data.csv" 
		self.testing_data_csv = self.DATA_DIR + "testing_data.csv" 
		
		# where to store the labelled data
		self.data_csv = self.DATA_DIR + "labeled_data.csv" 

		self.norm_data_csv = self.DATA_DIR + "labeled_norm_data.csv" 

		# where to store the model feature names
		self.model_feature_file = self.DATA_DIR + "model_features.txt"

		# extra info files
		self.aus_suburb_file = self.DATA_DIR + "aus_suburbs.txt"
		self.rugby_union_comps = self.DATA_DIR + "rugby_union_comps.txt"
		self.aus_team_nicknames_file = self.DATA_DIR + "aus_team_nicknames.txt"

		# primary keys (pks)
		self.negative_pks = []
		self.negative_pks_file = self.DATA_DIR + "confirmed_non_soccer_pks.txt"

		self.SOCC_PKS = []
		self.SOCC_PKS_file = self.DATA_DIR + "confirmed_soccer_pks.txt"
		self.AFL_PKS = []
		self.AFL_PKS_file = self.DATA_DIR + "confirmed_afl_pks.txt"

		self.aleague_teams_file = self.DATA_DIR + "aleague_teams.txt"
		self.countries_file = self.DATA_DIR + "countries.txt"
		self.COUNTRIES = []
		self.aleague_venues_file = self.DATA_DIR + "aleague_venues.txt"
		

		# word counter
		self.word_counter = defaultdict(int)  # {"me": 1322, "pidgeon":2, ..}

		self.aleague_team_names_obvious = []

		self.SP_ENC = {"AFL": 1, "soccer": 2, "NRL": 3, "other": 0}
		
		# AFL

		
		self.afl_teams_file = self.DATA_DIR + "afl_teams.txt"
		self.afl_teams = []
		self.AFL_PKS = []
		self.NB_AFL_PKS = 0

		self.afl_team_names_norm = []

		# NRL

		self.NRL_PKS_file = self.DATA_DIR + "confirmed_nrl_pks.txt"
		self.NRL_PKS = []

		# EXTRAS 

		self.ENG_STPWords_file = self.DATA_DIR + "english_stopwords.txt"


		


		self.PCT_TRAIN = 0.70  # percentage of soccer/non-soccer for training

		# model features is a dict like {"pk_event_dim_1":{"feature_1":1, "feature_2": 1,..}, "pk_event_dim_2":{...}}
		self.model_feature_names = []

		

		self.aleague_words = []
		# self.self.aleague_team_nicknames = {}

		self.raw_data_df = pd.DataFrame()
		self.aleague_team_nicknames = set()
		self.data_df = pd.DataFrame()
		self.raw_train_df = pd.DataFrame()
		self.raw_test_df = pd.DataFrame()
		self.train_data = pd.DataFrame()
		self.test_data = pd.DataFrame()
		self.train_y = pd.DataFrame()

		self.test_y= pd.DataFrame()
		self.train_target = pd.DataFrame()
		self.test_target = pd.DataFrame()

		self.train_df = pd.DataFrame()
		self.test_df = pd.DataFrame()

		self.months = "january february march april may june july august september october november december".split()
		self.months_short = [month[:3] for month in self.months]

		# these are NLTK stopwords (possibly, extended)
		self.ENG_STPW = []

		self.wkdays = "monday tuesday wednesday thursday friday saturday sunday".split()

	
	def process_raw_data(self):

		"""
		OUT: data frame with columns [pk_event_dim event venue year month weekday hour sport] where "sport" contains target labels
		NOTE: need to read the extra data BEFORE running this 

		"""

		print("---> step 2: processing raw data")

		# read from file; should obtain data frame with 3 columns, event, venue and performance_time (pk_event_index is the index)
		self.raw_data_df = pd.read_csv(self.unlabeled_data_csv, parse_dates=["performance_time"], infer_datetime_format=True, index_col="pk_event_dim")
		print("raw data shape is...{}x{}...ok".format(*self.raw_data_df.shape))

		# check if columns have right names
		raw_data_actual_col_names = set(list(self.raw_data_df))

		if set(self.RAW_COL_NAMES) - raw_data_actual_col_names:
			sys.exit("wrong raw data column names - please, check...")

		# drop duplicates if any
		rbef = self.raw_data_df.shape[0]
		self.raw_data_df.drop_duplicates(inplace=True)
		raft = self.raw_data_df.shape[0]
		print("dropped duplicates...{}...ok".format(rbef - raft))
		print("now data shape is...{}x{}...ok".format(*self.raw_data_df.shape))
		# label the raw data: add a new column containing a sport label for each row where it is known;
		# the rows that we don't know what sport they are will be discarded

		all_labelled_pks = pd.concat([pd.Series([self.SP_ENC["AFL"] for _ in self.AFL_PKS], index=self.AFL_PKS),
									pd.Series([self.SP_ENC["soccer"] for _ in self.SOCC_PKS], index=self.SOCC_PKS), 
										pd.Series([self.SP_ENC["other"] for _ in self.negative_pks], index=self.negative_pks)])
		all_labelled_pks.name = "sport"

		# choose only the subset of the original data frame corresponding to all_labelled_pks

		print("adding sports labels to data...")
		self.data_df = pd.concat([self.raw_data_df, all_labelled_pks], axis=1, join='inner')
		print("now data shape is...{}x{}...ok".format(*self.data_df.shape))

		# create new columns that will be used as variables
		print("adding 4 new columns...", end="")
		self.data_df = self.data_df.assign(year = pd.Series(self.data_df.performance_time.apply(lambda x: x.year)))
		self.data_df = self.data_df.assign(month = pd.Series(self.data_df.performance_time.apply(lambda x: x.month)))
		self.data_df = self.data_df.assign(weekday = pd.Series(self.data_df.performance_time.apply(lambda x: x.weekday())))
		self.data_df = self.data_df.assign(hour = pd.Series(self.data_df.performance_time.apply(lambda x: x.hour)))  # Monday is 0
		print("ok")
		print("now data shape is...{}x{}...ok".format(*self.data_df.shape))
		# remove columns that are now useless
		print("dropping performance_time...")
		self.data_df.drop(["performance_time"], axis=1, inplace=True)
		
		print("saving to {}...".format(self.data_csv), end="")
		self.data_df.to_csv(self.data_csv, columns="pk_event_dim event venue year month weekday hour sport".split(), sep="\t", header=True)
		print("ok")

	def read_data(self):
		
		print("---> step 1: reading data")	

		def read_file_w_header(file_name, whats_there):
			
			print(whats_there + "...", end="")
			
			with open(file_name, "r") as f:

				all_lines = f.readlines()

			hrd = all_lines[0]
			out_lst = list(set([line.strip() for line in all_lines[1:] if line.strip()]))

			# convert to int if needed

			if sum([w.isdigit() for w in out_lst]) == len(out_lst):
				out_lst = [int(w) for w in out_lst]
			
			with open(file_name, "w") as f:
				
				f.write(hrd)
				for j in out_lst:
					f.write("{}\n".format(j))

			print(str(len(out_lst)) + "...ok")

			return out_lst

		def do_pks_overlap(pks1, pks2, spo1, spo2):

			idx_common = set(pks1) & set (pks2)

			if idx_common:
				sys.exit("[!] error: the following {} pks are both in {} and {}: {}".format(len(idx_common), spo1, spo2, idx_common))
			else:
				return 1


		self.SOCC_PKS = read_file_w_header(self.SOCC_PKS_file, "soccer")
		self.AFL_PKS = read_file_w_header(self.AFL_PKS_file, "AFL")
		self.NRL_PKS = read_file_w_header(self.NRL_PKS_file, "NRL")
		self.ENG_STPW = read_file_w_header(self.ENG_STPWords_file, "stopwords")
		self.negative_pks = read_file_w_header(self.negative_pks_file, "negative pks")
		self.COUNTRIES = read_file_w_header(self.countries_file, "world countries")

		# check if there are any pks that are present in several lsits at once, e.g. are both in soccer and afl
		
		do_pks_overlap(self.SOCC_PKS, self.AFL_PKS, "soccer", "AFL")
		do_pks_overlap(self.SOCC_PKS, self.NRL_PKS, "soccer", "NRL")
		do_pks_overlap(self.SOCC_PKS, self.negative_pks, "soccer", "negatives")
		do_pks_overlap(self.AFL_PKS, self.negative_pks, "AFL", "negatives")

		# load a-league venue information (list of dicts)
		print("a-league venues...", end="")
		with open(self.aleague_venues_file, "r") as f:
			self.aleague_venues = json.load(f)
		print("{}...ok".format(len(self.aleague_venues)))

		# load australian sports organisation information (list of dicts)
		# print("aussie sport organisations...", end="")
		# with open(self.aus_sports_orgs_file, "r") as f:
		# 	self.aus_sports_orgs = json.load(f)
		# print("{}...ok".format(len(self.aus_sports_orgs)))

		# load australian sport team nicknames (list of dicts)
		print("aussie sport team nicknames...", end="")
		with open(self.aus_team_nicknames_file, "r") as f:
			self.aus_team_nicknames = json.load(f)
		print("{}...ok".format(len(self.aus_team_nicknames)))

		# load a-league team information (list of dicts)
		print("a-league teams...", end="")
		with open(self.aleague_teams_file, "r") as f:
			self.aleague_teams = json.load(f)
		print("{}...ok".format(len(self.aleague_teams)))

		print("afl teams...", end="")
		with open(self.afl_teams_file, "r") as f:
			self.afl_teams = json.load(f)
		print("{}...ok".format(len(self.afl_teams)))

		self.ru_comps = read_file_w_header(self.rugby_union_comps, "rugby union competitions")
		self.aus_suburbs = read_file_w_header(self.aus_suburb_file, "australian suburbs")

		self.aleague_team_names = {nam["name"] for nam in self.aleague_teams} 
		self.aleague_team_nicknames = {v for nam in self.aleague_teams for v in nam["nickname"] if isinstance(nam["nickname"], list) } | \
									{nam["nickname"] for nam in self.aleague_teams if not isinstance(nam["nickname"],list) }

		self.aleague_words = {v for w in list(self.aleague_team_names | self.aleague_team_nicknames) for v in w.split() if v not in self.ENG_STPW}

		self.afl_team_names = {nam["name"] for nam in self.afl_teams} 
		#print("self.afl_team_names=",self.afl_team_names)

		self.afl_words = {v for w in list(self.afl_team_names) for v in w.split() if v not in self.ENG_STPW}
		#print("self.afl_words =",self.afl_words )

		self.aleague_team_names_obvious = list(set(self.aleague_team_names) - set(self.afl_team_names))
		#print("self.aleague_team_names_obvious=",self.aleague_team_names_obvious)

		self.afl_team_names_norm = list(set(self.afl_team_names - set(self.aleague_team_names)))
		#print("self.afl_team_names_norm=",self.afl_team_names_norm)

		self.afl_soccer_words = list(set(self.afl_words - set(self.aleague_words)))
		#print("self.afl_soccer_words =",self.afl_soccer_words )	


	def create_train_test(self):

		print("---> step 3: creating training and testing data")

		self.train_data, self.test_data, self.train_y, self.test_y = train_test_split(self.data_df.ix[:, self.data_df.columns != "sport"], 
																		self.data_df.loc[:,"sport"], test_size=1.0-self.PCT_TRAIN, random_state=111, 
																			stratify=self.data_df.loc[:,"sport"])
	
		print("training data...{}x{}...ok".format(*self.train_data.shape))
		print("saving to {}...".format(self.train_data_csv), end="")
		pd.concat([self.train_data, self.train_y], axis=1).to_csv(self.train_data_csv)
		print("ok")
		print("testing data...{}x{}...ok".format(*self.test_data.shape))
		print("saving to {}...".format(self.testing_data_csv), end="")
		pd.concat([self.test_data, self.test_y], axis=1).to_csv(self.testing_data_csv)
		print("ok")


	def normalize_data(self, df):

		
		for col in ['event', 'venue']:

			# make everything lower case
			df[col] = df[col].str.lower()

			# remove all non-alphanumeric characters (i.e. all NOT in [a-zA-Z0-9_])
			df[col] = df[col].str.replace("\W"," ")

			# find what might be a year and replace it with a label
			df[col] = df[col].str.replace("(1{1}9{1})|(2{1}0{1})\d{2}","_YEAR_")

			# remove stop words if surrounded by white spaces
			for w in self.ENG_STPW:
				df[col] = df[col].str.replace(" " + w + " ", " ")

			# if there's an exact match of full a-league name, replace with a label
			for t in self.aleague_team_names_obvious:
				if len(t.split()) > 1 and sum([len(w) > 2 for w in t.split()]) == len(t.split()):  # if the team name contains at least 2 words and not st
					df[col] = df[col].str.replace(t, "_ALEAGUE_TEAM_")
			# same for AFL teams
			for afl_team in self.afl_team_names_norm:
				if len(afl_team.split()) > 1 and sum([len(w) > 2 for w in afl_team.split()]) == len(afl_team.split()):
					df[col] = df[col].str.replace(afl_team, "_AFL_TEAM_")

			for country in self.COUNTRIES:
				df[col] = df[col].str.replace(country, "_COUNTRY_")


			# remove weekdays
			for wd in self.wkdays:
				df[col] = df[col].str.replace(wd, wd[:3])
				df[col] = df[col].str.replace(wd[:3], " ")
			
			# remove months
			for wd in self.months:
				df[col] = df[col].str.replace(wd, wd[:3])
				df[col] = df[col].str.replace(wd[:3], " ")

			# remove remaining numbers
			df[col] = df[col].str.replace("\d","")
			
			# replace all multiple whitespaces with a sigle white space
			df[col] = df[col].str.replace("\s+"," ") 

			# merge some letters; may result in nonsense
			df[col] = df[col].str.replace(r"(?<!\w)(\w)[.\s](\w)\.", r"\1\2")  # f.c. -> fc or f c. -> fc


		return df

	def get_aleague_words_feature(self, str, nm):

		num_aleague_words = 0

		for str_word in str.split():
			if str_word in self.aleague_words:
				num_aleague_words += 1

		return {("aleague_words_in_" + nm): num_aleague_words}

	def get_1g_features(self, str, nm):
		"""
		IN: string
		OUT: a dictionary containing unigram features extracted from the string
		"""

		return {("word_[{}]_in_" + nm).format(w): c for w, c in Counter(str.split()).items()}

	def get_2g_features(self, str, nm):
		"""
		IN: string
		OUT: a dictionary containing bigram features extracted from the string
		"""
		str_list = str.split()

		res_dict = defaultdict(int)
		
		if len(str_list) > 1:
			for i, w in enumerate(str_list):
				if i > 0:
					res_dict[("words_[{}]->[{}]_in_" + nm).format(str_list[i-1], w)] += 1

		# {("words_[{}]->[{}]_in_" + nm).format(str_list[i-1], w): 1  for i, w in enumerate(str_list) if (len(str_list) > 1 and i > 0)}

		return res_dict

	def get_3g_features(self, str, nm):
		"""
		IN: string
		OUT: a dictionary containing 3-gram features extracted from the string
		"""
		str_list = str.split()

		return {("words_[{}]->[{}]->[{}]_in_" + nm).format(str_list[i-2],str_list[i-1],w): 1  for i, w in enumerate(str_list) if (len(str_list) > 2 and i > 1)}

	def get_suburb_features(self, some_string):

		num_suburbs = 0

		suburb_count_dict = defaultdict(int)

		for suburb in self.aus_suburbs:
			if suburb in some_string:
				num_suburbs += 1

		#print("returning:",{"number_suburbs": num_suburbs})
		
		return {"number_suburbs": num_suburbs}

	def get_country_features(self, some_string):

		num_countries = 0

		suburb_count_dict = defaultdict(int)

		for country in self.COUNTRIES:
			if country in some_string:
				num_countries += 1

		#print("returning:",{"number_suburbs": num_countries})
		
		return {"number_countries": num_countries}

	# def get_aus_city_features(self, str, nm):

	# 	str_list = str.split()

	# 	cities_found = set(self.aus_cities) & set(str_list)  # may be empty, of course

	# 	res_dict = {}

	# 	if len(cities_found) == 1:  # only 1 city detected
	# 		res_dict.update({("one_city_in_" + nm): 1})
	# 	elif len(cities_found) == 2:  # 2 differnt cities mentiones
	# 		res_dict.update({("two_cities_in_" + nm): 1})
	# 	else:
	# 		pass


		return res_dict

	def get_event_timeofday_feature(self, hour):

		if (int(hour) >= 1) and (int(hour) < 12):
			time_of_day = "morning"
		elif  (int(hour) >= 12) and (int(hour) < 18):
			time_of_day = "afternoon"
		else:
			time_of_day = "evening"

		return {"event_time_[{}]".format(time_of_day): 1}

	def get_event_code_feature(self, ecode):

		"""
		use all letters from the start to the first number as feature
		"""

		return {"ecode_[{}]".format(re.search("^([a-z]+)", ecode.lower()).group(0)): 1}


	def get_aus_team_nickname_features(self, str, nm):

		str_list = str.split()

		team_nicks = [nik["nickname"] for nik in self.aus_team_nicknames] # recall nik are dictionaries

		res_dict = {}

		nicks_found = set(team_nicks) & set(str_list)

		if len(nicks_found) == 1:  # only 1 city detected
			res_dict.update({("one_team_nick_in_" + nm): 1})
		elif len(nicks_found) == 2:  # 2 differnt cities mentiones
			res_dict.update({("two_team_nicks_in_" + nm): 1})
		else:
			pass

		# also create the sport name word features
		if nicks_found:
			for nik_found in nicks_found:
				for nik in self.aus_team_nicknames:
					if nik["nickname"] == nik_found:
						res_dict.update({("word_[{}]_in_" + nm).format(nik["sports"]): 1})


		return res_dict

	def get_aleague_team_features(self, str, nm):
		
		# recall nam are dictionaries

		res_dict = {}

		aleague_teams_found = []

		for aleague_team in self.aleague_team_names:
			if re.search(aleague_team, str):
				aleague_teams_found.append(aleague_team)
		
		if len(aleague_teams_found) == 1:
			res_dict.update({("one_aleague_team_in_" + nm): 1})
		elif len(aleague_teams_found) == 2:
			res_dict.update({("two_aleague_teams_in_" + nm): 1})
		else:
			pass
		
		

		if aleague_teams_found:
			#print("aleague_teams_found=",aleague_teams_found)
			for team_found in aleague_teams_found:
				for tname in self.aleague_teams:
					if tname["name"] == team_found:
						# look into team's nicknames; these can be a sting or list of strings
						if isinstance(tname["nickname"], list):  # suppose it's a list
							for nik in tname["nickname"]:
								for nik_part in nik.split():
									if nik_part not in self.ENG_STPW:
										res_dict.update({("word_[{}]_in_" + nm).format(nik_part): 1})
						else:  # if nickname is just a string
							for nik_part in tname['nickname'].split():
									if nik_part not in self.ENG_STPW:
										res_dict.update({("word_[{}]_in_" + nm).format(nik_part): 1})
						
						# add city where the team is based as a word feature
						res_dict.update({("word_[{}]_in_" + nm).format(tname["city"]): 1})
						# add home venue features; home_venue may be a string or list of strings
						if isinstance(tname["home_venue"], list):  # suppose it's a list
							for ven in tname["home_venue"]:
								for ven_part in ven.split():
									if ven_part not in self.ENG_STPW:
										res_dict.update({("word_[{}]_in_" + nm).format(ven_part): 1})
								
								# also do the bigrams
								ven_as_list = [w for w in ven.split() if w.lower() not in self.ENG_STPW]

								if len(ven_as_list) == 2:
									res_dict.update({("words_[{}]->[{}]_in_" + nm).format(*ven_as_list): 1})
								elif len(ven_as_list) == 3:
									res_dict.update({("words_[{}]->[{}]->[{}]_in_" + nm).format(*ven_as_list): 1})

						else:  # if venue is just a string
							ven_as_list = [w for w in tname['home_venue'].split() if w.lower() not in self.ENG_STPW]

							if len(ven_as_list) == 2:
								res_dict.update({("words_[{}]->[{}]_in_" + nm).format(*ven_as_list): 1})
							elif len(ven_as_list) == 3:
								res_dict.update({("words_[{}]->[{}]->[{}]_in_" + nm).format(*ven_as_list): 1})
		return res_dict

	
	def extract_features(self, d, k="train"):

		if k == "train":

			for l in ["event", "venue"]:
				for lst in d.loc[:,l].str.split():
					 for w in lst:
					 	self.dictionary[w] +=1 

			# remove words tha toccur only once
			for l in ["event", "venue"]:

				for word in self.dictionary:
					if self.dictionary[word] == 1:
						d.loc[:,l] = d.loc[:,l].str.replace(word,"")

		feature_dict = defaultdict(lambda: defaultdict(int))

		def get_features_from_string(st, lab):

			di = defaultdict(int)

			#di.update(self.get_1g_features(st, lab))
			di.update(self.get_2g_features(st, lab))
			#di.update(self.get_3g_features(st, lab))
			#di.update(self.get_aus_city_features(st, lab))
			di.update(self.get_aus_team_nickname_features(st, lab))
			#di.update(self.get_aleague_team_features(st, lab))
			di.update(self.get_aleague_words_feature(st, lab))

			return di

		for s in d.itertuples():

			pk = s.Index

			feature_dict[pk].update(get_features_from_string(s.event, "event"))
			feature_dict[pk].update(get_features_from_string(s.venue, "venue"))

			feature_dict[pk].update(self.get_suburb_features(s.event + s.venue))
			feature_dict[pk].update(self.get_country_features(s.event))
			
			feature_dict[pk].update(self.get_event_timeofday_feature(s.hour))

		# make the collected features the model features if these have been collected from the
		# training data

		fdf = pd.concat([d.loc[:, ["year", "month", "weekday"]], pd.DataFrame.from_dict(feature_dict, orient='index'), d.loc[:, ["sport"]]], axis=1, join_axes=[d.index]).fillna(0)
			
		if k=="train":
			self.model_feature_names = list(fdf)
			with open(self.model_feature_file, "w") as f:
				for feature in self.model_feature_names:
					f.write("{}\n".format(feature))

			print("model features...{}...ok".format(len(self.model_feature_names)))


		return  fdf
	

	def adjust_test_features(self):

		test_picked_features = list(self.test_data)


		features_to_add_to_test = list(set(self.model_feature_names) - set(test_picked_features))
		features_to_remove_from_test = list(set(test_picked_features) - set(self.model_feature_names))

		if features_to_remove_from_test:
			self.test_data.drop(features_to_remove_from_test, axis=1, inplace=True)
		if features_to_add_to_test:
			self.test_data = pd.concat([self.test_data, pd.DataFrame(np.zeros(shape=(self.test_data.shape[0],len(features_to_add_to_test))), 
														columns=features_to_add_to_test, index=self.test_data.index)], axis=1, join_axes=[self.test_data.index])

	def train_and_test(self):

		print("training random forest...", end="")

		# forest = RandomForestClassifier(class_weight="auto")

		# forest.fit(self.train_data[self.model_feature_names], self.train_y)

		# print("ok")

		# print("predicting on testig data..", end="")
		# y1 = forest.predict(self.test_data[self.model_feature_names])
		# print("ok")

		# # df_with_prediction = pd.concat([self.test_data, pd.Series(y1, index=self.test_y.index)], axis=1, ignore_index=True)
		
		# # df_with_prediction.to_csv("kuki.csv")
		# print("training accuracy: {}".format(round(accuracy_score(self.train_y,forest.predict(self.train_data[self.model_feature_names])),2)))
		# print("accuracy: {}".format(round(accuracy_score(self.test_y,y1),2)))

		from sklearn import svm
		# imps = zip(list(train_df), clf.feature_importances_)   svm.LinearSVC()
		print("training SVC..")
		clf = svm.SVC(decision_function_shape='ovo')
		print("done. now predicting...")
		#print(self.train_data[self.model_feature_names].astype(bool).sum(axis=1))
		clf.fit(self.train_data[self.model_feature_names], self.train_y)
		y1 = clf.predict(self.test_data[self.model_feature_names])

		print("training accuracy: {}".format(round(accuracy_score(self.train_y,clf.predict(self.train_data[self.model_feature_names])),2)))
		print("accuracy: {}".format(round(accuracy_score(self.test_y,y1),2)))


		# imps_sorted = sorted([(f,i) for f, i in imps], key=lambda x: x[1], reverse=True)

		# for f in imps_sorted[:30]:
		# 	print("feature: {}, importance: {}".format(f[0],round(f[1],3)))

		# # grb = GradientBoostingClassifier()
		# grb.fit(train_df[se], self.train_y)
		# y_grb = grb.predict(test_df[se])
		# print("gradient boosting accuracy: {}".format(round(accuracy_score(self.test_y,y_grb),2)))

		# print("doing bagging...")
		# bgg = BaggingClassifier(n_estimators = 20, n_jobs=2, max_samples=0.5)
		# bgg.fit(train_df[se], self.train_y)
		# y_bgg = bgg.predict(test_df[se])

		# print("bagging accuracy: {}".format(round(accuracy_score(self.test_y,y_bgg),2)))

		# df_with_prediction = pd.concat([self.test_data, pd.Series(y_bgg, index=self.test_y.index)], axis=1, ignore_index=True)
		
		# df_with_prediction.to_csv("kuki_bgg.csv")


if __name__ == '__main__':

	# initialize classifier
	cl = SportsClassifier()
	# read in all required data
	cl.read_data()
	cl.process_raw_data()
	

	cl.data_df = cl.normalize_data(cl.data_df)
	print("saving to {}...".format(cl.norm_data_csv), end="")		
	cl.data_df.to_csv(cl.norm_data_csv, columns="pk_event_dim event venue year month weekday hour sport".split(), sep="\t", header=True)
	print("ok")

	t0=time.time()
	cl.data_df = cl.extract_features(cl.data_df, k="train")
	t1=time.time()

	print("done.elapsed time {} minutes".format(round((t1-t0)/60,1)))

	cl.create_train_test()

	sys.exit("!!")
	

	

	cl.train_data = cl.normalize_data(cl.train_data)

	print(cl.train_data.head())
	
	#sys.exit("stop here")
	
	t0 = time.time()
	cl.train_data = cl.extract_features(cl.train_data)
	t1 = time.time()
	print("done. elapsed time {} minutes...".format(round((t1-t0)/60,1)))

	
	# initialize the PCA
	pca = PCA()

	# t0 = time.time()
	# cl.test_data = cl.extract_features(cl.test_data, k="test")
	# t1 = time.time()
	# print("done. elapsed time {} minutes...".format(round((t1-t0)/60,1)))

	# cl.adjust_test_features()

	# #print(cl.test_data.memory_usage())

	# cl.train_and_test()




	# # create the training and test data frames
	#cl.create_training_test_dfs()
	#cl.raw_data_df = cl.normalize_data(cl.raw_data_df)
	#print(cl.raw_data_df.head())
	#cl.train_and_test()

	#cl.train_and_test()





