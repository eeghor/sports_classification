import json
import re
import pandas as pd
import itertools
from collections import defaultdict, Counter
import time
import random
import math 

"""
scikit-learn
"""

from sklearn.svm import SVC
from sklearn.linear_model import Lars
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier

import numpy as np
import sys
import dill as pickle
from sklearn import svm
from sklearn.decomposition import PCA
#import progressbar
# see https://pypi.python.org/pypi/multiprocessing_on_dill
from multiprocessing_on_dill import Pool
#from functools import wraps


from sklearn.model_selection import train_test_split

def run_and_time(method):

	#@wraps(method)
	
	def wrap_func(self,*method_args,**kwargs):
		t_start = time.time()
		res = method(self,*method_args,**kwargs)
		
		t_end = time.time()
		t_elapsed = t_end-t_start  # in seconds
		if t_elapsed < 60:
			form_str = "{} elapsed time {:.2f} s".format(u"\u2713", t_elapsed)
		else:
			form_str = "{} elapsed time {:.0f} m {:.0f} s".format(u"\u2713", t_elapsed//60, t_elapsed%60)
		print(form_str)
		return res

	return wrap_func


class SportsClassifier(object):

	def __init__(self):
		
		# sports currently supported; nonsport is everything we are NOT interested in at all
		self.sports = "nonsport afl soccer nrl rugby_union basketball netball tennis horse_racing".split()
		
		print("""
			|
			| SPORTS CLASSIFIER
			| 
			\n::: {} :::\n""".format(" ".join(self.sports[1:])))

		# specifically, team and individual sports (useful later on)
		self.team_sports = "afl soccer nrl rugby_union basketball netball".split()
		self.individual_sports = "tennis horse_racing".split()
		# encode sports like {"tennis": 3, ..}
		self.sports_encoded = {v:k for k,v in enumerate(self.sports)}
		# decode sports, e.g. {2: "soccer", ..}
		self.sports_decoded = {v: sport_name for sport_name, v in self.sports_encoded.items()}

		print("[initialising classifier]")

		# the raw data and data directories
		self.PKS_DIR = "pks/"

		self.DATA_DIR = "data/"
		self.data_file = self.DATA_DIR + "data.csv"
		self.train_target_file = self.DATA_DIR + "data_train_t.csv"
		self.test_target_file = self.DATA_DIR + "data_test_t.csv" 
		self.feature_file = self.DATA_DIR + "model_features.txt"
		
		self.RAW_DATA_DIR = "raw_data/"
		self.raw_data_file = self.RAW_DATA_DIR + "all_events.csv"

		self.TEMP_DATA_DIR = "temp_data/"	
		self.train_nofeatures_file = self.TEMP_DATA_DIR + "data_train_nofeatures.csv" 
		self.test_nofeatures_file = self.TEMP_DATA_DIR + "data_test_nofeatures.csv" 
		self.train_rare_words_file = self.TEMP_DATA_DIR  + "train_rare_words.txt"

		self.data_df = pd.DataFrame()
		self.train = pd.DataFrame()  # training data w features
		self.test = pd.DataFrame()  # testing data w features

		self.venue_names = defaultdict(lambda: defaultdict(list))
		self.team_names = defaultdict(lambda: defaultdict(list))
		self.team_name_words = defaultdict(list)
		self.comp_names = defaultdict(list)
		self.comp_name_words = defaultdict(list)

		# dictionary to store words from the processed training data frame
		self.train_dict = defaultdict(int)
		# model feature names - the same as the features extracted from the training set 
		self.model_feature_names = []

		self.pks_dic = {}
		# all pk files MUST be called pks_[SPORT].txt!
		self.pks_fl_dic = {sport: self.PKS_DIR + "pks_" + sport + ".txt" for sport in self.sports} 

		self.AUS_THEATRE_COMPANIES = frozenset()
		self.AUS_OPERA_COMPANIES = frozenset()
		self.NAMES_M = self.NAMES_F = self.STPW = self.AUS_SUBURBS = self.COUNTRIES = self.PERFORMERS = frozenset()


	def __read2list(self,filename, msg):
			"""
			read lines from a text file "filename" and put them into a list; 
			show message "msg" 
			"""
			with open(self.DATA_DIR + filename, "r") as f:
				lst = [line.strip() for line in f if line.strip()]
			print(msg + "...{}...ok".format(len(lst)))
			return frozenset(lst)

	# def run_and_time(self, *argv):

	# 	t_start = time.time()
	# 	def wrap_func(some_function):
	# 		return some_function(*argv)
	# 	return wrap_func


	@run_and_time
	def read_data(self):

		print("[reading data]")

		self.AUS_THEATRE_COMPANIES = self.__read2list("theatre_companies_australia.txt", "australian theatre companies")
		self.AUS_OPERA_COMPANIES = self.__read2list("opera_companies_australia.txt", "australian opera companies")
		self.NAMES_M = self.__read2list("names_m_5k.txt", "male names")
		self.NAMES_F = self.__read2list("names_f_5k.txt", "female names")
		self.STPW = self.__read2list("english_stopwords.txt", "english stopwords")
		# australian suburb/city list (by Australia Post, Sep 2016)
		self.AUS_SUBURBS = self.__read2list("aus_suburbs.txt", "australian suburbs")
		# load australian sports team nicknames
		self.AUS_TEAM_NICKS = self.__read2list("aus_team_nicknames.txt", "australian team nicknames")
		# load artist/musician list (from www.rollingstone.com)
		self.PERFORMERS = self.__read2list("performer_list.txt", "artists")
		self.COUNTRIES = self.__read2list("countries.txt", "world countries")
		# # rugby union competitions
		# self.COMPS_RU = self.__read2list("cnam_rugby_union.txt", "rugby union competitions")
		# self.COMPS_NRL = self.__read2list("cnam_nrl.txt", "rugby league competitions")
		# self.COMPS_TENNIS = self.__read2list("cnam_tennis.txt", "tennis competitions")
		# self.COMPS_SOCCER = self.__read2list("cnam_soccer.txt", "soccer competitions")
		# # horse racing competitions
		# self.COMPS_HORSE_RACING = self.__read2list("cnam_horse_racing.txt", "horse racing competitions")
		
		# dictionary to keep all sports primary keys (pks) 
		self.pks_dic = {sport: pd.read_csv(self.pks_fl_dic[sport], sep="\n", dtype=np.int32).drop_duplicates().ix[:,0].tolist() for sport in self.pks_fl_dic}
		
		# check if the pks happen to belong to some sport AND non-sport at the same time; if this is the case, remove that pk from non-sports
		print("checking for pks in multiple sports...", end="")
		for sport in self.pks_dic:
			for sport2 in self.pks_dic:
				if sport != sport2:
					cmn = set(self.pks_dic[sport]) & set(self.pks_dic[sport2])  # whick pks are in common
					if cmn:
						sys.exit("\nERROR: pks on two lists! in both {} and {}: {}".format(sport, sport2, ",".join([str(w) for w in cmn])))
		print("ok")
		
		#
		# create the data frame we will work with; it contains only the events we have pks for
		# note that we add a new column called "sport" containing sports codes speficied in self.sports_encoded
		#
		self.data_df = pd.concat([pd.read_csv(self.raw_data_file, parse_dates=["performance_time"], 
										infer_datetime_format=True, index_col="pk_event_dim").drop_duplicates(),
										pd.DataFrame([(pk, self.sports_encoded[sport]) for sport in self.pks_dic for pk in self.pks_dic[sport]], 
											columns=["pk_event_dim", "sport"]).set_index("pk_event_dim")],
												axis=1, join="inner")
		
		# add new columns (to be used as features)
		self.data_df = self.data_df.assign(month = pd.Series(self.data_df.performance_time.apply(lambda x: x.month)))
		self.data_df = self.data_df.assign(weekday = pd.Series(self.data_df.performance_time.apply(lambda x: x.weekday())))
		self.data_df = self.data_df.assign(hour = pd.Series(self.data_df.performance_time.apply(lambda x: x.hour)))  # Monday is 0
		
		# remove columns that are now useless
		self.data_df.drop(["performance_time"], axis=1, inplace=True)

		# save data to a CSV file
		print("saving data to {}...".format(self.data_file), end="")
		self.data_df.to_csv(self.data_file, columns="event venue month weekday hour sport".split())
		print("ok")

		"""
		create a dict of team names like {'soccer': {'england_championship': ['brighton & hove albion', 'newcastle united', 'reading',..],
											'australia_a_league': ['sydney fc', 'brisbane roar',..],..},
								  'rugby_union': {...}, ..}
		"""

		

		for team_sport in self.team_sports:
			# read the available team names to lists
				with open(self.DATA_DIR + "list_" + "tnam_" + team_sport + ".txt","r") as f_list:
					for fl in f_list:
						if fl.strip():
							with open(self.DATA_DIR + fl.strip(), "r") as f:
								self.team_names[team_sport][re.search("(?<=" + "tnam_" + ")\w+(?=.txt)",fl).group(0)] = \
																			[line.strip() for line in f if line.strip()]
		"""
		create a dictionary of team name words: {"soccer": ["sydney", "fc", "united",..], "basketball": ["bullets",..]}

		"""
		for team_sport in self.team_sports:
			for league in self.team_names[team_sport]:
				self.team_name_words[team_sport] = list({w.strip() for team in self.team_names[team_sport][league] for w in team.split() if w not in self.STPW})

		# print(self.team_name_words)

		"""
		create venue names just like the team names above
		"""

		

		for team_sport in self.team_sports:
			# read the available team names to lists
				with open(self.DATA_DIR + "list_" + "vnam_" + team_sport + ".txt","r") as f_list:
					for fl in f_list:
						if fl.strip():
							with open(self.DATA_DIR + fl.strip(), "r") as f:
								self.venue_names[team_sport][re.search("(?<=" + "vnam_" + ")\w+(?=.txt)",fl).group(0)] = \
																			[line.strip() for line in f if line.strip()]


		"""
		create a dictionary of competition names like {"soccer": ["a-league", "asian cup",..], "nrl": [..]}
		"""																	

		

		for sport in self.sports:
			if sport != "nonsport":
				with open(self.DATA_DIR + "cnam_" + sport + ".txt","r") as f:
					self.comp_names[sport] = [line.strip().lower() for line in f if line.strip()]

		for sport in self.sports:
			self.comp_name_words[sport] = list({w.strip() for comp in self.comp_names[sport]  for w in comp.split() if (w.strip() not in self.STPW) and len(w.strip()) > 2})

	
	@run_and_time
	def create_train_test(self):
		"""
		split into the training and teating sets; the 

		"""
		print("[creating the training and testing sets]")

		self.train_nofeatures, self.test_nofeatures, self.y_train, self.y_test = train_test_split(self.data_df.loc[:, "event venue month weekday hour".split()], self.data_df.sport, test_size=0.3, 
																stratify = self.data_df.sport, random_state=113)
		
		print("training data shape...{}x{}".format(*self.train_nofeatures.shape))
		print("saving training data (no features) to {}...".format(self.train_nofeatures_file), end="")
		
		pd.concat([self.train_nofeatures, self.y_train.apply(lambda _: self.sports_decoded[_])], axis=1, join="inner").to_csv(self.train_nofeatures_file)
		
		print("ok")
		print("testing data shape...{}x{}".format(*self.test_nofeatures.shape))
		print("saving testing data set to {}...".format(self.test_nofeatures_file), end="")
		self.test_nofeatures.to_csv(self.test_nofeatures_file)
		self.y_test.to_csv(self.test_target_file)
		print("ok")
	
	def __prelabel_from_list(self, st, lst,lab,min_words):

			c = set([v for w in lst for v in w.split()]) & set(st.split())

			if c:
				for s in c:
					for l in lst:
						if l.startswith(s):  # there is a chance..

							if (l in st) and ((len(l.split()) > min_words - 1) or ("-" in l)):
								st = st.replace(l,lab)
							else:
								pass
						else:
							pass
			else:
				pass

			return st

	def __remove_duplicates_from_string(self, st):
		
		ulist = []
		[ulist.append(w) for w in st.split() if w not in ulist]  # note that comprehension or not, ulist grows

		return " ".join(ulist)

	def normalize_string(self, st):
		
		# make the string lower case, strip and replace all multiple white spaces with a single white space	
		# st = re.sub(r"\s+"," ",st.lower().strip())
		st = " ".join([t for t in [w.strip(",:;") for w in st.split()] if len(t) > 1])
			
		# merge letters like f.c. -> fc or f c. -> fc
		#st = [for w in st.split()]
		#st = re.sub(r"(?<!\w)(\w{1})[.\s]{1,2}(\w{1})[.\s]{1}", r"\1\2", st)  
		# by now st only has single white spaces, e.g. a f c sydney game
		#st = [st.split()]
		
		st = self.__remove_duplicates_from_string(st)

		# remove all numbers or non-alphanumeric characters if they aren't part of word
		# st = re.sub(r"(?<!\w)[\d\W](?!\w+)","",st)
		
		for sport in self.team_names:
			for comp in self.team_names[sport]:
				st = self.__prelabel_from_list(st, self.team_names[sport][comp], "_" + sport.upper() + "_TEAM_", 2)

		for n in self.AUS_TEAM_NICKS:
			nick, sport = list(map(str.strip, n.split("-")))
			st = self.__prelabel_from_list(st, [nick], "_" + sport.upper() + "_TEAM_", 1)


		for sport in self.comp_names:
			st = self.__prelabel_from_list(st, self.comp_names[sport], "_" + sport.upper() + "_COMPETITION_", 2)

		for sport in self.venue_names:
			for comp in self.venue_names[sport]:
				st = self.__prelabel_from_list(st, self.venue_names[sport][comp], "_SPORTS_VENUE_", 2)

		st = self.__prelabel_from_list(st, self.AUS_THEATRE_COMPANIES, "_THEATRE_COMPANY_", 2)
		st = self.__prelabel_from_list(st, self.AUS_OPERA_COMPANIES, "_OPERA_COMPANY_", 2)
		st = self.__prelabel_from_list(st, self.COUNTRIES, "_COUNTRY_", 1)
		st = self.__prelabel_from_list(st, self.AUS_SUBURBS, "_AUS_LOCATION_", 1)

		st = self.__prelabel_from_list(st, self.PERFORMERS, "_ARTIST_", 2)
		
		st = self.__prelabel_from_list(st, self.NAMES_M, "_NAME_", 1)

		# remove stopwords
		st = " ".join([w for w in st.split() if w not in self.STPW])
		# for stop_word in self.STPW:
		# 	st = re.compile(r"(?<!\w)" + stop_word + "(?!\w+)").sub('', st) 

		st = re.compile(r"(?<!\w)\w*\d+\w*(?!\w+)").sub('', st)

		# remove the multiple white spaces again
		st = re.sub(r"\s+"," ",st)

		return st	

	def normalize_df(self, df):

		# do don't do much to the event column, just make it lower case
		df["event"] = df["event"].str.lower().str.split().str.join(" ")
		df["event"] = df["event"].str.replace("."," ")

		df["venue"] = df["venue"].str.lower().str.split().str.join(" ")
		# not everything is lower case and with only 1 white space between words

		# remove venue from event
		df["event"] = [_event.replace(_venue,"") for _event, _venue in zip(df["event"],df["venue"])]

		# process the event column
		df['event'] = df['event'].apply(lambda x: self.normalize_string(x))		

		return df


	def parallelize_dataframe(self, df, func):

   		df_split = np.array_split(df, 4)
   		pool = Pool(4)
   		df = pd.concat(pool.map(func, df_split))
   		pool.close()
   		pool.join()

   		return df


	@run_and_time
	def normalize_data(self, df, k="training"):

		print("[normalising {} data]".format(k))

		df = self.parallelize_dataframe(df, self.normalize_df)	
		
		if k == "training":

			for col in ['event', 'venue']:
				for st_as_lst in df[col].str.split():
					for w in st_as_lst:
							self.train_dict[w] += 1

			self.train_rare_words = [w for w in self.train_dict if self.train_dict[w] == 1]

			print("found {} rare words ({}% of all words in training set)...".format(len(self.train_rare_words), round(len(self.train_rare_words)/len(self.train_dict)*100.0),1))
			
			with open(self.train_rare_words_file, "w") as f:
				for w in self.train_rare_words:
					f.write("{}\n".format(w))

			self.train_word_list = [w for w in self.train_dict if w not in self.train_rare_words]

			print("words to be used to extract features: {}".format(len(self.train_word_list)))

		# only leave words that made it into self.train_word_list for ANY data frame, not just training
		for col in ['event', 'venue']:
			df[col] = df[col].apply(lambda x: " ".join([w for w in x.split() if  w in self.train_word_list]))

		df.to_csv(self.TEMP_DATA_DIR + "normalised_" + k + "_df.csv")

		return df

	def getf_special_word(self, st):

		res_dict = defaultdict()  # to keep collected features in

		for sport in self.team_name_words:

			in_both_sets = set(self.team_name_words[sport]) & set(st.lower().split()) 

			if in_both_sets:

				res_dict["@" + sport.upper() + "_team_words"] = len(in_both_sets)

		for sport in self.comp_name_words:

			in_both_sets = set(self.comp_name_words[sport]) & set(st.lower().split()) 

			if in_both_sets:

				res_dict["@" + sport.upper() + "_comp_words"] = len(in_both_sets)

		return res_dict


	def getf_1g(self, st):

		c = Counter(st.split())

		return {"@word_[{}]".format(w.strip()): c[w] for w in c}

	def getf_2g(self, st):

		res_dict = defaultdict(int)
		
		str_list = st.split()
		
		if len(str_list) > 1:
			for i, w in enumerate(str_list):
				if i > 0:
					res_dict[("@words_[{}]->[{}]").format(str_list[i-1], w)] += 1

		return res_dict

	def getf_timeofday(self, hour):

		if (int(hour) >= 1) and (int(hour) < 12):
			time_of_day = "morning"
		elif  (int(hour) >= 12) and (int(hour) < 18):
			time_of_day = "afternoon"
		else:
			time_of_day = "evening"

		return {"event_timeofday_[{}]".format(time_of_day): 1}


	@run_and_time
	def get_features(self, d, k="training"):

		"""
		extracts features from a data frame
		"""
		
		di = defaultdict(lambda: defaultdict(int))  # will keep extracted features here	
		
		print("[extracting {} features]".format(k))


		for i, s in enumerate(d.itertuples()):  #  go by rows
			
			pk = s.Index  # recall that pks are data frame index
			
			di[pk].update(self.getf_special_word(s.event))
			di[pk].update(self.getf_timeofday(s.hour))
			di[pk].update(self.getf_1g(s.event))
			di[pk].update(self.getf_2g(s.event))
	
		# merge the original data frame with a new one created from extracted features to make one feature data frame

		fdf = pd.concat([d[d.columns.difference(["event", "venue", "hour"])], 
							pd.DataFrame.from_dict(di, orient='index')], axis=1, join_axes=[d.index]).fillna(0)
			
		if k == "training":

			self.model_feature_names = list(fdf)
			print("model features...{}...ok".format(len(self.model_feature_names)))

			with open(self.feature_file, "w") as f:
				for feature in self.model_feature_names:
					f.write("{}\n".format(feature))

		elif k == "testing":
			"""
			now we ignore features that are not in self.model_feature_names () 
			"""

			fdf.drop(fdf.columns.difference(self.model_feature_names), axis=1, inplace=True)

			for fch in self.model_feature_names:
				if fch not in fdf.columns:
					fdf[fch] = 0

		return  fdf   # returns new data frame that has feature columns attached

	def run_classifier(self):

		from sklearn.multiclass import OneVsRestClassifier
		from sklearn.svm import LinearSVC
		from sklearn.metrics import accuracy_score
		
		classifier = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=4)
		classifier.fit(self.train, self.y_train)

		y_pred = classifier.predict(self.test)
		print("acuracy:", accuracy_score(self.y_test, y_pred))


if __name__ == '__main__':

	# initialize classifier
	cl = SportsClassifier()
	
	cl.read_data()

	cl.create_train_test()

	cl.train = cl.get_features(cl.normalize_data(cl.train_nofeatures, k="training"))
	cl.test = cl.get_features(cl.normalize_data(cl.test_nofeatures, k="testing"))

	cl.run_classifier()











