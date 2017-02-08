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
import dill as pickle
from sklearn import svm
from sklearn.decomposition import PCA
import progressbar
# see https://pypi.python.org/pypi/multiprocessing_on_dill
from multiprocessing_on_dill import Pool
#from functools import wraps
#from pathos.multiprocessing import ProcessingPool
#import jellyfish

from sklearn.model_selection import train_test_split

def run_and_time(method):

	#@wraps(method)
	
	def wrap_func(self,*method_args):
		t_start = time.time()
		method(self,*method_args)
		t_end = time.time()
		t_elapsed = t_end-t_start  # in seconds
		if t_elapsed < 60:
			form_str = "{} elapsed time {:.2f} s".format(u"\u2713", t_elapsed)
		else:
			form_str = "{} elapsed time {:.0f} m {:.0f} s".format(u"\u2713", t_elapsed//60, t_elapsed%60)
		print(form_str)
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
		self.train = pd.DataFrame()

		self.venue_names = defaultdict(lambda: defaultdict(list))
		self.team_names = defaultdict(lambda: defaultdict(list))
		self.comp_names = defaultdict(list)

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

		
		# def __make_dict_of_words_by_sport(dic):

		# 	no_comp_dic = defaultdict(list)

		# 	for sport in dic:
		# 		no_comp_dic[sport] = list(set([part_name.strip() for league in dic[sport] for w in dic[sport][league] for part_name in w.split() 
		# 			if part_name.strip() not in self.STPW and len(part_name.strip()) > 2]))

		# 	return no_comp_dic

		# def __make_list_of_words(lst):

		# 	return list(set([prt for w in lst for prt in w.split() if prt.strip() not in self.STPW and len(prt.strip()) > 2]))

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
	
		#print(self.comp_names)

		#self.team_names = __make_dict_by_sport_and_comp("tnam_")
		#self.sport_words = __make_dict_of_words_by_sport(self.team_names)

		# similarly, venue names
		# self.venue_names = __make_dict_by_sport_and_comp("vnam_")
		# self.venue_words = __make_dict_of_words_by_sport(self.venue_names)

		#self.horse_racing_comp_words = __make_list_of_words(self.COMPS_HORSE_RACING)

		#self.PCT_TRAIN = 0.70  # percentage of soccer/non-soccer for training


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

			# nonlocal st

			c = set([v for w in lst for v in w.split()]) & set(st.split())

			# print("st.split()=",st.split())
			# print("set(lst)=",set(lst))
			# print("c=",c)

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

			# for s in lst:
			# 	if (len(s.split()) > min_words - 1) or ("-" in s):  # if there are at least min_words in this name from list..
			# 		#st = re.sub("(?<!\w)" + s.lower() + "(?!\w+)", lab, st)
			# 		st = [w for w in st.split() if s not in ]
			# return st
	def __remove_duplicates_from_string(self, st):
		
		ulist = []
		[ulist.append(w) for w in st.split() if w not in ulist]  # note that comprehension or not, ulist grows

		return " ".join(ulist)

	def normalize_string(self, st):
		
		# make the string lower case, strip and replace all multiple white spaces with a single white space	
		# st = re.sub(r"\s+"," ",st.lower().strip())
		st = " ".join([t for t in [w.strip(".,:;") for w in st.lower().split()] if len(t) > 1])
			
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

		for sport in self.comp_names:
			st = self.__prelabel_from_list(st, self.comp_names[sport], "_" + sport.upper() + "_COMPETITION_", 2)

		for sport in self.venue_names:
			for comp in self.venue_names[sport]:
				st = self.__prelabel_from_list(st, self.venue_names[sport][comp], "_SPORTS_VENUE_", 2)

		st = self.__prelabel_from_list(st, self.AUS_THEATRE_COMPANIES, "_THEATRE_COMPANY_", 2)
		st = self.__prelabel_from_list(st, self.AUS_OPERA_COMPANIES, "_OPERA_COMPANY_", 2)
		st = self.__prelabel_from_list(st, self.COUNTRIES, "_COUNTRY_", 1)
		st = self.__prelabel_from_list(st, self.AUS_SUBURBS, "_AUS_SUB_CITY_", 1)

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

		df["venue"] = df["venue"].str.lower()

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

		# for col in ['event']:
		# 	df = df.iloc[:200,:]
		# 	df[col] = df[col].apply(lambda x: self.normalize_string(x))	

		df = self.parallelize_dataframe(df, self.normalize_df)	
		
		if k == "training":

			for col in ['event', 'venue']:
				for st_as_lst in df[col].str.split():
					for w in st_as_lst:
						if w not in self.STPW:
							self.train_dict[w] += 1

			self.train_rare_words = [w for w in self.train_dict if self.train_dict[w] < 3]
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

		print("normalisation completed.")
			
		return df

	def get_sports_word_features(self, st):

		res_dict = defaultdict()

		st_words = set(st.lower().split())  # only distinct words

		for sport in self.sport_words:
			in_both_sets = set(self.sport_words[sport]) & st_words
			if len(in_both_sets):
				res_dict[str(len(in_both_sets)) + "_[" + sport.upper() + "]_word"] = 1

		return res_dict

	def get__horse_racing_comp_word_features(self,st):

		res_dict = defaultdict(int)
		st_words = set(st.lower().split())  # only distinct words
		in_both_sets = set(self.horse_racing_comp_words) & st_words

		if len(in_both_sets):  # if any suburb names happen to be in this string
			res_dict[str(len(in_both_sets)) + "_HORSE_RACING_COMP_WORDS"] = 1
		
		return res_dict


	def get_1g_features(self, st):

		return {"word_[{}]".format(w.strip()): 1 for w in st.lower().split()}

	def get_2g_features(self, st):
		
		str_list = st.split()

		res_dict = defaultdict(int)
		
		if len(str_list) > 1:
			for i, w in enumerate(str_list):
				if i > 0:
					res_dict[("words_[{}]->[{}]").format(str_list[i-1], w)] = 1

		return res_dict

	def get_3g_features(self, st, nm):
		"""
		IN: string
		OUT: a dictionary containing 3-gram features extracted from the string
		"""
		str_list = st.split()

		return {("words_[{}]->[{}]->[{}]_in_" + nm).format(str_list[i-2],str_list[i-1],w): 1  for i, w in enumerate(str_list) if (len(str_list) > 2 and i > 1)}

	def get_suburb_features(self, st):

		res_dict = defaultdict(int)

		st_words = set(st.lower().split())  # only distinct words

		in_both_sets = set(self.AUS_SUBURBS) & st_words

		if len(in_both_sets):  # if any suburb names happen to be in this string
			res_dict[str(len(in_both_sets)) + "_AUSSIE_SUBURBS"] = 1
		
		return res_dict


	def get_event_timeofday_feature(self, hour):

		if (int(hour) >= 1) and (int(hour) < 12):
			time_of_day = "morning"
		elif  (int(hour) >= 12) and (int(hour) < 18):
			time_of_day = "afternoon"
		else:
			time_of_day = "evening"

		return {"event_time_[{}]".format(time_of_day): 1}

	def get_aus_team_nickname_features(self, st):

		
		res_doc = defaultdict(int)

		for nick_line in self.AUS_TEAM_NICKS:
			if nick_line:

				nick, sport = list(map(str.strip, nick_line.split("-")))

				if re.search("(?<!\w)" + nick + "(?!\w+)", st):  # if found nickname
					res_doc[sport.upper() + "_nick"] = 1

		return res_doc

	
	def extract_features_from_data(self, d, k="training"):

		"""
		extracts features from a data frame
		"""
		
		di = defaultdict(lambda: defaultdict(int))  # will keep extracted features here	
		
		print("extracting features...")
		with progressbar.ProgressBar(max_value=d.shape[0]) as progress:

			for i, s in enumerate(d.itertuples()):  #  go by rows
				
				pk = s.Index  # recall that pks are data frame index
				
				di[pk].update(self.get_sports_word_features(s.event + " " + s.venue))
				di[pk].update(self.get__horse_racing_comp_word_features(s.event + " " + s.venue))
				di[pk].update(self.get_event_timeofday_feature(s.hour))
				di[pk].update(self.get_aus_team_nickname_features(s.event))
				di[pk].update(self.get_suburb_features(s.event + " " + s.venue))
				di[pk].update(self.get_1g_features(s.event))
				di[pk].update(self.get_2g_features(s.event))

				if i%100 == 0 or i == d.shape[0]:
					progress.update(i)
	

		
		# merge the original data frame with a new one created from extracted features to make one feature data frame

		fdf = pd.concat([d[d.columns.difference(["event", "venue", "hour"])], pd.DataFrame.from_dict(di, orient='index')], axis=1, join_axes=[d.index]).fillna(0)
			
		if k == "training":
			self.model_feature_names = list(fdf)
			with open(self.feature_file, "w") as f:
				for feature in self.model_feature_names:
					f.write("{}\n".format(feature))

			print("model features...{}...ok".format(len(self.model_feature_names)))
		elif k == "testing":
			"""
			now we ignore features that are not in self.model_feature_names () 
			"""
			print("fdf.shape",fdf.shape)
			fdf.drop(fdf.columns.difference(self.model_feature_names), axis=1, inplace=True)
			print("dropped some columns, now testing shape", fdf.shape)

			for fch in self.model_feature_names:
				if fch not in fdf.columns:
					fdf[fch] = 0

		return  fdf
	

	# def train_and_test(self):

		print("training random forest...", end="")

		# forest = RandomForestClassifier(class_weight="auto")

		# forest.fit(self.train_nofeatures[self.model_feature_names], self.train_y)

		# print("ok")

		# print("predicting on testig data..", end="")
		# y1 = forest.predict(self.test_nofeatures[self.model_feature_names])
		# print("ok")

		# # df_with_prediction = pd.concat([self.test_nofeatures, pd.Series(y1, index=self.test_y.index)], axis=1, ignore_index=True)
		
		# # df_with_prediction.to_csv("kuki.csv")
		# print("training accuracy: {}".format(round(accuracy_score(self.train_y,forest.predict(self.train_nofeatures[self.model_feature_names])),2)))
		# print("accuracy: {}".format(round(accuracy_score(self.test_y,y1),2)))

		
		# imps = zip(list(train_df), clf.feature_importances_)   svm.LinearSVC()
		


if __name__ == '__main__':

	


	# initialize classifier
	cl = SportsClassifier()
	cl.read_data()
	# print(cl.normalize_string(" sydney swans vs.  brisbane roar 19pm sydney f. c. with John   hi Melbourne Storm! played in the resimax stakes... ben decided to perform at the a-league show as well as holden cup "))

	#sys.exit()

	# tlen = 0
	# for sport in cl.pks_dic:
	# 	tlen += len(cl.pks_dic[sport])
	# print("total pks:",tlen)

	cl.create_train_test()

	t0 = time.time()
	cl.train_nofeatures = cl.normalize_data(cl.train_nofeatures)
	t1 = time.time()

	sys.exit()

	X_train = cl.extract_features_from_data(cl.train_nofeatures)

	X_test = cl.extract_features_from_data(cl.test_nofeatures, k="testing")

	print("training SVC..")
	t0 = time.time()
	clf = svm.SVC()

	clf.fit(X_train, cl.y_train)
	t1 = time.time()
	
	print("elapsed time: {} minutes".format(round((t1-t0)/60,1)))
	print("done. now predicting...")

	y1 = clf.predict(X_test)

	print("training accuracy: {}".format(round(accuracy_score(cl.y_train,clf.predict(X_train)),2)))
	print("accuracy: {}".format(round(accuracy_score(cl.y_test,y1),2)))











