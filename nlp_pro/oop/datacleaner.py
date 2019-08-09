import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re
from sklearn import preprocessing

class Cleaner():
	path = ""	
	def __init__(self, path):
		self.path = path

	def Clean(self):

		data = pd.read_csv(self.path)

		data = data.drop('author', axis=1)


		data = data.drop(data[data.sentiment == 'anger'].index)
		data = data.drop(data[data.sentiment == 'boredom'].index)
		data = data.drop(data[data.sentiment == 'enthusiasm'].index)
		data = data.drop(data[data.sentiment == 'empty'].index)
		data = data.drop(data[data.sentiment == 'fun'].index)
		data = data.drop(data[data.sentiment == 'relief'].index)
		data = data.drop(data[data.sentiment == 'surprise'].index)
		data = data.drop(data[data.sentiment == 'love'].index)
		data = data.drop(data[data.sentiment == 'hate'].index)
		data = data.drop(data[data.sentiment == 'neutral'].index)
		data = data.drop(data[data.sentiment == 'worry'].index)

		data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))


		data['content'] = data['content'].str.replace('[^\w\s]',' ')



		stop = stopwords.words('english')
		data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


		from textblob import Word
		data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

		import re
		def del_repeat(text):
		    pattern = re.compile(r"(.)\1{2,}")
		    return pattern.sub(r"\1\1", text)

		data['content'] = data['content'].apply(lambda x: " ".join(del_repeat(x) for x in x.split()))

		freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]


		freq = list(freq.index)
		data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))



		lbl_enc = preprocessing.LabelEncoder()
		y = lbl_enc.fit_transform(data.sentiment.values)

		return data











