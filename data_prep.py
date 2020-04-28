from pathlib import Path

from tqdm import tqdm
import string
import re

import pandas as pd




def create_necessary_paths():

	Path("data/train").mkdir(parents=True, exist_ok=True)
	Path("data/test").mkdir(parents=True, exist_ok=True)


def dataset_to_files(data):
	for index, row in tqdm(data.iterrows()):

		row["review"] = row["review"].rstrip()

		if row["type"] == "train":

			if row["label"] == "pos":
				
				with open(f"data/train/pos.txt", 'a+') as pos_file_name:
					pos_file_name.write(row["review"] + "\n")

			elif row["label"] == "neg":
				

				with open(f"data/train/neg.txt", 'a+') as neg_file_name:
					neg_file_name.write(row["review"] + "\n")

			elif row["label"] == "unsup":
				

				with open(f"data/train/unsup.txt", 'a+') as unsup_file_name:
					unsup_file_name.write(row["review"] + "\n")

		elif row["type"] == "test":

			if row["label"] == "pos":
				
				with open(f"data/test/pos.txt", 'a+') as pos_file_name:
					pos_file_name.write(row["review"] + "\n")

			elif row["label"] == "neg":
				

				with open(f"data/test/neg.txt", 'a+') as neg_file_name:
					neg_file_name.write(row["review"] + "\n")

			elif row["label"] == "unsup":
				

				with open(f"data/test/unsup.txt", 'a+') as unsup_file_name:
					unsup_file_name.write(row["review"] + "\n")


def read_data_from_txt(path_to_txt):

	list_of_reviews = []
	with open(path_to_txt, 'r', encoding="ISO-8859-1") as some_file:
		line = some_file.readline()
		while line:
			line = some_file.readline()
			list_of_reviews.append(line)

	print(f"Length of the {path_to_txt} reviews is: ", len(list_of_reviews))

	return list_of_reviews




def prepare_data(data, stopwords, lower = False):
    
    """
		This fucntion reads the data and removes the stopwords
		data: The path to the csv file that should be tokenized
		stopwords: List of stopwords that should be excluded
		lower: Can convert the text to lowercase
	"""
	

    if lower == True:
        data = [i.lower() for i in data]

    all_docs = []
    
    for entry in tqdm(data):
	

        paragraph_words = []
        entry = remove_non_ascii(entry)
        entry = remove_trailing_new_line(entry)
        entry = remove_punctuation(entry)
        entry = entry.split()

        for word in entry:
			
            
            if word not in stopwords:
                paragraph_words.append(word)

        all_docs.append(paragraph_words)

    return all_docs

def remove_trailing_new_line(document):
    document = document.replace('<br /><br />','')
    return document

def remove_punctuation(document):
	
	document = document.translate(str.maketrans('', '', string.punctuation))
	return document

def remove_non_ascii(text):
	return re.sub(r'[^\x00-\x7F]+','', text)

def tokenize(path_train_pos,path_train_neg,path_train_unsup,path_test_pos,path_test_neg,stopwords):

	pos_train_reviews = read_data_from_txt(path_train_pos)
	neg_train_reviews = read_data_from_txt(path_train_neg)
	unsup_train_reviews = read_data_from_txt(path_train_unsup)

	pos_test_reviews = read_data_from_txt(path_test_pos)
	neg_test_reviews = read_data_from_txt(path_test_neg)



	pos_train_tokens = prepare_data(pos_train_reviews, stopwords, lower = True)
	neg_train_tokens = prepare_data(neg_train_reviews, stopwords, lower = True)
	unsup_train_tokens = prepare_data(unsup_train_reviews, stopwords, lower = True)


	pos_test_tokens = prepare_data(pos_test_reviews, stopwords, lower = True)
	neg_test_tokens = prepare_data(neg_test_reviews, stopwords, lower = True)


	return pos_train_tokens, neg_train_tokens, unsup_train_tokens, pos_test_tokens, neg_test_tokens



def prep_files(pos_test_tokens, neg_test_tokens, pos_train_tokens, neg_train_tokens, unsup_train_tokens):
	with open(f"data/test-pos.txt", 'a+') as pos_test:
		for entry in pos_test_tokens:
			for word in entry:
				pos_test.write(word + " ")
			pos_test.write("\n")

	with open(f"data/test-neg.txt", 'a+') as neg_test:
		for entry in neg_test_tokens:
			for word in entry:
				neg_test.write(word + " ")
			neg_test.write("\n")

	with open(f"data/train-pos.txt", 'a+') as pos_train:
		for entry in pos_train_tokens:
			for word in entry:
				pos_train.write(word + " ")
			pos_train.write("\n")

	with open(f"data/train-neg.txt", 'a+') as neg_train:
		for entry in neg_train_tokens:
			for word in entry:
				neg_train.write(word + " ")
			neg_train.write("\n")

	with open(f"data/train-unsup.txt", 'a+') as unsup_train:
		for entry in unsup_train_tokens:
			for word in entry:
				unsup_train.write(word + " ")
			unsup_train.write("\n")

def save_all_docs(all_docs):
	with open(f"data/alldata-id.txt", 'a+') as all_documents:
		for i, entry in enumerate(all_docs):
			all_documents.write(f"_*{i} " + entry)

def preprocess_data():

    data = pd.read_csv("data/imdb_master.csv", encoding="ISO-8859-1")
    create_necessary_paths()
    path_train_pos = "data/train/pos.txt"
    path_train_neg = "data/train/neg.txt"
    path_train_unsup = "data/train/unsup.txt"

    path_test_pos = "data/test/pos.txt"
    path_test_neg = "data/test/neg.txt"
    dataset_to_files(data)

    with open("data/stopwords.txt", "r") as f:
        text = f.readlines()
    stopwords = [n.rstrip("\n") for n in text]
    pos_train_tokens, neg_train_tokens, unsup_train_tokens, pos_test_tokens, neg_test_tokens = tokenize(path_train_pos,path_train_neg,path_train_unsup,path_test_pos,path_test_neg,stopwords)


    print("The length of the training positive tokens is: ", len(pos_train_tokens))
    print("The length of the training negative tokens is: ", len(neg_train_tokens))
    print("The length of the training unsupervised tokens is: ", len(unsup_train_tokens))
    print("The length of the test positive tokens is: ", len(pos_test_tokens))
    print("The length of the test negative tokens is: ", len(neg_test_tokens))
			

    prep_files(pos_test_tokens, neg_test_tokens, pos_train_tokens, neg_train_tokens, unsup_train_tokens)


    test_pos = read_data_from_txt("data/test-pos.txt")
    test_neg = read_data_from_txt("data/test-neg.txt")
    train_pos = read_data_from_txt("data/train-pos.txt")
    train_neg = read_data_from_txt("data/train-neg.txt")
    train_unsup = read_data_from_txt("data/train-unsup.txt")


    all_docs = []
    all_docs = test_pos + test_neg + train_pos + train_neg + train_unsup

    print("The length of all documents that will be used for embedding is: ",len(all_docs))

    save_all_docs(all_docs)

if __name__ == "__main__":


	preprocess_data()	