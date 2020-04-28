from data_prep import preprocess_data
from data_load import load_all
from doc2vec import Doc2Vec_Embedding
from svm import Support_Vector_Machine

import multiprocessing


def main():

	
    # with open("data/stopwords.txt", "r") as f:
    #     text = f.readlines()
    # stopwords = [n.rstrip("\n") for n in text]

    # path_train_pos = "data/train/pos.txt"
    # path_train_neg = "data/train/neg.txt"
    # path_train_unsup = "data/train/unsup.txt"

    # path_test_pos = "data/test/pos.txt"
    # path_test_neg = "data/test/neg.txt"

    # preprocess_data()	

    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, train_unsup_reviews = load_all()
    
    cores = multiprocessing.cpu_count() - 1

    embedding = Doc2Vec_Embedding(X_TRAIN,
                                   Y_TRAIN,
                                   X_TEST,
                                   Y_TEST,
                                   train_unsup_reviews,
                                   workers=cores)

    train_vecs, Y_train, test_vecs, Y_test = embedding.train_doc2vec()

    print("Train_vectors are: ", len(train_vecs))
    print("Test_vecs are: ", len(test_vecs))
    print("Y_train are: ", len(Y_train))
    print("Y_test are: ", len(Y_test))

    svm = Support_Vector_Machine(train_vecs, Y_TRAIN, test_vecs, Y_TEST)

    svm.train_model()
    svm.validate_model()

    predictions = svm.predict_input(test_vecs)

    print(predictions)

if __name__ == "__main__":
    main()
    