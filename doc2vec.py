import numpy as np
import os.path
from random import shuffle


# import gensim
import gensim
from gensim.models.callbacks import CallbackAny2Vec
import gensim.models.doc2vec




class EpochLogger(CallbackAny2Vec):
     '''Callback to log information about training'''

     def __init__(self):
         self.epoch = 0

     def on_epoch_begin(self, model):
         print("Epoch #{} start".format(self.epoch + 1))

     def on_epoch_end(self, model):
         print("Epoch #{} end".format(self.epoch + 1))
         self.epoch += 1
        


epoch_logger = EpochLogger()


class Doc2Vec_Embedding(object):

    """
    The Doc2Vec_Embedding class trains a doc2vec model on the corpus of reviews and returns matrix of doc2vec embedded vectors
    """
    def __init__(self, X_train, Y_train, X_test, Y_test, unlab_reviews,
                 dm=0, vec_size=200, min_count=7, window=10, sample=1e-1,
                 negative=5, alpha=0.025, alpha_min=0.0001,  
                 workers=1, epochs=20):
        # Define data structures as class variables
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.unlab_reviews = unlab_reviews
        self.workers = workers
        # Define internal Doc2vec parameters
        self.vec_size = vec_size
        self.min_count = min_count
        self.window = window
        self.sample = sample
        self.negative = negative
        self.workers = workers
        self.dm = dm
        self.alpha = alpha
        self.alpha_min = alpha_min

        # Set number of training epochs
        self.epochs = epochs


    @staticmethod
    def shuffle_docs(X, Y):
        # shuffles X and Y without changing the indexes
        shuffled_x = []
        shuffled_y = []
        all_idx = list(range(len(X)))
        shuffle(all_idx)

        for doc_id in all_idx:
            shuffled_x.append(X[doc_id])
            shuffled_y.append(Y[doc_id])

        return shuffled_x, shuffled_y


    def review_to_vec(self, model, review):
        # Convert review to vector using trained model
        vec = np.array(model.infer_vector(review.words)).reshape((1, self.vec_size))

        return vec


    def corpus_to_vec(self, model, corpus):
        # Convert the whole corpus to vectors
        vecs = [self.review_to_vec(model, review) for review in corpus]

        # Convert list to numpy array
        vec_arr = np.concatenate(vecs)

        return vec_arr


    def build_vocabulary(self):
        # Initialize doc2vec PV-DBOW
        model = gensim.models.Doc2Vec(min_count=self.min_count,
                                      window=self.window,
                                      vector_size=self.vec_size,
                                      sample=self.sample,
                                      negative=self.negative,
                                      workers=self.workers,
                                      dm=self.dm)

        # Build vocabulary over all reviews
        print ("Building vocabulary...\n")
        all_reviews = self.X_train + self.X_test + self.unlab_reviews
        model.build_vocab(all_reviews)

        return model


    def embed(self, model, X, Y, unlab_set=[]):
        # Run through the dataset multiple times, shuffling the data each time to improve accuracy
        train_reviews = X + unlab_set
        for epoch in range(self.epochs):
            print ("Training epoch: {0}/{1}".format(epoch+1, self.epochs))
            model.train(train_reviews, callbacks=[epoch_logger], total_examples=len(train_reviews), epochs=1)

            
            print ("Shuffling data...")
            X, Y = self.shuffle_docs(X, Y)
            shuffle(unlab_set)
            train_reviews = X + unlab_set

        print ("Calculating doc2vec vectors...")
        train_vecs = self.corpus_to_vec(model, X)
        print ("Done training...\n")

        return train_vecs, model, X, Y, unlab_set


    def train_doc2vec(self):
        if os.path.isfile("models/doc2vec_model.doc2vec"):
            print ("Doc2vec model is already trained. Loading existing model...")
            model = gensim.models.Doc2Vec.load("models/doc2vec_model.doc2vec")

            # Calculate doc2vec vectors
            print ("Calculating doc2vec vectors...")
            train_vecs = self.corpus_to_vec(model, self.X_train)
            test_vecs = self.corpus_to_vec(model, self.X_test)

            return (train_vecs,
                    self.Y_train,
                    test_vecs,
                    self.Y_test)

        else:
            print("The Doc2vec model has not been found.")
            # Initialize doc2vec model and build vocabulary
            model = self.build_vocabulary()

            # Train doc2vec model and get vector embeddings for training set
            print ("Training doc2vec model on training dataset...")
            train_vecs, model, self.X_train, self.Y_train, self.unlab_reviews = self.embed(model=model, 
                                                                                           X=self.X_train, 
                                                                                           Y=self.Y_train,
                                                                                           unlab_set=self.unlab_reviews)

            # Get vector embeddings for testing set
            print ("Training doc2vec model on testing dataset...")
            test_vecs, model, self.X_test, self.Y_test, _ = self.embed(model=model,
                                                                       X=self.X_test,
                                                                       Y=self.Y_test)

            # Create model directory if necessary
            if not os.path.isdir("models"):
                os.mkdir("models")

            # Save model for reusability
            model.save("models/doc2vec_model.doc2vec")

            return (train_vecs,
                    self.Y_train,
                    test_vecs,
                    self.Y_test)