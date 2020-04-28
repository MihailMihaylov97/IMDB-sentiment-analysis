# sklearn imports
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt

import os.path
import _pickle as cPickle

class Support_Vector_Machine(object):
    """
    The Support_Vector_Machine class trains a support vector machine classifier, 
    employing stochastic gradient descent with L2 regularization.
    """
    def __init__(self, train_vecs, Y_TRAIN, test_vecs, Y_TEST,
                 C=1, kernel="RBF", gamma="scale", probability=True):

        # data structures to be used

        self.train_vecs = train_vecs
        self.Y_TRAIN = Y_TRAIN
        self.test_vecs = test_vecs
        self.Y_TEST = Y_TEST


        # SVM variables
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability

    @staticmethod
    def save_model(model):
        # Save model for reusability
        with open("models/svm_model.pkl", 'wb') as my_model:
            cPickle.dump(model, my_model)


    @staticmethod
    def use_model():
        # Load saved model
        with open("models/svm_model.pkl", 'rb') as pretrained_model:
            model = cPickle.load(pretrained_model)
        return model


    def train_model(self):
        if not os.path.isfile("models/svm_model.pkl"):
            print ("Training support vector machine classifier...")
            # Initialize support vector machine classifier
            support_vector_classifier = SVC(C=1, kernel="rbf", gamma="scale", probability=True)

            # Define a parameter grid to search over
            param_grid = {'kernel':['rbf', 'poly'], 'C':[1]}

            # Setup 5-fold stratified cross validation
            cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

            # Perform grid search over hyperparameter configurations with 5-fold cross validation for each
            hyper_param = GridSearchCV(support_vector_classifier,
                               param_grid=param_grid,
                               cv=cross_validation,
                               n_jobs=5,
                               verbose=5)
            hyper_param.fit(self.train_vecs, self.Y_TRAIN)

            # Extract best estimator and save model
            self.save_model(hyper_param.best_estimator_)
        else:
            print ("Support vector machine classifier classifier is already trained...")


    def validate_model(self):
        print ("Validating classifier...")

        # Load classifier
        model = self.use_model()

        # Classify test dataset
        Y_PRED = model.predict(self.test_vecs)

        # Calculate AUC score
        roc_auc = roc_auc_score(self.Y_TEST, Y_PRED)

        # Print full classification report
        class_rep = classification_report(self.Y_TEST, Y_PRED, target_names=["negative", "positive"])
        print(class_rep)

        print ("Area under ROC curve: {:0.3f}".format(roc_auc))

        # Compute ROC curve and area under the curve
        probs = model.predict_proba(self.test_vecs)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.Y_TEST, probs)

        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=11)
        plt.ylabel('True Positive Rate', fontsize=11)
        plt.title('Receiver Operating Characteristic (ROC) curve', fontsize=11)
        plt.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
        plt.grid(True, linestyle = 'dotted')
        plt.savefig("doc2vec_roc.png")
        print ("ROC curve created...")
        print ("Loading ROC curve graph...")
        plt.show()
        
    def predict_input(self, test_vecs):

        model = self.use_model()

        predictions = model.predict_proba(test_vecs)

        return predictions