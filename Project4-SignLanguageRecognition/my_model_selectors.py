import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_n = self.min_n_components
        best_bic = np.float('inf')
        for n in range(self.min_n_components, self.max_n_components+1):
            hmm_model = self.base_model(n)
            try:
                logL = hmm_model.score(self.X, self.lengths)
                num_state_params = hmm_model.transmat_.shape[0] * (hmm_model.transmat_.shape[1] - 1)
                num_output_params = n * self.X.shape[1] * 2 # As it is a Gaussian HMM
                total_num_params = num_state_params + num_output_params + (n - 1)
                bic = - (2 * logL) + (total_num_params * np.log(self.X.shape[0]))
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except:
                pass

        return self.base_model(best_n)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_n = self.min_n_components
        best_dic = np.float('-inf')
        for n in range(self.min_n_components, self.max_n_components+1):
            hmm_model = self.base_model(n)
            try:
                this_word_logL = hmm_model.score(self.X, self.lengths)
                avg_other_words_logL = 0
                num_other_words = 0
                for word in self.words:
                    if word == self.this_word: continue
                    other_word_X, other_word_lengths = self.hwords[word]
                    avg_other_words_logL += hmm_model.score(other_word_X, other_word_lengths)
                    num_other_words += 1
                avg_other_words_logL /= num_other_words
                dic = this_word_logL - avg_other_words_logL
                if dic >= best_dic:
                    best_dic = dic
                    best_n = n
            except:
                pass

        return self.base_model(best_n)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        if len(self.sequences) == 1:
            return self.base_model(self.n_constant)

        best_n = self.min_n_components
        best_logL = np.float('-inf')
        n_splits = 3 if len(self.sequences) >= 3 else len(self.sequences)
        split_method = KFold(n_splits)
        for n in range(self.min_n_components, self.max_n_components+1):
            avg_test_logL = 0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                cv_train_X, cv_train_lengths = combine_sequences(cv_train_idx, self.sequences)
                cv_test_X, cv_test_lengths = combine_sequences(cv_test_idx, self.sequences)
                try:
                    hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(cv_train_X, cv_train_lengths)
                    if self.verbose:
                        print("model created for {} with {} states".format(self.this_word, num_states))

                    test_logL = hmm_model.score(cv_test_X, cv_test_lengths)
                    avg_test_logL += test_logL
                except:
                    if self.verbose:
                        print("failure on {} with {} states".format(self.this_word, num_states))
                    # return None

            # Compute average log-likelihood on the cross-validation test splits
            avg_test_logL /= n_splits
            # Update best model based on average log-likelihood
            if avg_test_logL >= best_logL:
                best_logL = avg_test_logL
                best_n = n

        best_model = GaussianHMM(n_components=best_n, covariance_type="diag", n_iter=1000,
                                 random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

        return best_model
