import warnings
from asl_data import SinglesData
import numpy as np

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses'

    for test_item in range(test_set.num_items):
        test_word_X, test_word_lengths = test_set.get_item_Xlengths(test_item)
        scores_dict = {}
        best_word_guess = None
        best_logL = np.float('-inf')
        for word in models:
            hmm_model = models[word]
            try:
                logL = hmm_model.score(test_word_X, test_word_lengths)
                scores_dict[word] = logL
                if logL >= best_logL:
                    best_logL = logL
                    best_word_guess = word
            except:
                scores_dict[word] = np.float('-inf')
        probabilities.append(scores_dict)
        guesses.append(best_word_guess)

    return probabilities, guesses
