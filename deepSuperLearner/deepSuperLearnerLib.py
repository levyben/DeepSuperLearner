import numpy as np
from scipy.optimize import fmin_l_bfgs_b, nnls, fmin_slsqp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn import clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt



class DeepSuperLearner(object):
    '''
    classdocs
    '''


    def __init__(self, blearners, K=5, sample_weight=None):
        self.BL = blearners
        self.Kfolds = K
        self.coef_optimization_method = 'SLSQP'
        self.n_baselearners = len(blearners)
        self.trim_eps = 1e-5
        self.trim_func = lambda x: np.clip(x, self.trim_eps, 1 - self.trim_eps)
        self.sample_weight = sample_weight
        self.weights_per_iteration = []
        self.fitted_learners_per_iteration = []
        self.__classes_n = 0
        self.label_onehotencoder = LabelBinarizer()
    def _get_weighted_prediction(self, m_set_predictions, weights):
        """
        Calculate weighted combination of predictions probabilities

        Parameters
        ----------

        m_set_predictions: numpy.array of shape [n, m, j]
                    where each column is a vector of j-class probablities 
                    from each base learner (each channel represent probability of
                    different class).

        weights: numpy.array of length m (base learners count),
        to be used to combine columns of m_set_predictions.

        Returns
        _______

        avgprobs: numpy.array of shape [n,j].
        
        
        """
#         invp = _inv_segmoid(self.trim_func(m_set_predictions))
#         weights_probs = _segmoid(np.stack([np.dot(invp[:,:,i],weights) 
#                   for i in range(invp.shape[-1])]).T)
        trimp = self.trim_func(m_set_predictions)
        weights_probs = np.stack([np.dot(trimp[:, :, i], weights) 
                  for i in range(trimp.shape[-1])]).T
        return weights_probs

    
    def _get_logloss(self, y, y_pred):
        """
        Calculate the normalized logloss given ground-truth y and y-predictions

        Parameters
        ----------
        y: numpy array of shape [n,j] (ground-truth)

        y_pred: numpy array of shape [n,j] (predictions)

        Returns
        -------
        Logloss: estimated logloss of ground-truth and predictions.

        """
        return log_loss(y, y_pred, eps=self.trim_eps,
                        sample_weight=self.sample_weight)
    
    def _get_weights(self, y, m_set_predictions_fold):
        """
        Find weights that minimize the estimated logloss.

        Parameters
        ----------
        y: numpy.array of shape [n,j]

        m_set_predictions_fold: numpy.array of shape [n, m, j] of fold-k

        Returns
        _______
        weights: numpy.array of normalized non-negative weights to combine
              base learners
              
        
        """
        def objective_f(w):  # Logloss(y,w*y_pred)
            return self._get_logloss(y, self._get_weighted_prediction(m_set_predictions_fold, w))
        def normalized_constraint(w):  # Sum(w)-1 == 0
            return np.array([ np.sum(w) - 1 ])
        w0 = np.array([1. / self.n_baselearners] * self.n_baselearners)
        wbounds = [(0, 1)] * self.n_baselearners
        out, _, _, imode, _ = fmin_slsqp(objective_f, \
            w0, f_eqcons=normalized_constraint, bounds=wbounds, \
             disp=0, full_output=1)
        if imode is not 0:
            raise Exception("Optimization failed to find weights")

        out = np.array(out)
        out[out < np.sqrt(np.finfo(np.double).eps)] = 0
        weights = out / np.sum(out)
        return weights
    def _get_prediction(self, bl, X):
        """
        Calculates baselearner(X).
        
        Parameters
        ----------
        bl : baselearner instance
        X : numpy array of shape [n] 
        
        Returns
        -------
        pred : returns prediction of shape [n,j] where j is the number of classes.
        """
        
        if hasattr(bl, "predict_proba"):
            pred = bl.predict_proba(X)
        else:
            raise Exception("No predict_proba function found for {}"
                            .format(bl.__class__.__name__))
        return pred
    
    def fit(self, X, y, max_iterations=20):
        """
        Fit DeepSuperLearner.

        Parameters
        ----------
        X : numpy array of shape [n,l features]
        y : numpy array of shape [n,j] 
        Returns
        -------
        self : returns an instance of self.
        """
        n, j = len(y) , len(np.unique(y))
        self.__classes_n = j
        latest_loss = np.finfo(np.double).max
        weights_per_iteration = []
        fitted_learners_per_iteration = []
        for iteration in range(max_iterations):
            fitted_learners_per_fold = np.empty(shape=(self.Kfolds, self.n_baselearners),
                                                dtype=np.object)
            y_pred_fold = np.empty(shape=(n, self.n_baselearners, j))
            folds = StratifiedKFold(n_splits=self.Kfolds, shuffle=False)
            for fold_i, fold_indexes in enumerate(folds.split(X, y)):
                train_index, test_index = fold_indexes
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                for i, baselrn in enumerate(self.BL.items()):
                    name, bl = baselrn
                    baselearner = clone(bl)
                    baselearner.fit(X_train, y_train)
                    fitted_learners_per_fold[fold_i, i] = baselearner
                    y_pred_fold[test_index, i, :] = self._get_prediction(baselearner, X_test)
            
            fitted_learners_per_iteration.append(fitted_learners_per_fold)
            tmp_weights = self._get_weights(y, y_pred_fold)
            avg_probs = self._get_weighted_prediction(y_pred_fold, tmp_weights)
            loss = self._get_logloss(y, avg_probs)
            weights_per_iteration.append(tmp_weights)
            print("Iteration: {} Loss: {}".format(iteration, loss))
            print("Weights: ", tmp_weights)
            if loss < latest_loss:
                latest_loss = loss
                X = np.hstack((X, avg_probs))
            else:
                weights_per_iteration = weights_per_iteration[:-1]
                fitted_learners_per_iteration = fitted_learners_per_iteration[:-1]
                break
        
        self.weights_per_iteration = weights_per_iteration
        self.fitted_learners_per_iteration = fitted_learners_per_iteration

        return self

    def predict(self, X, return_base_learners_probs=False):
        """
        Calculates DeepSuperLearner(X) of fitted learners.

        Parameters
        ----------
        X : numpy.array of shape [n, l]
        return_base_learners_probs : return also fitted base learners probs on X.

        Returns
        -------
        prediction probabilities numpy.array of shape [n,j] and optionally 
            base learners probs numpy.array of shape [n,m,j]
        """
        iterations = len(self.weights_per_iteration)
        if iterations == 0:
            raise Exception("DeepSuperLearner wasn't fitted!")
        n = len(X)
        j = self.__classes_n
        base_learners_probs = None
        for iteration in range(iterations):
            y_pred_fold = np.empty(shape=(n, self.n_baselearners, j))
            fitted_base_learners_per_fold = self.fitted_learners_per_iteration[iteration]
            for bl_i in range(len(self.BL)):
                base_learner_probs_per_fold = np.empty(shape=(self.Kfolds, n, j)) 
                for fold_i in range(self.Kfolds):
                    base_learner = fitted_base_learners_per_fold[fold_i, bl_i]
                    base_learner_probs_per_fold[fold_i, :, :] = self._get_prediction(base_learner, X)
                base_learner_avg_probs = np.mean(base_learner_probs_per_fold, axis=0)
                y_pred_fold[:, bl_i, :] = base_learner_avg_probs
            
            if base_learners_probs is None:  # 1st iteration are normal base_learners classic estimates
                base_learners_probs = y_pred_fold
            optimized_weights = self.weights_per_iteration[iteration]
            avg_probs = self._get_weighted_prediction(y_pred_fold, optimized_weights)
            X = np.hstack((X, avg_probs))
        
        if return_base_learners_probs:
            return avg_probs, base_learners_probs
        
        return avg_probs


    def get_precision_recall(self, X_test, y_test, show_graphs=False):
        dsl_probs, base_learners_probs = \
        self.predict(X_test, return_base_learners_probs=True)
        _, labels_count = dsl_probs.shape
        dsl_predictions = np.argmax(dsl_probs, axis=1)
        base_learners_predictions = np.argmax(base_learners_probs, axis=2)
        
        dsl_precision_scores = precision_score(y_test, dsl_predictions, average=None)
        dsl_recall_scores = recall_score(y_test, dsl_predictions, average=None)
        bl_precision_scores = []
        bl_recall_scores = []
        for bn_i in range(self.n_baselearners):
            bl_precision_scores.append(precision_score(y_test, \
                    base_learners_predictions[:, bn_i], average=None))
            bl_recall_scores.append(recall_score(y_test, \
                    base_learners_predictions[:, bn_i], average=None))
        if show_graphs:
            label_indice = np.arange(labels_count)
            bl_names = list(self.BL.keys())
            fig = plt.figure(0, figsize=(20, 20))
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            ax1.set_ylabel("Recall")
            # ax1.set_xlabel("Label Index")
            # ax1.set_ylim([-0.05, 1.05])
            # ax1.legend(loc="upper center")
            ax1.set_title('Recall rates of the different learners')
            ax2.set_ylabel("Precision")
            ax2.set_xlabel("Label Index")
            # ax2.set_ylim([-0.05, 1.05])
            ax2.set_title('Precision rates of the different learners')
            ax1.plot(label_indice, dsl_recall_scores, "s--",
                     label="{}".format(self.__class__.__name__), linewidth=2.0)
            ax2.plot(label_indice, dsl_precision_scores, "s--",
                     label="{}".format(self.__class__.__name__), linewidth=2.0)
            for bn_i in range(self.n_baselearners):
                ax1.plot(label_indice, bl_recall_scores[bn_i], "s--",
                         label="{}".format(bl_names[bn_i]), linewidth=1.0)
                ax2.plot(label_indice, bl_precision_scores[bn_i], "s--",
                         label="{}".format(bl_names[bn_i]), linewidth=1.0)
            
            ax1.legend(loc="lower right")
            ax2.legend(loc="lower right")

            plt.show()
        
        return dsl_recall_scores, dsl_precision_scores, \
                bl_recall_scores, bl_precision_scores
        
        
        














def _inv_segmoid(p):
    """
    Calculate the inverse of the segmoid a.k.a the logit function.
    
    Paramters
    ---------
    p: numpy.array of numbers between 0 and 1

    Returns
    -------
    Inverse_segmoid(p)
    """
    return np.log(p / (1 - p))

def _segmoid(x):
    """
    Calculates the segmoid

    Paramters
    ---------
    x: numpy.array of real numbers

    Returns
    -------
    Segmoid(x)

    """
    
    return 1 / (1 + np.exp(-x))



    
