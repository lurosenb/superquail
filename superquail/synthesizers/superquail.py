import numpy as np
import pandas as pd
import time
from numpy.random import default_rng
from joblib import Parallel, delayed

from diffprivlib.tools import mean as DPMean
from diffprivlib.tools import var as DPVar 

from scipy.stats import truncnorm

def get_truncated_normal(mean, sd, low, upp):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class dumb_predictor():
    """
    Dummy classifier to be used if any of conf.KNOWN_MODELS break.
    Returns single class as prediction.
    """
    def __init__(self, label):
        self.label = label
        
    def predict(self, instances):
        return np.full(len(instances), self.label)

class SuperQUAILSynthesizer():
    """
    Quailified Architecture to Improve Labeling.

    Divide epsilon in a known classification task
    between a differentially private synthesizer and
    classifier. Train DP classifier on real, fit DP synthesizer
    to features (excluding the target label)
    and use synthetic data from the DP synthesizer with
    the DP classifier to create artificial labels. Produces
    complete synthetic data
    """
    
    def __init__(self, epsilon, dp_classifier, dp_linear_classifier, class_args, gaussian=False, test_size=0.2, seed=42, eps_split=0.9):
        self.epsilon = epsilon
        self.eps_split = eps_split
        self.dp_classifier = dp_classifier
        self.dp_linear_classifier = dp_linear_classifier
        self.class_args = class_args
        self.test_size = test_size
        self.seed = seed
        
        # Model
        self.private_models = None
        self.private_synth = None
        
        # Pandas check
        self.pandas = False
        self.pd_cols = None
        self.pd_index = None

        self.continuous_ranges = None
        self.categorical_ranges = None
        self.ordinal_ranges = None
        self.mean_per_column = None
        self.var_per_column = None
        self.gaussian = gaussian

        # Randomness
        self.rng = np.random.default_rng(self.seed)
        
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        
        if isinstance(data, pd.DataFrame):
            self.pandas = True
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='ignore')
            self.data = data
            self.pd_cols = data.columns
            self.pd_index = data.index
        else:
            raise('Only pandas dataframes for data as of now.')

        self.private_models = {}
        
        self.mean_per_column, self.var_per_column, spent_eps = self._report_noisy_mean_variance(data)

        self.epsilon = self.epsilon #-spent_eps
        eps_per_column = self.epsilon / float(len(data.columns) + 1)
        
        self.continuous_ranges = {}
        self.categorical_ranges = {}
        self.ordinal_ranges = {}

        for c in categorical_columns:
            # TODO: Delve into this
            # Pretty sure its safe to just grab all the possible categories
            self.categorical_ranges[c] = np.unique(data[c])
        
        print(self.categorical_ranges)
        
        for c in ordinal_columns:
            # We do same thing we do for ordinal
            self.ordinal_ranges[c] = (int(self._report_noisy_max_min(data[c], eps_per_column, 'min')),
                                        int(self._report_noisy_max_min(data[c], eps_per_column, 'max')))

        cat_ord_columns = list(categorical_columns) + list(ordinal_columns)
        
        print(self.ordinal_ranges)
        print(cat_ord_columns)
        
        for c in data.columns:
            ## Take care of continuous column distribution ranges here
            # print('Training model for ' +  c)
            if c not in cat_ord_columns:
                self.continuous_ranges[c] = (self._report_noisy_max_min(data[c], eps_per_column, 'min'),
                                        self._report_noisy_max_min(data[c], eps_per_column, 'max'))
        print(self.continuous_ranges)
        for c in data.columns:
            ## Train Model
            features = data.loc[:, data.columns != c]
            target = data.loc[:, data.columns == c]
            x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size, random_state=self.seed)
            try:
                if c in cat_ord_columns:
                    private_model = self.dp_classifier(epsilon=eps_per_column, **self.class_args)
                    private_model.fit(x_train, y_train.values.ravel())
                    
                    predictions = private_model.predict(x_test)
                    class_report = classification_report(np.ravel(y_test), predictions, labels=np.unique(predictions))
                    target_accuracy = accuracy_score(np.ravel(y_test), predictions)
                    print(target.columns)
                    print(class_report)
                else:
                    private_model = self.dp_linear_classifier(epsilon=eps_per_column, **self.class_args)
                    private_model.fit(x_train, y_train.values.ravel())
                    
#                     predictions = private_model.predict(x_test)
#                     print(y_test)
#                     print(np.ravel(y_test))
#                     score = r2_score(np.ravel(y_test), predictions, multioutput='variance_weighted')
#                     print(score)
                
            except BaseException as err:
                print(err)
                print('Unsuccessful when training model for ' +  c + ', using dumb_predictor.')
                y, counts = np.unique(y_train.values.ravel(), return_counts=True)
                label = y[np.argmax(counts)] # this should be DP mode!!
                private_model = dumb_predictor(label)
            
            if c not in self.private_models:
                self.private_models[c] = private_model
            else:
                raise ValueError("Duplicate column model built.")

    def sample(self, samples):
        
        def _a_sample(arg):
            _, sample_shape = arg
            sample = np.empty(sample_shape)
            shuffled_column_indexes = np.arange(len(self.data.columns))
            self.rng.shuffle(shuffled_column_indexes)
            
            shuffled_columns = self.data.columns[shuffled_column_indexes]
            reordered = self._reorder(shuffled_column_indexes)
            return self._iterative_sample_predict(sample, sample_shape, shuffled_columns, reordered)
        
        print(self.data.columns)
        start = time.time()
        job_num = 10
        sample_shape = self.data.iloc[0].shape

        runs = [(i, sample_shape) for i in range(samples)]
        results = Parallel(n_jobs=job_num, verbose=1, backend="loky")(
            map(delayed(_a_sample), runs))
        end = time.time() - start
        # print('Sampling took ' + str(end))

        return pd.DataFrame(np.array(results), columns = self.data.columns)
        

    def _report_noisy_mean_variance(self, data):
        mean_per_column = {}
        var_per_column = {}
        spent_eps = 0

        for c in data.columns:
            c_m = DPMean(data[c], epsilon = 0.01)
            mean_per_column[c] = c_m
            c_v = DPVar(data[c], epsilon = 0.03)
            var_per_column[c] = c_v
            spent_eps+=0.04
        
        print(mean_per_column, var_per_column, spent_eps)
        return mean_per_column, var_per_column, spent_eps

    def _report_noisy_max_min(self, s, epsilon, min_or_max='max'):
        best = 0
        r = 0
        for i,d in enumerate(s):
            d = d + self._laplace(epsilon)
            if min_or_max == 'min':
                if d < best:
                    r = d
            elif min_or_max == 'max':
                if d > best:
                    r = d
            else:
                raise ValueError('Must specify either min or max.')
        return r
    
    def _laplace(self, sigma):
        return sigma * np.log(self.rng.random()) * self.rng.choice([-1, 1])
    
    def _reorder(self, splits):
        flat = splits.ravel()
        reordered = np.zeros(len(flat))
        for i, ind in enumerate(flat):
            reordered[ind] = i
        return reordered.astype(int)
    
    def _ordinal_distribution(self, l, u, n, mean=None, var=None):
        if self.gaussian:
            trunc_norm = get_truncated_normal(mean, np.sqrt(var), l, u)
            return int(trunc_norm.rvs())
        else:
            return self.rng.integers(low=l, high=u, size=n)

    def _categorical_distribution(self, categories, n, mean=None, var=None):
        if self.gaussian:
            # no sensible notion of normal distribution for categorical
            return self.rng.choice(categories, size=n)
        else:
            return self.rng.choice(categories, size=n)

    def _continuous_distribution(self, l, u, n, mean=None, var=None):
        if self.gaussian:
            draw = self.rngnormal(loc=mean, scale=np.sqrt(var), size=1)
            return draw
        else:
            return self.rng.uniform(low=l, high=u, size=n)

    def _iterative_sample_predict(self, sample, shape, shuffled_columns, reordered):
        for i, c in enumerate(shuffled_columns):
            pred_sample = np.empty(shape)
            pred_sample[:] = None
            for j, col in enumerate(shuffled_columns):
                if c != col:
                    if sample[j]:
                        pred_sample[j] = sample[j]
                    else:
                        if col in self.continuous_ranges:
                            pred_sample[j] = self._continuous_distribution(l = self.continuous_ranges[col][0],
                                                                           u = self.continuous_ranges[col][1],
                                                                           n = 1,
                                                                           mean = (self.mean_per_column[c] if self.gaussian else None),
                                                                           var = (self.var_per_column[c] if self.gaussian else None))
                        elif col in self.ordinal_ranges:
                            pred_sample[j] = self._ordinal_distribution(l = self.ordinal_ranges[col][0],
                                                                        h = self.ordinal_ranges[col][1] + 1, 
                                                                        n=1,
                                                                        mean = (self.mean_per_column[c] if self.gaussian else None),
                                                                        var = (self.var_per_column[c] if self.gaussian else None))
                        elif col in self.categorical_ranges:
                            pred_sample[j] = self._categorical_distribution(categories = self.categorical_ranges[col], 
                                                                            n=1)

            pred_sample = pred_sample[pred_sample != np.array(None)]
            pred_sample = pred_sample[~np.isnan(pred_sample)]
            c_pred = self.private_models[c].predict(pred_sample.reshape(1, -1))
            sample[i] = c_pred

        return sample[reordered]