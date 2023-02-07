import numpy as np
import itertools


#data = datasets.load_breast_cancer()
#X, y = data.data, data.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


class HyperparametersTuning:

    def __init__(self, n_iter, model, dicto):
        self.dict_hyperparams = dicto
        self.list_hyperparams = []  # list of lists of hyperparams
        self.n_iter = n_iter  # for bootstrap
        self.model = model  # model used
        self.accuracy = 0

        self.combinaisons()

    def combinaisons(self):
        # *args is n lists of hyperparameters
        # allow us to find every single combinaison of hyperparameters that will try to find the best models
        lists = list(self.dict_hyperparams.values())
        self.list_hyperparams = list(itertools.product(*lists))

    def block_bootstrap(self, data: np.array, y: np.array, size_block=30, validation_set = 0.3, type_tree='RF', type_boot = "CBB", met='mse'):
        """
        bootstrap following the following methodology:
            1) randomly create a dataset by picking with replacement in block from the data
            2) do it n times
            3) seelct the one which have the best mean
        :param data: X
        :param y: y
        :param size_block:
        :return: better parameters
        """
        best = {}
        avg = {}
        n = data.shape[0]
        idx_valid = int(validation_set * n)
        df = np.hstack((data, y.reshape(-1, 1)))
        for iteration in range(self.n_iter):
            # randomly picking a block of data and use it as validation set
            if type_boot == "stationary":
                bootstrap_data = self.generate_stationary_bootstrap(df, size_block, df.shape[0])
            elif type_boot == 'CBB':
                idx_array = np.array_split(df, int(len(data)/size_block))
                bootstrap_data = self.create_boostrap_data(idx_array)

            # train and validation sets
            X_boot, y_boot = bootstrap_data[:idx_valid, :-1], bootstrap_data[:idx_valid, -1]
            X_valid, y_valid = bootstrap_data[idx_valid:, :-1], bootstrap_data[idx_valid:, -1]
            for parameters in range(len(self.list_hyperparams)):

                max_min = self.compute_hyperparams("min_samples_split", 2, parameters)
                n_fts = self.compute_hyperparams("max_features", None, parameters)
                max_depth = self.compute_hyperparams("max_depth", 5, parameters)

                if type_tree in ['RF', 'GB']:
                    n_est = self.compute_hyperparams("n_estimators", 20, parameters)
                    if type_tree == 'RF':
                        mod = self.model(min_samples_split=max_min, max_depth=max_depth, max_features=n_fts,
                                         n_estimators=n_est)
                    elif type_tree == 'GB':
                        lr = self.compute_hyperparams("learning_rate", 0.01, parameters)
                        mod = self.model(min_samples_split=max_min, max_depth=max_depth, max_features=n_fts,
                                         n_estimators=n_est, learning_rate=lr)
                elif type_tree in ['DT', 'RT']:
                    if type_tree == 'DT':
                        mod = self.model(min_samples_split=max_min, max_depth=max_depth, max_features=n_fts)
                    elif type_tree == 'RT':
                        mod = self.model(min_samples_split=max_min, max_depth=max_depth, max_features=n_fts, tree_alone=True)
                else:
                    raise ValueError('Model not implemented')

                mod.fit(X_boot, y_boot)
                y_pred = mod.predict(X_valid)
                # accuracy of model with params
                if met == 'accuracy':
                    score = sum(y_pred == y_valid) / len(y)
                elif met == 'mse':
                    score = -np.mean(sum((y_pred - y_valid)**2))
                else:
                    raise ValueError('met not define')
                # add accuracy to a list, if key doesnt exist, create it
                best.setdefault(parameters, []).append(score)

        for key in best:
            # mean of the accuracy of all models tested
            avg[key] = np.mean(best[key])

        # return optimized hyperparameters (key which has the highest item)
        return self.list_hyperparams[max(avg, key=avg.get)]

    @staticmethod
    def create_boostrap_data(liste, bootstarp_data=np.array([])):
        n = len(liste)
        for _ in range(0, n):
            idx_new_col = np.random.choice(range(1, n), size=1)[0]
            if len(bootstarp_data) == 0:
                bootstarp_data = liste[idx_new_col]
            else:
                bootstarp_data = np.vstack((bootstarp_data, liste[idx_new_col]))
        return bootstarp_data

    @staticmethod
    def generate_stationary_bootstrap(data, m, length):
        """
        data: the X and y stacked
        m: indicats the average length of blocks in the final sample
        sampleLength: 1 x 1 integer setting the lenght of the output sample.
        """
        accept = 1 / m
        lenData = data.shape[0]
        sampleLength = length  # Set sampleLength to desired value
        sampleIndex = np.random.choice(lenData, 1)
        sample = np.zeros((sampleLength, data.shape[1]))

        for iSample in range(sampleLength):
            if np.random.uniform() >= accept:
                sampleIndex += 1
                if sampleIndex >= lenData:
                    sampleIndex = 0
            else:
                sampleIndex = np.random.choice(lenData, 1)

            sample[iSample] = data[sampleIndex]
        return sample

    def compute_hyperparams(self, hyper_to_opt: str, else_cond, param):
        """
        grab the value (of the hyper parameters we want) in the list_hyperparams
        :param hyper_to_opt: parameters we wanna optimize
        :param else_cond: in case we do not want to optimize that hyper params, it would take that value
        :param param: indices in list_hyper
        :return: value hyper for this loop
        """
        if hyper_to_opt in self.dict_hyperparams:
            idx = list(self.dict_hyperparams.keys()).index(hyper_to_opt)
            hyper_ = self.list_hyperparams[param][idx]
        else:
            hyper_ = else_cond
        return hyper_
