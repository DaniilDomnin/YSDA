import contextlib
import inspect
import json
import os
import pathlib
import sys
import typing as tp
import uuid
from copy import copy

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection._validation import _score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class MyBinaryTreeGradientBoostingClassifier:
    """
    *Binary* gradient boosting with trees using
    negative log-likelihood loss with constant learning rate.
    Trees are to predict logits.
    """
    big_number = 1 << 32
    eps = 1e-8

    def __init__(
            self,
            n_estimators: int,
            learning_rate: float,
            seed: int,
            **kwargs
    ):
        """
        :param n_estimators: estimators count
        :param learning_rate: hard learning rate
        :param seed: global seed
        :param kwargs: kwargs of base estimator which is sklearn TreeRegressor
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_logits = None
        self.rng = np.random.default_rng(seed)
        self.base_estimator = DecisionTreeRegressor
        self.base_estimator_kwargs = kwargs
        self.estimators = []
        self.n_class = None
        self.loss_history = []  # this is to track model learning process
        self.coeff = []

    def create_new_estimator(self, seed):
        return self.base_estimator(**self.base_estimator_kwargs, random_state=seed)

    @staticmethod
    def cross_entropy_loss(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute negative log-likelihood for logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions.
        This is used to track model learning process
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """
        res = 0
        for ind, logit in enumerate(logits):
            prob = 1 / (1 + np.exp(-logit))
            y = true_labels[ind]
            res += y * np.log(prob + MyBinaryTreeGradientBoostingClassifier.eps) + (1 - y) * np.log(
                1 - prob + MyBinaryTreeGradientBoostingClassifier.eps)

        res *= -1

        return res

    @staticmethod
    def cross_entropy_loss_gradient(
            true_labels: np.ndarray,
            logits: np.ndarray
    ):
        """
        compute gradient of log-likelihood w.r.t logits,
        use clipping for logarithms with self.eps
        or use numerically stable special functions
        :param true_labels: [n_samples]
        :param logits: [n_samples]
        :return:
        """
        ...
        prob = 1 / (1 + np.exp(-logits))
        return prob - true_labels

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray
    ):
        """
        sequentially fit estimators to reduce residual on each iteration
        :param X: [n_samples, n_features]
        :param y: [n_samples]
        :return: self
        """
        self.loss_history = []
        # only should be fitted on datasets with binary target
        assert (np.unique(y) == np.arange(2)).all()
        # init predictions with mean target (mind that these are logits!)
        one_count = np.count_nonzero(y == 1)
        prob = one_count / len(y)
        self.initial_logits = np.log(prob / (1 - prob))
        # create starting logits
        logits = np.full(len(y), self.initial_logits)
        # init loss history with starting negative log-likelihood
        self.loss_history.append(self.cross_entropy_loss(y, logits))
        # sequentially fit estimators with random seeds
        for seed in self.rng.choice(
                max(self.big_number, self.n_estimators),
                size=self.n_estimators,
                replace=False
        ):
            # add newly created estimator
            new_est = self.create_new_estimator(seed)
            self.estimators.append(new_est)
            # compute gradient
            gradient = self.cross_entropy_loss_gradient(y, logits)
            # fit estimator on gradient residual
            new_est.fit(X, gradient)
            pred = new_est.predict(X)
            nums = np.arange(0, 20, 0.2)
            nums = np.append(nums, [1])
            min_loss = None
            arg_min = None
            for num in nums:
                cur_loss = self.cross_entropy_loss(y, logits - num * pred)
                if min_loss is None:
                    min_loss = cur_loss
                    arg_min = num
                else:
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        arg_min = num

            self.coeff.append(arg_min)
            # adjust logits with learning rate
            logits += arg_min * self.learning_rate * pred
            # append new loss to history
            self.loss_history.append(self.cross_entropy_loss(y, logits))
        return self

    def predict_proba(
            self,
            X: np.ndarray
    ):
        """
        :param X: [n_samples]
        :return:
        """
        # init logits using precalculated values
        logits = np.full(X.shape[0], self.initial_logits)
        # sequentially adjust logits with learning rate
        for ind, estimator in enumerate(self.estimators):
            logits += self.coeff[ind] * estimator.predict(X)
        # don't forget to convert logits to probabilities
        proba = []
        for logit in logits:
            proba.append(1 / (1 + np.exp(logit)))

        proba = np.array(proba)
        return proba

    def predict(
            self,
            X: np.ndarray
    ):
        """
        calculate predictions using predict_proba
        :param X: [n_samples]
        :return:
        """
        pred = []
        res = self.predict_proba(X)
        for el in res:
            if el > 0.5:
                pred.append(1)
            else:
                pred.append(0)

        pred = np.array(pred)
        return pred
