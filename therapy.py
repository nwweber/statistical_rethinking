"""
Purpose:

* Provide useful wrapper to PyMC3 Model and Trace classes, emulating sklearn-ish interface
* Create analogous functions to the 'rethinking' R package
(* Restore sanity)

Author: Niklas Weber
"""
from contextlib import contextmanager
import pandas as pd
import pymc3 as pm
import numpy as np


class Model(object):
    """
    Encapsulate constituent parts of a PyMC3 model and provide useful functionality

    Holds:
    * model
    * trace
    * shared variables
    """

    def __init__(self, model, shared_variables, name):
        """
        Create new instance

        :param model: pymc3 Model instance
        :param shared_variables: dictionary: {'shared_variable_name': theanor tensor}
        :param name: string
        """

        self.model = model
        self.shared_variables = shared_variables
        self.name = name
        self.trace = None

    def _repr_latex_(self):
        """
        Automatic support for nice rendering
        :return: latex representation this model
        """
        # noinspection PyProtectedMember
        return self.model._repr_latex_()

    def fit(self, X):
        """
        Fit model to data set. Sets 'trace' property. Sets values of shared variables.
        Modifies model state. Side-effects only. (X, y) interface to be consistent with
        sklearn.

        :param X: DataFrame, having a column for each shared variable of this model except for outcome
        :return: None
        """

        self._permanently_update_shared_variables(X)
        self.trace = pm.sample(draws=1000, tune=1000, progressbar=False, model=self.model)

        return None

    def _permanently_update_shared_variables(self, X):
        """
        Set shared variables to values in X. Side-effects only.

        :param X: DataFrame with a column for each shared variable
        :return: None
        """

        for var_name, var in self.shared_variables.items():
            var.set_value(X[var_name].values)

        return None

    def predict(self, X, vars):
        """
        For each data point (row) in X: predict mean and hpd for each variable in vars.

        Does not affect internal model state (trace, shared variables).

        :param vars: iterable of Variable objects from self.model
        :param X: DataFrame, columns = predictors, rows = observations
        :return: DataFrame, columns = var_1_mean, var_1_hpd_lower, var_1_hpd_upper, ..., rows = observations
        """

        # TODO: wrap below into self-contained function like 'predict_ppc_samples()'?
        # maybe only if i actually repeat this bit of code somewhere
        with self._temporarily_update_shared_variables(X) as _:
            ppc_samples = pm.sample_ppc(
                trace=self.trace,
                model=self.model,
                vars=vars,
                progressbar=False
            )

        return self.summarize_ppc_samples(ppc_samples)

    def _get_shared_variable_values(self):
        """
        Record current state of shared variables

        :return: DataFrame, column = shared var
        """
        return pd.DataFrame({var_name: var.get_value() for var_name, var in self.shared_variables.items()})

    @contextmanager
    def _temporarily_update_shared_variables(self, X):
        """
        Change shared variables, then change them back. To be used as context manager
        :param X: DataFrame, column name = shared variable name
        :return: dummy object
        """
        # record original data and change
        original_shared_variable_values = self._get_shared_variable_values()

        try:
            # update
            self._permanently_update_shared_variables(X)
            # context created here
            yield None
        except Exception as e:
            raise e
        finally:
            # tear-down: reset values to original
            self._permanently_update_shared_variables(original_shared_variable_values)

    def summarize_ppc_samples(self, ppc_samples):
        """
        Calculate mean and hpd for each observation x variable.

        :param ppc_samples: pymc3 ppc samples ordered dict
        :return: DataFrame, 3 columns per variable (mean, hpd lower, hpd upper), one row per observation
        """
        summaries = [
            self._summarize_one_variable(ppc_samples, variable)
            for variable in list(ppc_samples.keys())
        ]

        return pd.concat(summaries, axis=1)

    def _summarize_one_variable(self, ppc_samples, variable):
        """
        Provide mean and hpd summaries of given variable.

        :param ppc_samples: pymc3 ppc samples
        :param variable: key of dict ppc_samples
        :return: DataFrame, (variable_mean, variable_hpd_lower, variable_hpd_upper), n rows = n columns in ppcs_samples[variable]
        which should correspond to number of input data points
        """
        # row = sample, column = original data point
        samples = ppc_samples[variable]

        d = dict()
        d['mean'] = np.mean(samples, axis=0)
        hpds = pm.hpd(samples, alpha=.3)
        d['hpd_lower'] = hpds[:, 0]
        d['hpd_upper'] = hpds[:, 1]

        return (
            pd
            .DataFrame(d)
            .rename(columns=lambda colname: '{}_{}'.format(variable, colname))
        )

