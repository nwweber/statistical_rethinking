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
import theano


def create_theano_shared_from_df(df):
    """
    For each column in pd df: create theano shared variable, ready to be plugged into pymc3 model
    :param df:
    :return: dict, {df_column_name: theano shared variable based on df[df_column_name]}
    """
    shared = {}
    for key in df.columns:
        shared[key] = theano.shared(df[key].values, name=key)

    return shared


def interesting_summary(model):
    """
    For Model: generate summary of trace, but filter out auto-generated and deterministic variables, such as 'mu__n'
    :param model:
    :return: pd df summary
    """
    summary = pm.summary(model.trace)
    return summary.loc[['__' not in ix for ix in summary.index]]


def sample_mu_ensemble_ppc(models, weights, n_total_samples):
    """
    For given models and weights: generate posterior samples of 'mu' variable proportional to given weights.
    Will skip models with 0 weight.
    :param models: iterable of Model objects
    :param weights: iterable of floating point weights
    :param n_total_samples: int
    :return: pandas df with posterior mu samples and 'model' column indiciating which model generated this sample
    """
    post_sample_dfs = list()

    for model, weight in zip(models, weights):
        if weight == 0:
            continue

        samples = pm.sample_ppc(
            model.trace,
            samples=int(weight*n_total_samples),
            model=model.model,
            vars=[model.model.mu]
        )['mu']

        df = pd.DataFrame(
            data=samples,
            columns=['mu__{}'.format(i) for i in range(samples.shape[1])]
        ).assign(model=model.name)

        post_sample_dfs.append(df)

    return pd.concat(post_sample_dfs, ignore_index=True)


def precis(model):
    """
    Take a fitted model, give summaries about posterior estimates

    In:
        fitted_model: instance of Model

    Out:
        pandas df with posterior summaries
    """

    summary = pm.summary(model.trace, alpha=.11)

    # filter out deterministic variables
    # recognized by the '__'  in the name
    return summary.loc[[ix for ix in summary.index if '__' not in ix]]


def coef(m):
    """
    Return posterior mean of parameters necessary to compute mu, i.e. the predicted mean

    :param m: fitted Model instance
    :return: pandas series, index = parameter names, values = parameter values
    """
    tdf = pm.trace_to_dataframe(m.trace)
    # filter for 'a', i.e. intercept, or for 'b.*', i.e. coefficients
    # together, specify formula for mean
    return tdf.filter(regex=r'^a$|^b.*', axis=1).mean()


def compare(models):
    """
    Compare models on WAIC (and some other measures)

    In:
        fitted_models: iterable of fitted Model instances

    Out:
        DataFrame, indexed by model names, columns having comparison values
    """

    # variable needs to be Series instead of just list b/c 'pm.compare' returns dataframe which is sorted
    # by information criterion value. need to match model names to entries of that dataframe by index,
    # which indicates initial position of the model when given to this function
    # note: silly design by pymc3
    model_names = pd.Series([fm.name for fm in models])

    model_dict = {fm.model: fm.trace for fm in models}

    return (
        pm
        .compare(
            model_dict = model_dict,
            method = 'BB-pseudo-BMA'
        )
        .assign(model =  model_names)
        .set_index('model')
        .sort_values('WAIC')
    )


def coeftab(models):
    """
    For each fitted model: provide MAP estimate of parameters. Display all of them next to each other
    in tabular format
    :param models: iterable of fitted Model objects
    :return: pandas dataframe with summaries
    """
    summaries = []

    for m in models:
        s = pm.summary(trace=m.trace)['mean']
        # filter out all deterministic values, like 'mu__1', 'mu__2', ...
        s = s.filter(items=[ix for ix in s.index if '__' not in ix])
        summaries.append(s)

    return pd.DataFrame(data=summaries, index=[m.name for m in models]).T


# TODO: check out new Data holder and class in pymc3 3.7, incorporate that
# start here for examples: https://github.com/pymc-devs/pymc3/blob/v3.7/pymc3/model.py#L1064
class Model(object):
    """
    Encapsulate constituent parts of a PyMC3 model and provide useful functionality

    Holds:
    * model
    * trace
    * shared variables
    * a name
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

    def fit(self, x=None, **kwargs):
        """
        Fit model to data set. Sets 'trace' property. Sets values of shared variables.
        Modifies model state. Side-effects only.

        If x is None: don't modify shared variables, fit model to whatever they are currently set to

        :param x: DataFrame, having a column for each shared variable of this model, or None
        :param kwargs: all other keyword args are passed on to the pymc 'sample' function
        :return: None
        """

        if x is not None:
            self._permanently_update_shared_variables(x)
        self.trace = pm.sample(model=self.model, **kwargs)

        return None

    def _permanently_update_shared_variables(self, x):
        """
        Set shared variables to values in x. Side-effects only. Ignores columns that don't match shared variables.
        Does nothing for shared variables that don't have a column.

        :param x: DataFrame with none or more column names matching shared variable names
        :return: None
        """

        # nothing to update
        if x is None:
            return

        for var_name, var in self.shared_variables.items():
            try:
                var.set_value(x[var_name].values)
            # key error: trying to access field that does not exist. just want to skip this
            except KeyError:
                continue

        return None

    def sample_posterior_predictive(self, **kwargs):
        """
        Wrapper around pm.sample_posterior_predictive. All arguments will be passed on to this function
        :param kwargs:
        :return: pm.sample_posterior_predictive output
        """
        return pm.sample_posterior_predictive(
            trace=self.trace,
            model=self.model,
            **kwargs
        )

    def predict(self, x=None, sample_ppc_kwargs=None):
        """
        Does not affect internal model state (trace, shared variables). Shared variables will be
        temporarily updated with all information available in x before sampling ppc.

        For each observed variable in model, return: $varname_samples, $varname_mean, $varname_hpd_lower, $varname_hpd_upper

        If fewer or other variables are desired in output set the 'var_names' key in sample_ppc_kwargs.
        Ex:
        sample_ppc_kwargs = {'vars': ['mu']}

        Columns in x that are not shared variables are ignored. Shared variables not in x are not updated, original
        values are used.

        If x is none ppc samples are drawn for current values of shared variables

        :param sample_ppc_kwargs: dict of args to pass to sample_posterior_predictive
        :param x: DataFrame, columns = predictors, rows = observations.
        :return: DataFrame
        """

        # needs to be dict for call below
        sample_ppc_kwargs = sample_ppc_kwargs if sample_ppc_kwargs is not None else dict()
        with self._temporarily_update_shared_variables(x) as _:
            ppc_samples = self.sample_posterior_predictive(**sample_ppc_kwargs)

        return self.summarize_ppc_samples(ppc_samples)

    def _get_shared_variable_values(self):
        """
        Record current state of shared variables

        :return: DataFrame, column = shared var
        """
        return pd.DataFrame({var_name: var.get_value() for var_name, var in self.shared_variables.items()})

    @contextmanager
    def _temporarily_update_shared_variables(self, x):
        """
        Change shared variables, then change them back. To be used as context manager
        :param x: DataFrame, column name = shared variable name
        :return: dummy object
        """
        # record original data and change
        original_shared_variable_values = self._get_shared_variable_values()

        try:
            # update
            self._permanently_update_shared_variables(x)
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

    @staticmethod
    def _summarize_one_variable(ppc_samples, variable):
        """
        Provide mean and hpd summaries of given variable.

        :param ppc_samples: pymc3 ppc samples
        :param variable: key of dict ppc_samples
        :return: DataFrame, (variable_mean, variable_hpd_lower, variable_hpd_upper), n rows = n columns in ppcs_samples[variable]
        which should correspond to number of input data points
        """
        # row = sample, column = original data point
        sample_array = ppc_samples[variable]
        hpds = pm.hpd(sample_array, alpha=.3)

        d = dict()
        # collect all samples into one field per input row
        # more elegant way?
        d[f'{variable}_samples'] = list(sample_array.T)
        d[f'{variable}_hpd_lower'] = hpds[:, 0]
        d[f'{variable}_hpd_upper'] = hpds[:, 1]
        d[f'{variable}_mean'] = np.mean(sample_array, axis=0)

        return (
            pd
            .DataFrame(d)
        )

