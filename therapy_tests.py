import unittest
import numpy as np
import pandas as pd
import theano
import therapy as t


class SharedVariableTests(unittest.TestCase):

    def setUp(self):
        self.model = t.Model(
            model=None,
            shared_variables={
                'sv1': theano.shared(None)
            },
            name=None
        )

    def tearDown(self):
        self.model = None

    def test_resets_shared_variables_after_update(self):
        sv_ = self.model.shared_variables['sv1']
        fixed_value = np.array([42])
        sv_.set_value(fixed_value)

        X = pd.DataFrame({'sv1': [1, 2]})
        with self.model._temporarily_update_shared_variables(X):
            # value is changed here
            self.assertTrue(
                all(X['sv1'].values == sv_.get_value())
            )
        # value is reset after
        self.assertTrue(all(sv_.get_value() == fixed_value))

    def test_permanently_updates_shared_variable_values(self):
        X = pd.DataFrame({'sv1': [1, 2]})
        self.model._permanently_update_shared_variables(X)
        element_wise_comparison = X['sv1'].values == self.model.shared_variables['sv1'].get_value()
        self.assertTrue(
            element_wise_comparison.all()
        )

    def test_gets_share_values(self):
        values = np.array([1, 2])
        sv = self.model.shared_variables['sv1']

        # set values
        sv.set_value(values)

        # read
        df = self.model._get_shared_variable_values()

        # check
        element_wise_comparison = sv.get_value() == df['sv1'].values
        self.assertTrue((all(element_wise_comparison)))


if __name__ == '__main__':
    unittest.main(warnings='ignore')
