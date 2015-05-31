import unittest
from utils import *
from features_handler import *

class TestMLx(unittest.TestCase):
    def __init__(self):
        self.train_df, self.train_labels = load_data('../samples/census_income.train.csv',
                                                     label_col='Label')

    def test_QuantileNormalizer(self):
        handler = QuantileNormalizer(['Age'])
        handler.learn(self.train_df)
        self.assertEqual(handler._threshold_groups,
                         [[20, 23, 25, 27, 29, 31, 33, 35, 38, 41, 44, 47, 51, 56, 62]])

    def test_isupper(self):
      self.assertTrue('FOO'.isupper())
      self.assertFalse('Foo'.isupper())

    def test_split(self):
      s = 'hello world'
      self.assertEqual(s.split(), ['hello', 'world'])
      # check that s.split fails when the separator is not a string
      with self.assertRaises(TypeError):
          s.split(2)

if __name__ == '__main__':
    unittest.main()