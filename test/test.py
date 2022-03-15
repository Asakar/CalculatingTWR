import pandas as pd
import unittest

from CashFlow import TWR


class TestTaskOne(unittest.TestCase):
    """
    This uses values from https://www.investopedia.com/terms/t/time-weightedror.asp to test the
    application.
    """

    def testTwrPositiveCashFlow(self):
        data = {"total_valuation": [1000000, 1162484, 1192328], "cash_flow": [0, 100000, 0]}
        df = pd.DataFrame(data)
        twr = TWR.calculate_total_time_weighted_return(df)
        self.assertEqual(0.1, twr)

    def testTwrNegativeCashFlow(self):
        data = {"total_valuation": [1000000, 1162484, 1003440], "cash_flow": [0, -100000, 0]}
        df = pd.DataFrame(data)
        twr = TWR.calculate_total_time_weighted_return(df)
        self.assertEqual(0.1, twr)

    def testCalculateReturnsNegativeCashFlow(self):
        data = {"total_valuation": [1000000, 1162484, 1003440], "cash_flow": [0, -100000, 0]}
        df = pd.DataFrame(data)
        df = TWR.preprocess_data(df)
        returns = TWR.calc_returns(df["total_valuation"].values,
                                   df["prev_cash_flow"].values,
                                   df["prev_total_valuation"].values)
        for result, actual in zip(returns, [0.162, -0.056]):
            self.assertEqual(actual, round(result, 3))

    def testCalculateReturnsPositiveCashFlow(self):
        data = {"total_valuation": [1000000, 1162484, 1192328], "cash_flow": [0, 100000, 0]}
        df = pd.DataFrame(data)
        df = TWR.preprocess_data(df)
        returns = TWR.calc_returns(df["total_valuation"].values,
                                   df["prev_cash_flow"].values,
                                   df["prev_total_valuation"].values)
        for result, actual in zip(returns, [0.162, -0.056]):
            self.assertEqual(actual, round(result, 3))
