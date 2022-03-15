import pandas as pd
import numpy as np


class TWR:
    """
    This calculates the time weighted returns, makes use of vectorisation with numpy.
    """

    @classmethod
    def preprocess_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data in order to calculate the returns optimally.
        :param data: DataFrame, containing the raw data.
        :return: DataFrame, holding data processed for calculating returns.
        """
        df = data[(data["cash_flow"] != 0)]
        df = pd.concat([data[:1], df, data[-1:]], axis=0)
        df.reset_index(inplace=True, drop=True)
        df["prev_cash_flow"] = df["cash_flow"].shift(1).fillna(0).astype(int)
        df["prev_total_valuation"] = df["total_valuation"].shift(1).fillna(0).astype(int)
        df.drop(0, inplace=True)
        return df

    @classmethod
    def calc_returns(cls, tot_val: int, prev_cash_flow: int,
                     prev_tot_val: int) -> float:
        """
        This calculates the returns for each sub period.
        :param tot_val: int, the final total valuation.
        :param prev_cash_flow: int, the cash flow to be considered for this period.
        :param prev_tot_val: int, the initial total valuation.
        :return: float, the return for the sub-period.
        """
        temp = prev_tot_val + prev_cash_flow
        return (tot_val - temp) / temp

    @classmethod
    def calculate_total_time_weighted_return(cls, data: pd.DataFrame) -> pd.Series:
        """
        This calculates the TWR using helper methods.
        :param data: DataFrame, this contains the raw data.
        :return: pd.Series containing the TWR.
        """
        df = cls.preprocess_data(data)
        returns = cls.calc_returns(df["total_valuation"].values,
                                   df["prev_cash_flow"].values,
                                   df["prev_total_valuation"].values)
        return np.round(np.prod((returns + 1)) - 1, 2)
