import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def calc_resid_ols(
    ret: pd.DataFrame,
    window_size: int = 400,
    mkt_ticker: str = "BTCUSDT",
    min_obs_ratio: float = 0.8,
):
    """
    Compute market-neutral residuals using rolling OLS regression.

    Parameters:
        ret (df): Raw return matrix.
        window_size (int): Rolling window length in bars.
        mkt_ticker (str): Market-proxy ticker.
        min_obs_ratio (float): Min fraction of window required for a column to be included.
    """

    ret = ret.copy().astype(float)
    min_obs = int(window_size * min_obs_ratio)

    if mkt_ticker not in ret.columns:
        raise ValueError(f"{mkt_ticker} not in return columns")

    mkt = ret[mkt_ticker]

    # Rolling variance of market
    var_m = (
        mkt.rolling(window_size, min_periods=min_obs)
        .var()
        .replace(0, np.nan)
    )
    cov = ret.rolling(window_size, min_periods=min_obs).cov(mkt)
    beta = cov.divide(var_m, axis=0)
    resid = ret - beta.multiply(mkt, axis=0)

    resid = resid.drop(columns=[mkt_ticker])

    
    return resid
