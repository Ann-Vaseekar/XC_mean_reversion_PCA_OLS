import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def calc_resid_ols(
    ret,
    window_size=400,
    mkt_ticker="BTCUSDT",
    min_obs_ratio=0.8,
):
    """
    Compute market-neutral residuals using rolling OLS regression.
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
