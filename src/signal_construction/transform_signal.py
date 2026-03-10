import pandas as pd
from scipy.stats import norm


def transform_signal(
        signal: pd.DataFrame,
        how: str = "winsorize",
        thresh: float = None,
        rank_thresh: float = None,
):
    """
    Transforms raw signal.

    Parameters:
        signal (df): Raw signal matrix.
        how (str): Transformation method ("winsorize", "truncate", "rank", "inv_cdf").
        thresh (float): Threshold for winsorize/truncate.
        rank_thresh (float): Threshold for rank transform.
    """

    valid_how = ["winsorize", "truncate", "rank", "inv_cdf"]

    if how not in valid_how:
        raise ValueError(f"Invalid how='{how}', must be one of {valid_how}")

    if how in ("winsorize", "truncate"):
        high = signal.quantile(1 - thresh, axis=1)
        low = signal.quantile(thresh, axis=1)
        if how == "winsorize":
            return signal.clip(lower=low, upper=high, axis=0)
        mask = signal.le(high, axis=0) & signal.ge(low, axis=0)
        return signal.where(mask, 0)
    
    if how == "rank":
        ranked = signal.rank(axis=1, pct=True)
        scaled = ranked.sub(0.5).mul(2)
        if rank_thresh is None:
            return scaled
        mask = (ranked <= rank_thresh) | (ranked >= 1 - rank_thresh)
        return scaled.where(mask, 0)
    
    # how == "inv_cdf"
    ranked = signal.rank(axis=1, pct=True)
    eps = 1e-6
    ranked = ranked.clip(eps, 1 - eps)
    return pd.DataFrame(norm.ppf(ranked), index=ranked.index, columns=ranked.columns)


def standardise(signal: pd.DataFrame, window: int, min_period: int = 1):
    """
    Standardise signal with rolling window calculation.

    Parameters:
        signal (df): Raw signal matrix.
        window (int): Rolling window length.
        min_period (int): Min rolling window period length.
    """

    df_mean = signal.rolling(window, min_periods=min_period).mean()
    df_std = signal.rolling(window, min_periods=min_period).std()

    return (signal-df_mean) / df_std.replace(0, float("nan"))


def dollar_neutral_weights(signal: pd.DataFrame):
    """
    Converts a signal into dollar-neutral portfolio weights.

    Parameters:
        signal (df): Raw signal matrix (positive = long, negative = short).
    """

    longs = signal.where(signal > 0, 0)
    shorts = signal.where(signal < 0, 0)
    
    long_weights = 0.5 * longs.div(longs.sum(axis=1), axis=0)
    short_weights = 0.5 * shorts.div(shorts.abs().sum(axis=1), axis=0)
    
    weights = long_weights + short_weights
    return weights.fillna(0)