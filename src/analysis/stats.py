import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(rets):
    stats = {}
    stats["avg"] = np.mean(rets)
    stats["hit_rate"] = sum([x>0 for x in rets]) / len(rets)
    stats["max_ret"] = max(rets)
    return pd.DataFrame(stats)


def compute_annualised_stats(rets):

    stats = {}
    stats["avg"] = rets.mean()*252
    stats["vol"] = rets.std()*np.sqrt(252)
    stats["sharpe"] = stats["avg"]/stats["vol"]
    stats["hit_rate"] = (rets>0).mean()

    stats = pd.DataFrame(stats)

    return stats


def analyze_signal(rets,signal):
    analysis = {}

    pos_rets = []
    neg_rets = []
    for i in range(len(rets)):
        if signal[i] > 1:
            pos_rets.append(rets[i])
        elif signal[i] < -1:
            neg_rets.append(rets[i])

    analysis["pos_ret"] = np.mean(pos_rets)
    analysis["neg_ret"] = np.mean(neg_rets)
    analysis["spread"] = analysis["pos_ret"] - analysis["neg_ret"]

    return pd.DataFrame(analysis)


def compute_alpha(ret, days=252, mkt_ticker="BTCUSDT"):
    corr = ret.rolling(days).corr(ret[mkt_ticker])
    vol = ret.rolling(days).std()
    beta = (corr*vol).divide(vol[mkt_ticker],axis=0)
    resid = ret - beta.multiply(ret[mkt_ticker],0)

    return resid


def drawdown(px):

    dd = (px / px.expanding(min_periods=1).max() - 1)

    dd.plot()
    plt.show()

    return dd


def duration(px):
    
    peak = px.expanding(min_periods=1).max()
    res = pd.DataFrame(index=px.index,columns=px.columns)
    
    for col in px.columns:
        for dt in px.index:
            
            if px.loc[dt,col] >= peak.loc[dt,col]:
                 res.loc[dt,col] = 0
    
            else:    
                res.loc[dt,col] = res.loc[:dt,col].iloc[-2] + 1
    
    res.plot()
    plt.show()

    return res