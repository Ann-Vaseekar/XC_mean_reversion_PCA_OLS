import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calc_resid_pca(
    ret,
    n_components=3,
    window_size=60,
    min_obs_ratio=0.8,
    plot_variance=False,
    freq="4h"
):
    """
    Compute rolling PCA-based market-neutral residual returns.

    daily_rets (df):                Raw return matrix 
    n_components (int):             Number of PCA components to remove
    window_size (int):              Rolling window length in bars
    min_obs_ratio (float):          Min fraction of window required for a column to be included
    plot_variance (bool):           If True, plot rolling explained variance of first 3 components
    """

    min_obs = int(window_size * min_obs_ratio)
    valid_cols_mask = ret.notna().sum(axis=0) >= min_obs
    ret = ret.loc[:, valid_cols_mask]

    ret_np = ret.to_numpy().astype(float)         # (T, N)
    was_nan = np.isnan(ret_np)                    # track original missingness

    # Fill NaN with 0 only for computation; re-mask at the end
    ret_filled = np.where(was_nan, 0.0, ret_np)   # (T, N)

    n_bars = len(ret_filled)

    resid_list = []
    variance_records = []

    for i in range(window_size, n_bars):

        window_data = ret_filled[i - window_size: i]          # (W, N)

        mean = window_data.mean(axis=0)
        std = window_data.std(axis=0, ddof=1)
        std[std == 0] = 1.0                        

        window_scaled = (window_data - mean) / std

        pca = PCA(n_components=n_components)
        pca.fit(window_scaled)

        # enforce deterministic PCA component sign
        for k in range(n_components):
            if pca.components_[k, np.argmax(np.abs(pca.components_[k]))] < 0:
                pca.components_[k] *= -1

        if plot_variance:
            n_plot = min(3, n_components)
            evr = pca.explained_variance_ratio_[:n_plot].tolist()
            # pad to length 3 if n_components < 3
            evr += [np.nan] * (3 - len(evr))
            variance_records.append(evr)

        curr = ret_filled[i: i + 1]                               # (1, N)
        curr_scaled = (curr - mean) / std

        scores = pca.transform(curr_scaled)                # (1, n_components)
        common = scores @ pca.components_                  # (1, N)
        residual = curr_scaled - common                            # (1, N) — standardised space

        resid_list.append(residual)

    resid_array = np.vstack(resid_list)                           # (T - W, N)

    # Re-mask positions where original ret data was NaN
    original_nan_mask = was_nan[window_size:]
    resid_array[original_nan_mask] = np.nan

    resid_df = pd.DataFrame(
        resid_array,
        index=ret.index[window_size:],
        columns=ret.columns,
    )

    if plot_variance:
        n_plot = min(3, n_components)
        var_df = pd.DataFrame(
            variance_records,
            index=ret.index[window_size:],
            columns=[f"PC{k+1}" for k in range(3)],
        ).iloc[:, :n_plot]

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#2563eb", "#dc2626", "#16a34a"]
        for k, col in enumerate(var_df.columns):
            ax.plot(var_df.index, var_df[col] * 100, label=col,
                    color=colors[k], linewidth=1.5, alpha=0.9)

        ax.set_title(
            f"Rolling Explained Variance by Component (window={window_size // (24 // int(freq.replace('h', '')))}-days)"
        )
        ax.set_ylabel("Explained Variance (%)")
        ax.set_xlabel("")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.show()

    return resid_df


def plot_explained_variance(daily_rets, n_comps=3):

    ret_clean = daily_rets.dropna(how="all", axis=1).fillna(0)
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(ret_clean)

    pca = PCA()
    pca.fit(returns_scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        np.cumsum(pca.explained_variance_ratio_),
        "bo-",
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Scree Plot (full-sample — diagnostic only)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Explained variance for first {n_comps} components:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:n_comps]):
        print(f"  PC{i+1}: {ratio:.3f}")


def calc_pca_loadings(
    rets,
    window_size,
    n_components = 1,
):
    """
    Compute rolling PCA loadings over time.
    Returns DataFrame of shape (n_bars, n_assets) for each component.
    """

    loadings = []

    for i in range(window_size, len(rets)):
        window = rets.iloc[i - window_size:i].fillna(0)
        
        scaler     = StandardScaler()
        scaled     = scaler.fit_transform(window)
        pca        = PCA(n_components=n_components)
        pca.fit(scaled)

        # PC1 loadings — flip sign so BTC loading is always positive
        pc1 = pca.components_[0]
        if pc1[rets.columns.get_loc("BTCUSDT")] < 0:
            pc1 = -pc1

        loadings.append(pc1)

    return pd.DataFrame(
        loadings,
        index=rets.index[window_size:],
        columns=rets.columns,
    )