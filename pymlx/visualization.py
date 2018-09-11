import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

# faster in visualization


def visualize_feature_importance(feature_scores: dict, limit=-1, fig_width=8, fig_height=16):
    """
    :param feature_scores: the result from model.get_fscores().to_dict()
    :param limit: get only top n features
    :param fig_width: figure width
    :param fig_height: figure height
    :return:
    """
    scores_df = pd.DataFrame({
        'feature': list(feature_scores.keys()),
        'score':  np.fromiter(feature_scores.values(), dtype=int)
    })

    if limit < 0:
        limit = len(scores_df)

    plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    sns.barplot(x='score', y='feature', data=scores_df.sort_values('score', ascending=False).iloc[0:limit], ax=ax)
    ax.set_title('Feature importance scores', fontweight='bold', fontsize=14)
    plt.show()


def visualize_feat_density(df, feature_name, target_col):
    """
    :param result_df: DataFrame that includes target col
    :param feature_name: a feature name
    :param target_col: target col name
    :return:
    """
    sns.jointplot(feature_name, target_col, data=df, kind="kde")
    plt.show()