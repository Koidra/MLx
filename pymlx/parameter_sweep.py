from pandas import DataFrame
from sklearn.grid_search import RandomizedSearchCV

def random_sweep(X, y, model, params, n_iter=20, scoring='roc_auc', cv=3,
                 refit=False, n_jobs=1, verbose=2):
    sweeper = RandomizedSearchCV(model, params, scoring=scoring, n_iter=n_iter,
                                 cv=cv, refit=refit, n_jobs=n_jobs, verbose=verbose)
    sweeper.fit(X, y)  # run the random search, not fitting the model
    return sweeper

def sweep_stats(sweeper, high_is_good=True, save=None):
    stats = []
    for row in sorted(sweeper.grid_scores_,
                      key=lambda x: x.mean_validation_score, reverse=high_is_good):
        stat = {'Score': row.mean_validation_score, 'Std': row.cv_validation_scores.std()}
        parameters = row.parameters
        for param in parameters:
            stat[param] = parameters[param]
        stats.append(stat)
    return DataFrame(stats)

