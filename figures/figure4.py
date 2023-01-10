import itertools
import re
import time
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

from figures.clean_data import prepare_data
from figures.figure1 import find_leaves
from figures.util import load_data, mcc, get_subplots, produce_figure, plot_scatter

from tqdm import tqdm

import matplotlib.pyplot as plt

from scratch.svm_analysis import get_importance

CLASSES = {"M": 0, "F": 1}


def produce_data(raw_data, param, gender=None):
    if gender is not None:
        raw_data = raw_data[raw_data.gender == gender]
    grouped_data = raw_data[param].groupby(raw_data.experiment_id).apply(np.array)
    ids = grouped_data.index.to_numpy()
    values = grouped_data.values
    genders = [gender] * ids.shape[0] if gender is not None else [0] * ids.shape[0]
    return pd.DataFrame([ids, values, genders]).T


def dataset_to_xy(dataset):
    return np.stack(dataset[1].to_numpy()), np.array([CLASSES[g] for g in dataset[2]])


def dataset_to_x(dataset):
    return np.stack(dataset[1].to_numpy())


def svm_factory(kernel, c):
    def create_svm():
        return svm.SVC(kernel=kernel, C=c)

    return create_svm


def tree_factory(max_depth):
    def create_tree():
        return tree.DecisionTreeClassifier(max_depth=max_depth)

    return create_tree


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    xx, yy = np.meshgrid(
        np.arange(x_min - (x_max-x_min) / 50, x_max + (x_max-x_min)/ 25, (x_max-x_min)/100),
        np.arange(y_min - (y_max-y_min) / 50, y_max + (y_max-y_min) / 25, (y_max-y_min) / 100)
    )
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def evaluate_accuracy(t):
    c, X, y = t
    svc = svm.SVC(kernel="linear", C=c)
    svc.fit(X, y)
    return c, svc.score(X, y)


def figure4(data):
    experiments = mcc.get_experiments(dataframe=True)
    data = data.join(experiments[['gender', 'strain']], on='experiment_id')
    nan_data = data[['experiment_id', 'region', 'density3d']].set_index(['experiment_id', 'region']).unstack()
    columns = [t[1] for t in nan_data.columns.to_list()]
    nan_data.columns = columns
    brains_by_region = [(brain, nan_data.loc[brain, :].dropna().index.to_list()) for brain in
                        nan_data.index.to_list()]
    brains_by_region = sorted(brains_by_region, key=lambda r: len(r[1]), reverse=True)
    brains = [t[0] for t in brains_by_region]
    regions = [set(t[1]) for t in brains_by_region]

    intersections = [(tuple(brains[:i]), list(set.intersection(*(regions[:i])))) for i in
                     range(1, len(brains_by_region))]
    x, y = zip(*[[len(t[0]), len(t[1])] for t in intersections])

    valid_regions = intersections[-1][1]
    valid_brains = intersections[-1][0]

    valid_regions = np.array(valid_regions)[find_leaves(valid_regions)]

    data = data[data.experiment_id.isin(valid_brains) & data.region.isin(valid_regions)]

    valid_regions = data.region.unique()

    results = list()
    classifiers = dict()

    for strain in ['C57BL/6J', 'FVB.CD1(ICR)']:
        for param in ["density3d", 'volume']:
            title = f'strain: {strain}, parameter: {param}'
            fname = f"{re.sub(r'[^a-zA-Z0-9]', '', strain)}-{param}"

            if strain != "ALL":
                strain_data = data[(data.strain == strain)]
            else:
                strain_data = data
            males = produce_data(strain_data, param, "M")
            females = produce_data(strain_data, param, "F")
            svm_params = itertools.product(['linear'], [i / 10 for i in list(range(1, 6))])
            classifier_factories = {
                **{f"SVM (kernel = {kernel}, C = {C})": (svm_factory(kernel, C), True) for kernel, C in svm_params},
                # **{f"decision tree (max_depth={d})": (tree_factory(d), False) for d in range(1, 4)}
            }

            for name, (clf_factory, should_scale) in classifier_factories.items():
                train_scores = list()
                test_scores = list()
                clfs = list()
                for _ in tqdm(range(100), "Training SVM..."):
                    cur_time = time.time()
                    train_males, test_males = train_test_split(males, test_size=0.33,
                                                               random_state=int((cur_time - int(cur_time)) * 10000))
                    train_females, test_females = train_test_split(females, test_size=0.33,
                                                                   random_state=int(
                                                                       (cur_time - int(cur_time)) * 10000) + 1)
                    train_data = pd.concat([train_males, train_females])
                    test_data = pd.concat([test_males, test_females])

                    X_train, y_train = dataset_to_xy(train_data)
                    X_test, y_test = dataset_to_xy(test_data)

                    if should_scale:
                        sc = StandardScaler()
                        X_train = sc.fit_transform(X_train)
                        X_test = sc.transform(X_test)

                    clf = clf_factory()
                    clf.fit(X_train, y_train)
                    train_scores.append(clf.score(X_train, y_train))
                    test_scores.append(clf.score(X_test, y_test))
                    clfs.append(clf)
                    pass

                results.append({"strain": strain, "param": param, "classifier": name,
                                "train_acc_mean": np.mean(train_scores), "train_acc_std": np.std(train_scores),
                                "test_acc_mean": np.mean(test_scores), "test_acc_std": np.std(test_scores),
                                "train_acc": train_scores, "test_acc": test_scores,
                                })
                classifiers[(strain, param, name)] = clfs

    results = pd.DataFrame(results)
    results.to_csv('./gender_classification_results.tsv', sep='\t')
    influences = {k: sorted([(r, c, s) for r, c, s in zip(valid_regions,
                                                    np.mean(list(get_importance(v)), axis=0),
                                                    np.std(list(get_importance(v)), axis=0))],
                             key=lambda t: -abs(t[1])) for k, v in classifiers.items()}

    results['is_svm'] = results.apply(lambda r: r['classifier'].startswith('SVM'), axis=1)
    grouped = results.groupby(["strain", "param", "is_svm"]).test_acc_mean.apply(np.array)
    grouped_names = results.groupby(["strain", "param", "is_svm"]).classifier.apply(np.array)
    accs = dict(list(grouped.items()))
    names = dict(list(grouped_names.items()))
    best = [((k[0], k[1], names[k][np.argmax(v)]), np.max(v)) for k, v in accs.items()]
    best_influences = {k: (v, influences[k]) for k, v in best if k[2].startswith("SVM")}

    for (strain, param, clf_name), (orig_score, infs) in best_influences.items():
        fig, ax = get_subplots()
        ax.barh(np.arange(10), [t[1] for t in infs[:10][::-1]])
        ax.set_yticks(np.arange(10))
        ax.set_yticklabels([t[0] for t in infs[:10][::-1]])
        fig.tight_layout()
        produce_figure(ax, fig, f"{strain}_{param}_top10", format_xticks=False, format_yticks=False)

        dataset = pd.pivot_table(data[data.strain == strain], index="experiment_id", columns="region", values=param).join(experiments[['gender']], on='experiment_id')

        if strain == "C57BL/6J":
            # scores = list()
            # for f_num in tqdm(range(1, 11), "Evaluating accuracy improvements..."):
            #     features = [t[0] for t in infs[:f_num]]
            #     X = dataset[features].to_numpy()
            #
            #     y = np.array([CLASSES[g] for g in dataset.gender.tolist()])
            #     with Pool(10) as pool:
            #         res = [r[1] for r in pool.map(evaluate_accuracy, [(i / 10, X, y) for i in range(1, 11)])]
            #         #res = [r[1] for r in pool.map(evaluate_accuracy, [(i / 10, X, y) for i in range(1, 11)])]
            #
            #     scores.append(res[np.argmax(res)])
            #
            # fig, ax = get_subplots()
            # ax.plot(np.arange(10), scores)
            # ax.set_xticks(np.arange(10))
            # ax.set_xticklabels([t[0] for t in infs[:10]], rotation=45)
            # fig.tight_layout()
            # produce_figure(ax,
            #                fig,
            #                f"{strain}_{param}_accuracy",
            #                format_xticks=False,
            #                format_yticks=False,
            #                xlabel="Added feature",
            #                ylabel="Accuracy")

            if param == 'volume':
                for axes in [["MEA", "LSr"], ["BST", "ORBvl2/3"], ["MEA", "SSp-m4"]]:
                    X = dataset[axes].to_numpy()
                    y = np.array([CLASSES[g] for g in dataset.gender.tolist()])
                    with Pool(10) as pool:
                        res = pool.map(evaluate_accuracy, [(i / 10, X, y) for i in range(1, 11)])

                    clf = svm.SVC(kernel="linear", C=res[np.argmax([r[1] for r in res])][0])
                    clf.fit(X, y)

                    print(f"{axes}: {clf.coef_.squeeze()}")

                    fig, ax = get_subplots()
                    # title for the plots
                    # Set-up grid for plotting.
                    X0, X1 = X[:, 0], X[:, 1]
                    xx, yy = make_meshgrid(X0, X1)
                    pred = clf.predict(X)
                    print(clf.score(X, y))

                    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
                    plot_scatter(ax, X0[y == pred], X1[y == pred], c=y[y == pred], marker='o', cmap=plt.cm.coolwarm, edgecolor='k', label="Correct")
                    plot_scatter(ax, X0[y != pred], X1[y != pred], c=y[y != pred], marker='d', cmap=plt.cm.coolwarm, edgecolor='k', label="Incorrect")
                    ax.set_xticks(())
                    ax.set_yticks(())
                    produce_figure(ax,
                                   fig,
                                   f"{strain}_{param}_{axes[0]}_{axes[1]}_svm",
                                   format_xticks=False,
                                   format_yticks=False,
                                   xlabel=axes[0],
                                   ylabel=axes[1],
                                   legend=True)

    save_results(classifiers, results, valid_regions)


def save_results(classifiers, results, regions):
    pickle.dump((classifiers, regions), open("classifiers.pickle", "wb"))


if __name__ == '__main__':
    figure4(prepare_data(load_data()))
