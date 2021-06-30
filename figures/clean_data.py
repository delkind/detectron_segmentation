import itertools

import numpy as np
import seaborn as sns
import scipy.stats
from sklearn.linear_model import LinearRegression

from figures.util import produce_figure, main_regions, produce_plot_data, get_subplots, extract_data, plot_grid, \
    plot_annotations, plot_scatter

CORRELATION_THRESHOLD = 0.25
MIN_SAMPLE_COUNT = 100
SYMMETRY_SCORE_THRESHOLD = 0.155
COMMON_STRUCTS_THRESHOLD = 321
CELL_COUNT_THRESHOLD = 20000


def prepare_data(data, plot=False):
    valid_data = data
    valid_data = filter_dark_brains(valid_data)
    valid_data = filter_small_regions(valid_data)
    valid_data = filter_count_outliers(valid_data)
    valid_data = filter_brightness_correlated_data(valid_data, plot)
    valid_data = filter_asymmetric_regions(valid_data)
    return valid_data


def filter_count_outliers(data):
    experiments = list(data.keys())
    counts = [data[e]['grey']['count'] for e in experiments]
    valid_indices = np.where((counts > np.percentile(counts, 5)) & (counts < np.percentile(counts, 95)))
    valid_experiments = set(np.array(experiments)[valid_indices].tolist())
    return {e: d for e, d in data.items() if e in valid_experiments}


def filter_asymmetric_regions(data, plot=False):
    if plot:
        res = plot_symmetry_score(data, annotate_filtered=True)
    else:
        res = calculate_symmetry_score(data)

    data = {e: {s: d for s, d in ed.items() if res[s] <= SYMMETRY_SCORE_THRESHOLD}
            for e, ed in data.items()}
    return data


def filter_small_regions(data):
    regions = list(data[list(data.keys())[0]].keys())
    counts = {r: np.mean([data[e][r]['count'] for e in data.keys()]) for r in regions}
    data = {e: {s: d for s, d in ed.items() if counts[s] > CELL_COUNT_THRESHOLD} for e, ed in data.items()}
    return data


def filter_dark_brains(data):
    return {e: d for e, d in data.items() if d['grey']['brightness']['median'] > 25}


def determine_correlation_thresholds(data, param1, param2, correlation_threshold=CORRELATION_THRESHOLD):
    experiments = list(data.keys())
    structs = list(data[experiments[0]].keys())
    res = {p: {struct: [extract_data(data[e][struct], p) for e in experiments] for struct in structs}
           for p in (param1, param2)}

    results = dict()
    for region in structs:
        x = np.copy(res[param2][region])
        y = np.copy(res[param1][region])
        for br_threshold in range(5, 200, 5):
            valid_indices = np.where(x > br_threshold)
            x = x[valid_indices]
            y = y[valid_indices]
            if min(len(x), len(y)) < MIN_SAMPLE_COUNT:
                break
            correlation, pvalue = scipy.stats.pearsonr(x, y)
            if abs(correlation) < correlation_threshold:
                results[region] = (br_threshold, np.array(experiments)[valid_indices].tolist())
                break

    return results


def plot_symmetry_score(data, annotate_filtered=False):
    scores = calculate_symmetry_score(data)
    res = sorted(scores.items(), key=lambda e: e[1], reverse=True)
    x, y = np.array(list(zip(*res)))
    y = y.astype(np.double)
    fig, ax = get_subplots()

    for region, structs in main_regions.items():
        indices = np.where(np.isin(x.astype(str), structs))
        plot_scatter(ax, indices, y[indices], label=region)

    ax.set_xticks(list(range(len(x))[::50]))
    ax.set_xticklabels(x[::50], size=20)
    plot_grid(ax, [], [SYMMETRY_SCORE_THRESHOLD])

    if annotate_filtered:
        anns = np.where(y > SYMMETRY_SCORE_THRESHOLD)
        plot_annotations(ax, x[anns], np.arange(len(anns[0])), y[anns])

    produce_figure(ax, fig, "symmetry_score_linear", "", "", legend=True, format_xticks=False)
    ax.set_yscale('log')
    produce_figure(ax, fig, "symmetry_score_log", "", "", legend=False, format_xticks=False)
    return scores


def calculate_symmetry_score(data):
    res = produce_plot_data(data,
                            {
                                'l+r': lambda s: s['count_left'] + s['count_right'],
                                'l-r': lambda s: abs(s['count_left'] - s['count_right']),
                            },
                            {
                                'l+r': np.mean,
                                'l-r': np.mean
                            })
    res = {s: abs(res['l-r'][s] / res['l+r'][s]) for s in res['l+r'].keys()}
    return res


def find_inflection_point(xs, ys):
    diffs = np.abs(ys[:-1] - ys[1:])
    inf_pts = np.where(diffs > 40)[0]
    return inf_pts[-1] if len(inf_pts) > 0 else -1


def determine_correlation_intersection(ax, data, plot=False, correlation_threshold=CORRELATION_THRESHOLD):
    thresholds = determine_correlation_thresholds(data, 'count', 'brightness', correlation_threshold)
    # exps = sorted([(s, set(es)) for s, (t, es) in thresholds.items()], key=lambda s: len(s[1]), reverse=True)
    exp_thresholds = {e: {s for s, es in thresholds.items() if e in es[1]} for e in data.keys()}
    exps = sorted([(e, s) for e, s in exp_thresholds.items()], key=lambda s: len(s[1]), reverse=True)
    intersections = [tuple(zip(*exps[:i])) for i in range(1, len(exps))]
    intersections = [(set(s), set.intersection(*e)) for s, e in intersections]
    exp_sets, struct_sets = tuple(zip(*sorted(intersections, key=lambda t: len(t[1]), reverse=True)))
    ys = np.array([len(s) for s in struct_sets])
    xs = np.array([len(s) for s in exp_sets])
    inf_pt = find_inflection_point(xs, ys)
    if plot:
        plot_scatter(ax, xs, ys, label=str(correlation_threshold))
        plot_scatter(ax, xs[inf_pt], ys[inf_pt], s=100)
        plot_grid(ax, [xs[inf_pt]], [ys[inf_pt]])
    return thresholds, inf_pt, intersections, xs, ys


def plot_brightness_threshold_curves(data):
    fig, ax = get_subplots()
    for i in range(10, 51, 2):
        thresholds, inf_pt, intersections, xs, ys = determine_correlation_intersection(ax, data, True, i / 100)
        plot_annotations(ax, f'{i / 100} ({ys[inf_pt]} regions, {xs[inf_pt]} brains)', [xs[inf_pt]], [ys[inf_pt]])
    produce_figure(ax, fig, "valid_regions_vs_brains", "brains", "regions", legend=True)


def filter_brightness_correlated_data(data, plot=False):
    if plot:
        fig, ax = get_subplots()
    else:
        fig, ax = (None, None)

    thresholds, inf_pt, intersections, xs, ys = determine_correlation_intersection(ax, data, plot=plot)

    if plot:
        produce_figure(ax, fig, "valid_regions_vs_brains", "brains", "regions")

    valid_structs = list(intersections[inf_pt][1])
    valid_experiments = list(intersections[inf_pt][0])
    valid_data = {e: {s: data[e][s] for s in valid_structs} for e in valid_experiments}

    counts = {r: np.mean([e[r]['count'] for e in data.values()]) for r in thresholds.keys()}

    if plot:
        regions = sorted([(k, sorted(list(g), key=lambda x: counts[x], reverse=True))
                          for k, g in itertools.groupby(sorted(thresholds.keys(),
                                                               key=lambda x: thresholds[x][0]),
                                                        lambda s: thresholds[s][0])], key=lambda x: x[0])

        fig, ax = get_subplots()
        x = np.array([e[regions[0][1][1]]['count'] for e in valid_data.values()])
        y = np.array([e[regions[0][1][1]]['brightness']['mean'] for e in valid_data.values()])
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        plot_scatter(ax, x, y)
        ax.plot(x, reg.predict(x.reshape(-1, 1)), color='blue', linewidth=3)
        produce_figure(ax, fig, "valid_count_vs_brightness", xlabel=f'{regions[0][1][1]}. corr: {scipy.stats.pearsonr(x, y)[0]:.2f}')

        fig, ax = get_subplots()
        x = np.array([e[regions[-1][1][0]]['count'] for e in data.values()])
        y = np.array([e[regions[-1][1][0]]['brightness']['mean'] for e in data.values()])
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        plot_scatter(ax, x, y)
        ax.plot(x, reg.predict(x.reshape(-1, 1)), color='blue', linewidth=3)
        produce_figure(ax, fig, "invalid_count_vs_brightness", xlabel=f'{regions[-1][1][0]}. corr: {scipy.stats.pearsonr(x, y)[0]:.2f}')

    return valid_data
