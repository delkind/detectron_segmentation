import base64
import itertools
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display, Markdown, HTML

from explorer.explorer_utils import hist, retrieve_nested_path
from explorer.ui import ExperimentsSelector, ResultsSelector


class DataSelector(widgets.VBox):
    def __init__(self, data_dir, results_selector):
        # self.experiment_selector = ExperimentsSelector([e for e in os.listdir(data_dir) if e.isdigit()])
        self.experiment_selector = ExperimentsSelector(results_selector.get_available_brains())
        self.results_selector = results_selector
        self.add_button = widgets.Button(description='Add')
        self.add_button.on_click(lambda b: self.add_data())
        self.remove_button = widgets.Button(description='Remove')
        self.remove_button.on_click(lambda b: self.remove_data())
        self.clear_button = widgets.Button(description='Reset')
        self.clear_button.on_click(lambda b: self.reset_data())
        self.output = widgets.Output()
        self.messages = widgets.Output()
        self.added = widgets.SelectMultiple(options=[])
        self.data = {}
        super().__init__((
            self.experiment_selector, self.results_selector,
            widgets.HBox((self.add_button, self.remove_button, self.clear_button), layout=widgets.Layout(width='auto')),
            self.messages,
            widgets.HBox((self.added,)),
            self.output))

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def reset_data(self):
        self.output.clear_output()
        self.added.options = ()
        self.messages.clear_output()
        self.data = {}

    def remove_data(self):
        if self.added.value:
            for v in self.added.value:
                del self.data[v]
            vals = [o for o in self.added.value]
            self.added.options = [o for o in self.added.options if o not in vals]
            self.messages.clear_output()
            with self.messages:
                display(Markdown("Selection removed"))

    def extract_values(self):
        if not self.data:
            self.output_message("Nothing to process")
            return {}

        if self.added.value:
            values = {k: self.data[k] for k in self.added.value}
        else:
            values = self.data

        return values

    def add_data(self):
        relevant_experiments = self.experiment_selector.get_selection()
        if len(relevant_experiments) == 0:
            self.output_message('Nothing to add, no relevant brains available')
        else:
            path = self.results_selector.get_selection_label()
            data = self.results_selector.get_selection(relevant_experiments)

            if data is None:
                self.output_message(f'Nothing to add')
                return

            label = f"{self.experiment_selector.get_selection_label()}.{path} ({len(relevant_experiments)})"

            if label not in self.data:
                self.added.options += (label,)
                self.data[label] = data
                self.output_message(f'Added data for {len(relevant_experiments)} brains')
            else:
                self.output_message(f'Already added')


class BrainAggregatesHistogramPlot(widgets.VBox):
    def __init__(self, data_dir, raw_data_selector):
        # self.data_selector = DataSelector(data_dir, ResultsSelector(pickle.load(open(f'{data_dir}/../stats.pickle',
        #                                                                              'rb'))))
        self.data_selector = DataSelector(data_dir, raw_data_selector)
        self.bins = widgets.IntSlider(min=10, max=100, value=50, description='Bins: ')
        self.plot_hist_button = widgets.Button(description='Plot histogram')
        self.plot_hist_button.on_click(lambda b: self.plot_data(self.do_histogram_plot))
        self.plot_violin_button = widgets.Button(description='Plot violin')
        self.plot_violin_button.on_click(lambda b: self.plot_data(self.do_violin_plot))
        self.ttest_button = widgets.Button(description='T-Test')
        self.ttest_button.on_click(lambda b: self.test(stats.ttest_ind))
        self.kstest_button = widgets.Button(description='KS-Test')
        self.kstest_button.on_click(lambda b: self.test(stats.kstest))
        self.output = widgets.Output()
        self.messages = widgets.Output()
        header = widgets.Output()
        with header:
            display(Markdown("----"))
            display(Markdown("## Histogram"), )
        super().__init__((
            header,
            self.data_selector,
            self.messages,
            widgets.HBox((self.plot_hist_button, self.plot_violin_button, self.ttest_button, self.kstest_button)),
            self.output))
        self.histograms = dict()

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def plot_data(self, plotter):
        values = self.data_selector.extract_values()

        if values:
            self.output.clear_output()

            with self.output:
                df = pd.DataFrame({k: pd.Series(v) for k, v in values.items()})
                csv = df.to_csv()
                b64 = base64.b64encode(csv.encode())
                payload = b64.decode()
                html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
                html = html.format(payload=payload, title="Click to download data", filename='data.csv')
                display(HTML(html))
                plotter(values)
        else:
            with self.messages:
                self.messages.clear_output()

    def do_histogram_plot(self, values):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for l, d in values.items():
            hist(ax, d, bins=self.bins.value, label=l)
        ax.legend()
        plt.show()

    @staticmethod
    def do_violin_plot(values):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.violinplot(list(values.values()), showmeans=True, showmedians=True, showextrema=True)
        ax.xaxis.set_tick_params(direction='out', rotation=67)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(values) + 1))
        ax.set_xticklabels(list(values.keys()))
        ax.set_xlim(0.25, len(values) + 0.75)
        plt.show()

    def test(self, test):
        self.messages.clear_output()
        values = self.data_selector.extract_values()
        keys = list(values.keys())
        self.messages.clear_output()
        for l, r in set(itertools.combinations(range(len(keys)), 2)):
            with self.messages:
                display(Markdown(
                    f'({values[keys[r]].mean() - values[keys[l]].mean()}) {keys[l]}, {keys[r]}: {str(test(values[keys[r]], values[keys[l]]))}'))


class BrainAggregatesScatterPlot(widgets.VBox):
    def __init__(self, data):
        self.data = data
        self.experiment_selector = ExperimentsSelector(data.keys())
        self.results_selector_x = ResultsSelector(data)
        self.results_selector_y = ResultsSelector(data)
        self.add_button = widgets.Button(description='Add')
        self.add_button.on_click(lambda b: self.add_data())
        self.clear_button = widgets.Button(description='Reset')
        self.clear_button.on_click(lambda b: self.reset_data())
        self.plot_button = widgets.Button(description='Plot')
        self.plot_button.on_click(lambda b: self.plot())
        self.output = widgets.Output()
        self.messages = widgets.Output()
        self.added = widgets.SelectMultiple(options=[], layout=widgets.Layout(width='auto'))
        header = widgets.Output()
        with header:
            display(Markdown("---"))
            display(Markdown("## Scatter plot"), )
        super().__init__((
            header,
            self.experiment_selector,
            widgets.HBox((widgets.Label("X:", layout=widgets.Layout(width='auto')),
                          self.results_selector_x)),
            widgets.HBox((widgets.Label("Y:", layout=widgets.Layout(width='auto')),
                          self.results_selector_y)),
            widgets.HBox((self.add_button, self.clear_button), layout=widgets.Layout(width='auto')),
            self.messages,
            self.added,
            self.plot_button,
            self.output))
        self.plot_data = dict()

    def reset_data(self):
        self.output.clear_output()
        self.plot_data = {}
        self.added.options = ()

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def plot(self):
        if not self.plot_data:
            self.output_message("Nothing to plot")
            return

        if self.added.value:
            values = {k: self.plot_data[k] for k in self.added.value}
        else:
            values = self.plot_data

        self.output.clear_output()

        with self.output:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for l, (x, y) in values.items():
                print(x)
                print(y)
                ax.scatter(x, y, label=l)

            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.legend()

            ax.legend()
            plt.show()

    def add_data(self):
        path_x = self.results_selector_x.get_selection()
        path_y = self.results_selector_y.get_selection()
        relevant_experiments = self.experiment_selector.get_selection()
        if len(relevant_experiments) == 0:
            self.output_message('Nothing to add, no relevant brains available')
        else:
            label = f"({len(relevant_experiments)}) {'.'.join(path_x)}:{'.'.join(path_y)} for {self.experiment_selector.get_selection_label()}"
            if label not in self.plot_data:
                self.output_message(f'Added data for {len(relevant_experiments)} brains')
                self.added.options = self.added.options + (label,)
                self.plot_data[label] = ((np.array([retrieve_nested_path(d, path_x)
                                                    for e, d in self.data.items() if int(e) in relevant_experiments]),
                                          np.array([retrieve_nested_path(d, path_y)
                                                    for e, d in self.data.items() if int(e) in relevant_experiments])))
            else:
                self.output_message(f'Already added, ignored')
