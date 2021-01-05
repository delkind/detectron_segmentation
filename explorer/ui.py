import itertools
import os
import pickle

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from IPython.display import display, Markdown, Javascript
import PIL.Image as Image
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from annotate_cell_data import create_section_image
from explorer.explorer_utils import retrieve_nested_path, hist, DataFramesHolder, is_file_up_to_date, \
    plot_section_violin_diagram, plot_section_histograms
from localize_brain import detect_brain


class ExperimentsSelector(widgets.VBox):
    def __init__(self, available_brains=None):
        self.mcc = MouseConnectivityCache(manifest_file=f'../mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.messages = widgets.Output()
        self.set_available_brains(available_brains)
        self.filter = {
            col: widgets.SelectMultiple(description=col, options=self.get_column_options(self.experiments[col]))
            for col in ['gender', 'strain', 'transgenic_line', 'id']}
        for c, f in self.filter.items():
            f.observe(lambda change, col=c: self.selection_changed(change, col))
        self.change_handler = None
        self.selection_changed({'name': 'value'}, 'gender')

        super().__init__((widgets.HBox(list(self.filter.values())), self.messages))

    def reset(self):
        for v in self.filter.values():
            v.value = []

    @staticmethod
    def get_column_options(df):
        return sorted(df.unique().tolist())

    def set_available_brains(self, available_brains):
        self.experiments = self.mcc.get_experiments(dataframe=True)
        self.experiments.at[self.experiments.strain.isin([None]), 'strain'] = '<none>'
        self.experiments.at[self.experiments.transgenic_line.isin([None]), 'transgenic_line'] = '<none>'
        if available_brains is not None:
            self.experiments = self.experiments[self.experiments.id.isin([int(e) for e in available_brains])]
        self.messages.clear_output()
        with self.messages:
            display(Markdown(f'Selected {len(self.get_selection())} brains'))

    def get_filter_value(self, col):
        selection = self.filter[col].value
        if selection is None or not selection:
            selection = self.filter[col].options
        return selection

    def selection_changed(self, change, col):
        if change['name'] == 'value':
            self.messages.clear_output()
            with self.messages:
                display(Markdown(f'Selected {len(self.get_selection())} brains'))

            selection = self.get_filter_value(col)

            if col == 'gender':
                oldval = self.filter['strain'].value
                self.filter['strain'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(selection)].strain)
                oldval = [v for v in oldval if v in self.filter['strain'].options]
                self.filter['strain'].value = oldval
            elif col == 'strain':
                oldval = self.filter['transgenic_line'].value
                self.filter['transgenic_line'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(self.get_filter_value('gender'))
                                     & self.experiments.strain.isin(selection)].transgenic_line)
                oldval = [v for v in oldval if v in self.filter['transgenic_line'].options]
                self.filter['transgenic_line'].value = oldval
            elif col == 'transgenic_line':
                oldval = self.filter['id'].value
                self.filter['id'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(self.get_filter_value('gender')) &
                    self.experiments.strain.isin(self.get_filter_value('strain')) &
                    self.experiments.transgenic_line.isin(selection)].id)
                oldval = [v for v in oldval if v in self.filter['id'].options]
                self.filter['id'].value = oldval

            if self.change_handler is not None:
                self.change_handler(change, col)

    def get_selection(self):
        selection = {col: sel.value for col, sel in self.filter.items()}
        query_string = ' and '.join([f'{col} == {val}' for col, val in selection.items() if val])
        if query_string:
            selection = self.experiments.query(query_string)
        else:
            selection = self.experiments

        return set(selection.id.unique().tolist())

    def get_selection_label(self):
        label = '.'.join([f'({",".join([str(v) for v in val.value])})'
                          for val in self.filter.values() if val.value])
        if label == '':
            label = '<all>'

        return label

    def on_selection_change(self, handler):
        self.change_handler = handler


class ResultsSelector(widgets.HBox):
    def __init__(self, data):
        self.data = data
        self.selectors = [widgets.Dropdown(values=['Loading...']) for i in range(6)]
        for i, s in enumerate(self.selectors):
            s.observe(lambda c, i=i: self.on_change(i, c))
        self.template = list(data.values())[0]
        for s in self.selectors[1:]:
            self.enable_selector(s, False)
        self.selectors[0].options = list(self.template.keys())
        self.selectors[0].value = list(self.template.keys())[0]
        super().__init__(self.selectors)

    @staticmethod
    def enable_selector(selector, mode):
        if mode:
            selector.layout.visibility = 'visible'
        else:
            selector.value = None
            selector.options = []
            selector.layout.visibility = 'hidden'

    def on_change(self, num, change):
        if change['name'] == 'value' and self.selectors[num].options and self.selectors[num].value is not None:
            val = self.template

            for i in range(num):
                val = val[self.selectors[i].value]

            val = val[self.selectors[num].value]

            if not isinstance(val, dict):
                for i in range(num + 1, 6):
                    self.enable_selector(self.selectors[i], False)
            else:
                old_value = self.selectors[num + 1].value
                self.selectors[num + 1].options = list(val.keys())
                if old_value in val.keys():
                    self.selectors[num + 1].value = old_value
                else:
                    self.selectors[num + 1].value = self.selectors[num + 1].options[0]
                self.enable_selector(self.selectors[num + 1], True)

    def get_selection(self, relevant_experiments):
        return np.array([retrieve_nested_path(self.data[str(e)],
                                              self.get_selection_path()) for e in relevant_experiments])

    def get_selection_label(self):
        return '.'.join(self.get_selection_path())

    def get_selection_path(self):
        return [s.value for s in self.selectors if s.value is not None]


class RawDataResultsSelector(widgets.VBox):
    def __init__(self, data_dir):
        self.data_frames = DataFramesHolder(data_dir)
        self.data_template = self.data_frames[[e for e in os.listdir(data_dir) if e.isdigit()][0]]
        self.messages = widgets.Output()
        self.mcc = MouseConnectivityCache(manifest_file=f'mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.structure_tree = self.mcc.get_structure_tree()

        self.filter = {
            'structure': widgets.SelectMultiple(description="Region",
                                                options=[s['acronym']
                                                         for s in self.structure_tree.get_structures_by_id(
                                                        self.get_column_options(self.data_template['structure_id']))]),
            'dense': widgets.SelectMultiple(description="Dense", options=self.
                                            get_column_options(self.data_template.dense)),
            'parameter': widgets.Dropdown(description="Parameter",
                                          options=['density', 'area', 'perimeter']),
        }

        for c, f in self.filter.items():
            f.observe(lambda change, col=c: self.selection_changed(change, col))
        self.change_handler = None
        self.selection_changed({'name': 'value'}, 'structure')

        super().__init__((widgets.HBox(list(self.filter.values())), self.messages))

    def reset(self):
        for v in self.filter.values():
            v.value = []

    @staticmethod
    def get_column_options(df):
        return sorted(df.unique().tolist())

    def get_filter_value(self, col):
        selection = self.filter[col].value
        if selection is None or not selection:
            selection = self.filter[col].options
        return selection

    def selection_changed(self, change, col):
        if change['name'] == 'value':
            selection = self.get_filter_value(col)

            if col == 'structure':
                self.filter['dense'].options = self.get_column_options(self.data_template[self.data_template.structure_id.isin(
                        [self.structure_tree.get_id_acronym_map()[s] for s in selection])].dense)

            if self.change_handler is not None:
                self.change_handler(change, col)

    def process_data_frame(self, df):
        structs = [self.structure_tree.get_id_acronym_map()[s] for s in self.get_filter_value('structure')]
        frame = df[df.structure_id.isin(structs) & df.dense.isin(self.get_filter_value('dense'))]
        d = frame[self.filter['parameter'].value].to_numpy()
        return d

    def get_selection(self, relevant_experiments):
        return np.concatenate([self.process_data_frame(self.data_frames[e]) for e in relevant_experiments])

    def get_selection_label(self):
        label = '.'.join([f'({",".join([str(v) for v in self.filter[val].value])})'
                          for val in ['structure', 'dense'] if
                          self.filter[val].value]) + f'.{self.filter["parameter"].value}'
        if label == '':
            label = '<any>'

        return label

    def on_selection_change(self, handler):
        self.change_handler = handler


class DataSelector(widgets.VBox):
    def __init__(self, data_dir, results_selector):
        self.experiment_selector = ExperimentsSelector([e for e in os.listdir(data_dir) if e.isdigit()])
        self.results_selector = results_selector
        self.add_button = widgets.Button(description='Add')
        self.add_button.on_click(lambda b: self.add_data())
        self.clear_button = widgets.Button(description='Reset')
        self.clear_button.on_click(lambda b: self.reset_data())
        self.output = widgets.Output()
        self.messages = widgets.Output()
        self.added = widgets.SelectMultiple(options=[])
        self.data = {}
        super().__init__((
            self.experiment_selector, self.results_selector,
            widgets.HBox((self.add_button, self.clear_button), layout=widgets.Layout(width='auto')),
            self.messages,
            widgets.HBox((self.added, )),
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

    def extract_values(self):
        if not self.data:
            self.output_message("Nothing to process")
            return

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
            label = f"{self.experiment_selector.get_selection_label()}.{path} ({len(relevant_experiments)})"
            if label not in self.data:
                self.added.options += (label,)
                self.data[label] = data
                self.output_message(f'Added data for {len(relevant_experiments)} brains')
            else:
                self.output_message(f'Already added, ignored')


class BrainAggregatesHistogramPlot(widgets.VBox):
    def __init__(self, data_dir, raw_data_selector):
        # self.data_selector = DataSelector(data_dir, ResultsSelector(pickle.load(open(f'{data_dir}/../stats.pickle',
        #                                                                              'rb'))))
        self.data_selector = DataSelector(data_dir, raw_data_selector)
        self.bins = widgets.IntSlider(min=10, max=100, value=50, description='Bins: ')
        self.plot_button = widgets.Button(description='Plot')
        self.plot_button.on_click(lambda b: self.plot())
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
            widgets.HBox((self.plot_button, self.ttest_button, self.kstest_button)),
            self.output))
        self.histograms = dict()

    def output_message(self, message):
        self.messages.clear_output()
        with self.messages:
            display(Markdown(message))

    def plot(self):
        values = self.data_selector.extract_values()

        self.output.clear_output()

        with self.output:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for l, d in values.items():
                hist(ax, d, bins=self.bins.value, label=l)
            ax.legend()
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


class SectionHistogramPlotter(object):
    class HeatmapAndPatchButtons(object):
        def __init__(self, experiment_id, base_time, input_dir):
            self.experiment_id = experiment_id
            self.base_time = base_time
            patches_url = f'{input_dir}/{self.experiment_id}/patches-{self.experiment_id}.pdf'
            self.patches = self.create_button(patches_url, "patches", self.build_patches)
            heatmaps_url = f'{input_dir}/{self.experiment_id}/heatmaps-{self.experiment_id}.pdf'
            self.heatmaps = self.create_button(heatmaps_url, "heatmaps", self.build_heatmaps)
            self.panel = widgets.HBox((self.heatmaps, self.patches))

        def build_patches(self):
            pass

        def build_heatmaps(self):
            pass

        def create_button(self, url, display_name, builder):
            def on_click(b):
                if not is_file_up_to_date(url, self.base_time):
                    b.disabled = True
                    b.description = f"Building {display_name}..."
                    builder()
                    self.description = f"Open {display_name}"
                    self.disabled = False

                display(Javascript(f"window.open('{url}');"))

            if os.path.isfile(url):
                button = widgets.Button(description=f"Open {display_name}")
            else:
                button = widgets.Button(description=f"Build {display_name}")

            button.on_click(on_click)
            button.layout.width = 'auto'
            return button

    class AnnotatedImageButton(widgets.Button):
        def __init__(self, experiment_id, full_data, section, base_time, input_dir):
            self.base_time = base_time
            self.experiment_id = experiment_id
            self.directory = f'{input_dir}/{self.experiment_id}'
            self.full_data = full_data
            self.bboxes = pickle.load(open(f'{self.directory}/bboxes.pickle', 'rb'))
            self.annotated_url = f'{input_dir}/{self.experiment_id}/annotated-{self.experiment_id}-{section}.jpg'
            if is_file_up_to_date(self.annotated_url, self.base_time):
                super().__init__(description=f"Open annotated image")
            else:
                super().__init__(description=f"Build annotated image")
            self.section = section
            self.on_click(self.clicked)
            self.layout.width = 'auto'

        def clicked(self, b):
            if not is_file_up_to_date(self.annotated_url, self.base_time):
                self.disabled = True
                self.description = f"Building annotated image..."
                thumb = create_section_image(self.section, self.experiment_id, self.directory, self.full_data,
                                             self.bboxes)
                _, r, _ = detect_brain(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY))
                cv2.imwrite(self.annotated_url, thumb[r.y: r.y + r.h, r.x: r.x + r.w])
                self.description = f"Open annotated image"
                self.disabled = False

            display(Javascript(f"window.open('{self.annotated_url}');"))

    def __init__(self, experiment_id, section, data, base_time, input_dir, structure_tree):
        self.structure_tree = structure_tree
        self.experiment_id = experiment_id
        self.data = data
        self.hist_button = widgets.Button(description=f"Show detailed histograms")
        self.hist_button.layout.width = 'auto'
        self.hist_button.on_click(self.clicked)
        self.output = widgets.Output()
        self.section = section
        display(Markdown('---'))
        if self.section != 'totals':
            self.section = int(section)
            display(widgets.Label(f"Experiment {self.experiment_id}, section {self.section}"))
            self.annotated_image_button = self.AnnotatedImageButton(self.experiment_id, self.data, self.section,
                                                                    base_time, input_dir)
            display(self.output)
            display(widgets.HBox((self.hist_button, self.annotated_image_button)))
            if os.path.isfile(f'{input_dir}/{experiment_id}/thumbnail-{self.experiment_id}-{section}.jpg'):
                thumb = Image.open(f'{input_dir}/{experiment_id}/thumbnail-{self.experiment_id}-{section}.jpg')
                _, rect, _ = detect_brain(np.array(thumb.convert('LA'))[:, :, 0])
                thumb = thumb.crop((*(rect.corners()[0]), *(rect.corners()[1]),))
            else:
                thumb = None
        else:
            display(widgets.Label(f"Experiment {self.experiment_id}, totals"))
            display(self.output)
            display(widgets.HBox((self.hist_button, self.HeatmapAndPatchButtons(experiment_id, base_time, input_dir).panel)))
            thumb = None

        with self.output:
            plot_section_violin_diagram(self.data, structure_tree, thumb)

    def clicked(self, b):
        self.hist_button.disabled = True
        self.hist_button.description = 'Building detailed histograms...'
        # self.output.clear_output()
        with self.output:
            plot_section_histograms(self.data, self.structure_tree)
        self.hist_button.layout.display = 'none'
