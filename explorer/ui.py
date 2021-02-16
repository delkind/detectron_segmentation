import os

import ipywidgets as widgets
import numpy as np
from IPython.display import display, Markdown
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from explorer.explorer_utils import retrieve_nested_path, DataFramesHolder


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
                self.selection_changed({'name': 'value'}, 'strain')
            elif col == 'strain':
                oldval = self.filter['transgenic_line'].value
                self.filter['transgenic_line'].options = self.get_column_options(
                    self.experiments[self.experiments.gender.isin(self.get_filter_value('gender'))
                                     & self.experiments.strain.isin(selection)].transgenic_line)
                oldval = [v for v in oldval if v in self.filter['transgenic_line'].options]
                self.filter['transgenic_line'].value = oldval
                self.selection_changed({'name': 'value'}, 'transgenic_line')
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
                self.selectors[num + 1].disabled = len(self.selectors[num + 1].options) < 2

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
                                          options=['coverage', 'area', 'perimeter']),
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
