import os

import ipytree
import ipywidgets as widgets
import numpy as np
from IPython.display import display, Markdown
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import matplotlib.pyplot as plt

from aggregate_cell_data import get_struct_aggregates, acronyms
from explorer.explorer_utils import retrieve_nested_path, DataFramesHolder, init_model, predict_crop


class StructureTreeNode(ipytree.Node):
    show_icon = False

    def __init__(self, name, struct_id, children):
        super().__init__(name=name, nodes=children)
        self.struct_id = struct_id
        self.opened = False


class StructureTree(ipytree.Tree):
    def __init__(self, ids, multiple_selection):
        self.ids = ids
        self.mcc = MouseConnectivityCache(manifest_file=f'../mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.structure_tree = self.mcc.get_structure_tree()
        node = self.fill_node('grey')
        while len(node.nodes) < 2:
            node = node.nodes[0]
        super().__init__(nodes=[node], multiple_selection=multiple_selection)

    def fill_node(self, start_node):
        start_struct = self.structure_tree.get_structures_by_acronym([start_node])[0]
        children = [self.fill_node(c['acronym']) for c in self.structure_tree.children(
            [self.structure_tree.get_id_acronym_map()[start_node]])[0]]
        children = [c for c in children if c is not None]

        if not children and start_struct['acronym'] not in self.ids:
            return None

        node = StructureTreeNode(start_struct['acronym'],
                                 start_struct['acronym'],
                                 children)
        if start_struct['acronym'] not in self.ids:
            node.disabled = True

        return node


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
        self.tree = StructureTree(list(data.values())[0].keys(), multiple_selection=False)
        self.template = list(list(data.values())[0].values())[0]
        self.tree.layout.width = "100%"
        self.tree.layout.max_height = "240px"
        self.tree.layout.overflow_y = 'scroll'
        for s in self.selectors[1:]:
            self.enable_selector(s, False)
        self.selectors[0].options = sorted(list(self.template.keys()))
        self.selectors[0].value = self.selectors[0].options[0]
        super().__init__([self.tree, widgets.HBox(self.selectors)])

    @staticmethod
    def enable_selector(selector, mode):
        if mode:
            selector.layout.visibility = 'visible'
        else:
            selector.value = None
            selector.options = []
            selector.layout.visibility = 'hidden'

    def get_available_brains(self):
        return list(self.data.keys())

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
                self.selectors[num + 1].options = sorted(list(val.keys()))
                if old_value in val.keys():
                    self.selectors[num + 1].value = old_value
                else:
                    self.selectors[num + 1].value = self.selectors[num + 1].options[0]
                self.enable_selector(self.selectors[num + 1], True)
                self.selectors[num + 1].disabled = len(self.selectors[num + 1].options) < 2

    def get_selection(self, relevant_experiments):
        path = self.get_selection_path()

        if path is None:
            return None

        return np.array([retrieve_nested_path(self.data[str(e)], path) for e in relevant_experiments])

    def get_selection_label(self):
        path = self.get_selection_path()

        if path is None:
            return None

        return '.'.join(str(p) for p in path)

    def get_selection_path(self):
        if len(self.tree.selected_nodes) != 1:
            return None

        return [self.tree.selected_nodes[0].struct_id] + [s.value for s in self.selectors if s.value is not None]


class RawDataResultsSelector(widgets.VBox):
    def __init__(self, data_dir):
        self.data_frames = DataFramesHolder(data_dir)
        self.available_brains = [e for e in os.listdir(data_dir) if e.isdigit()]
        self.data_template = self.data_frames[self.available_brains[0]]
        self.messages = widgets.Output()
        self.mcc = MouseConnectivityCache(manifest_file=f'mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.structure_tree = self.mcc.get_structure_tree()
        self.aggregates = get_struct_aggregates(set(self.get_column_options(self.data_template['structure_id'])))

        self.structure_selector = StructureTree(ids=[s['acronym'] for s in self.structure_tree.get_structures_by_id(
            list(self.aggregates.keys()))],
                                                multiple_selection=True)
        self.parameter_selector = widgets.Dropdown(description="Parameter", options=['coverage', 'area', 'perimeter'])

        self.change_handler = None
        super().__init__((widgets.HBox([self.structure_selector, self.parameter_selector]), self.messages))

    def get_available_brains(self):
        return self.available_brains

    def reset(self):
        for v in self.filter.values():
            v.value = []

    @staticmethod
    def get_column_options(df):
        return sorted(df.unique().tolist())

    def get_selected_structs(self):
        selected_acronyms = [n.struct_id for n in self.structure_selector.selected_nodes]
        id_sets = [self.aggregates[self.structure_tree.get_id_acronym_map()[s]] for s in selected_acronyms]
        ids = set.union(*id_sets)
        return [acronyms[i] for i in ids]

    def process_data_frame(self, df, structs):
        frame = df[df.structure_id.isin(structs)]
        d = frame[self.parameter_selector.value].to_numpy()
        return d

    def get_selection(self, relevant_experiments):
        structs = [self.structure_tree.get_id_acronym_map()[s] for s in self.get_selected_structs()]
        return np.concatenate([self.process_data_frame(self.data_frames[e], structs) for e in relevant_experiments])

    def get_selection_label(self):
        label = '.'.join([n.struct_id for n in self.structure_selector.selected_nodes])
        if label == '':
            label = '<any>'
        label = f'{label}.{self.parameter_selector.value}'

        return label

    def on_selection_change(self, handler):
        self.change_handler = handler


class CropPredictor(widgets.VBox):
    def __init__(self):
        self.upload = widgets.FileUpload(multiple=True)
        self.go_button = widgets.Button(description='Go')
        self.go_button.on_click(self.do_predict)
        self.output = widgets.Output()
        self.cell_model = None
        super().__init__((widgets.HBox((self.upload, self.go_button,)), self.output,))

    def do_predict(self, b):
        import cv2

        if self.cell_model is None:
            self.cell_model = init_model('output/new_cells/model_0324999.pth', 'cpu', 0.5)

        with self.output:
            fig, ax = plt.subplots(1, len(self.upload.value), figsize=(len(self.upload.value) * 10, 10))
            for i, (name, file_info) in enumerate(self.upload.value.items()):
                crop = cv2.imdecode(np.frombuffer(file_info['content'], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                ax[i].set_title(name)
                ax[i].imshow(cv2.cvtColor(predict_crop(crop, self.cell_model), cv2.COLOR_BGR2RGB))

            plt.show()



