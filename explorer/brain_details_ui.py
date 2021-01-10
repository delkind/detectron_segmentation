import os
import pickle

import PIL.Image as Image
import cv2
import ipywidgets as widgets
import numpy as np
from IPython.display import display, Markdown, Javascript
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from annotate_cell_data import create_section_image
from explorer.explorer_utils import is_file_up_to_date, plot_section_violin_diagram, plot_section_histograms
from explorer.ui import ExperimentsSelector
from localize_brain import detect_brain


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


class BrainDetailsSelector(widgets.VBox):
    def __init__(self, input_dir):
        self.mcc = MouseConnectivityCache(manifest_file=f'../mouse_connectivity/mouse_connectivity_manifest.json',
                                          resolution=25)
        self.input_dir = input_dir
        self.experiments_selector = ExperimentsSelector()
        self.histograms_output = widgets.Output()
        self.section_combo = widgets.Dropdown(options=[],
                                         value=None,
                                         description='Choose section:')

        self.refresh_button = widgets.Button(description='Refresh experiment list')
        self.refresh_button.layout.width = 'auto'
        self.refresh_button.on_click(lambda b: self.refresh)

        self.display_button = widgets.Button(description='Go!')
        self.display_button.on_click(self.display_experiment)

        self.clear_button = widgets.Button(description='Clear plots')
        self.clear_button.on_click(lambda b: self.histograms_output.clear_output())

        self.experiments_selector.on_selection_change(lambda col, change: self.refresh())
        super().__init__((
            widgets.HBox((self.experiments_selector, self.refresh_button)),
            widgets.HBox((self.section_combo, self.display_button, self.clear_button)),
            self.histograms_output
        ))

        self.refresh()
        self.on_select_experiment([])

    def refresh(self):
        self.experiments_selector.set_available_brains(set([int(e) for e in os.listdir(self.input_dir) if e.isdigit()]))
        experiment_ids = list(set(self.experiments_selector.get_selection()))
        self.on_select_experiment(experiment_ids)

    def on_select_experiment(self, experiment_ids):
        if len(experiment_ids) != 1:
            self.section_combo.options = []
            self.section_combo.value = None
            self.section_combo.disabled = True
        else:
            experiment_id = experiment_ids[0]
            directory = f"{self.input_dir}/{experiment_id}"
            with open(f'{directory}/bboxes.pickle', "rb") as f:
                bboxes = pickle.load(f)
            bboxes = {k: v for k, v in bboxes.items() if v}
            self.section_combo.options = ['all', 'totals'] + list(str(i) for i in bboxes.keys())
            self.section_combo.value = self.section_combo.options[0]
            self.section_combo.disabled = False

    def display_experiment(self, b):
        experiment_id = list(self.experiments_selector.get_selection())[0]
        section = self.section_combo.value
        if not experiment_id or not section:
            return
        directory = f"{self.input_dir}/{experiment_id}"
        full_data = pd.read_parquet(f"{directory}/celldata-{experiment_id}.parquet")
        full_data_mtime = os.path.getmtime(f"{directory}/celldata-{experiment_id}.parquet")

        with self.histograms_output:
            if section == 'all':
                SectionHistogramPlotter(experiment_id, 'totals', full_data, full_data_mtime, self.input_dir,
                                        self.mcc.get_structure_tree())
                for section in sorted(full_data.section.unique().tolist()):
                    section_data = full_data[full_data.section == int(section)]
                    SectionHistogramPlotter(experiment_id, section, section_data, full_data_mtime, self.input_dir,
                                            self.mcc.get_structure_tree())
            else:
                if section != 'totals':
                    full_data = full_data[full_data.section == int(section)]
                SectionHistogramPlotter(experiment_id, section, full_data, full_data_mtime, self.input_dir,
                                        self.mcc.get_structure_tree())


# plot = SectionHistogramPlotter('brain', full_data)
