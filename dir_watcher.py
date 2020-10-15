import logging
import os
import sys
import time
from abc import abstractmethod, ABC


class DirWatcher(ABC):
    def __init__(self, input_dir, intermediate_dir, results_dir, error_dir, name):
        self.error_dir = error_dir
        self.results_dir = results_dir
        self.intermediate_dir = intermediate_dir
        self.input_dir = input_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel('INFO')
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%a %Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        for d in [intermediate_dir, results_dir, error_dir]:
            os.makedirs(d, exist_ok=True)

    def extract_item(self):
        items = sorted([item for item in os.listdir(self.input_dir)
                        if os.path.isdir(os.path.join(self.input_dir, item))])
        for item in items:
            try:
                os.replace(os.path.join(self.input_dir, item), os.path.join(self.intermediate_dir, item))
                return item
            except FileNotFoundError:
                continue

        return None

    def handle_item(self, item):
        try:
            self.process_item(item)
        except Exception as e:
            os.replace(os.path.join(self.intermediate_dir, item), os.path.join(self.error_dir, item))
            raise e

        os.replace(os.path.join(self.intermediate_dir, item), os.path.join(self.results_dir, item))


    def run_until_empty(self):
        while True:
            item = self.extract_item()
            if item is None:
                break
            self.handle_item(item)

    def run_until_count(self, results_count):
        while True:
            results = [result for result in os.listdir(self.results_dir)
                              if os.path.isdir(os.path.join(self.results_dir, result))]
            if len(results) >= results_count:
                break

            item = self.extract_item()
            if item is not None:
                self.handle_item(item)
            else:
                time.sleep(1)

    @abstractmethod
    def process_item(self, item):
        pass

