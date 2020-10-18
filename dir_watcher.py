import logging
import os
import sys
import time
from abc import abstractmethod, ABC


class DirWatcher(ABC):
    def __init__(self, input_dir, intermediate_dir, results_dir, name):
        self.__results_dir__ = results_dir
        self.__intermediate_dir__ = intermediate_dir
        self.__input_dir__ = input_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel('INFO')
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%a %Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        for d in [intermediate_dir, results_dir]:
            os.makedirs(d, exist_ok=True)

    def extract_item(self):
        while True:
            items = sorted([item for item in os.listdir(self.__input_dir__)
                            if os.path.isdir(os.path.join(self.__input_dir__, item))])

            if not items:
                return None, None

            try:
                item = items[0]
                directory = os.path.join(self.__intermediate_dir__, item)
                os.replace(os.path.join(self.__input_dir__, item), directory)
                return item, directory
            except OSError as e:
                continue

    def handle_item(self, item, directory):
        retval = None

        try:
            retval = self.process_item(item, directory)
        except Exception as e:
            self.on_process_error(item)
            raise e

        os.replace(os.path.join(self.__intermediate_dir__, item), os.path.join(self.__results_dir__, item))
        return retval

    def run_until_empty(self):
        res_list = list()
        while True:
            item, directory = self.extract_item()
            if item is None:
                break
            res_list.append(self.handle_item(item, directory))

        self.reduce(res_list, self.__results_dir__)

    def run_until_count(self, results_count):
        res_list = list()
        while True:
            results = [result for result in os.listdir(self.__results_dir__)
                       if os.path.isdir(os.path.join(self.__results_dir__, result))]
            if len(results) >= results_count:
                break

            item, directory = self.extract_item()
            if item is not None:
                res_list.append(self.handle_item(item, directory))
            else:
                time.sleep(1)

        self.reduce(res_list, self.__results_dir__)

    def run_until_stopped(self):
        while True:
            item, directory = self.extract_item()
            if item is not None:
                self.handle_item(item, directory)
            else:
                time.sleep(1)

    def on_process_error(self, item):
        os.replace(os.path.join(self.__intermediate_dir__, item), os.path.join(self.__input_dir__, item))

    @abstractmethod
    def process_item(self, item, directory):
        pass

    def reduce(self, results, output_dir):
        pass
