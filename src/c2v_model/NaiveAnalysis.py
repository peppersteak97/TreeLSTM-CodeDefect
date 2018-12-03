from Common import Common
from Config import Config
from NaiveExtractor import NaiveExtractor
from Model import Model

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class PathAnalysis(object):
    def __init__(self, model_path="models/java14_model/saved_model_iter8.release", config=None):
        self._config = config if config is not None else Config.get_default_config(load_path=model_path)
        self._path_extractor = NaiveExtractor(self._config)
        self._model = Model(self._config)

    @property
    def get_model(self):
        return self._model

    def predict_file(self, path_to_file):
        try:
            predict_lines, hash_to_string_dict = self._path_extractor.extract_paths(path_to_file)
        except ValueError as e:
            print(e)
            return None
        results = self._model.predict(predict_lines)
        prediction_results = Common.parse_results(results, hash_to_string_dict, topk=SHOW_TOP_CONTEXTS)
        return prediction_results


if __name__ == "__main__":
    path_analysis = PathAnalysis()
    print(path_analysis.predict_file("test.java"))
