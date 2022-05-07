from dobble_master_race.transformers import HuMomentsColorsHistogram
from dobble_master_race.toolkit import get_data_set

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_resources = os.path.join(os.path.dirname(curr_dir), "data/train")
    (X_train, Y_train), (X_test, Y_test) = get_data_set(path_to_resources)

    transformer = HuMomentsColorsHistogram(6, 10, {}, {})
    scaler = StandardScaler()
    classifier = RandomForestClassifier()

    pipeline = make_pipeline(transformer, scaler, classifier)
    pipeline.fit(X_train, Y_train)

    models_dir = os.path.join(curr_dir, "models_save")
    kmeans_path = os.path.join(models_dir, "kmeans_save.sav")
    scaler_path = os.path.join(models_dir, "scaler_save.sav")
    classifier_path = os.path.join(models_dir, "classifier_save.sav")

    with open(kmeans_path, "wb") as kmeans_file:
        pickle.dump(transformer.color_histogram.clustering, kmeans_file)
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open(classifier_path, "wb") as classifier_file:
        pickle.dump(classifier, classifier_file)
