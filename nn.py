import argparse
import zipfile
import sklearn.metrics
import pandas as pd


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}


def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:

    # doesn't train anything; just predicts 1 for all of dev set
    dev_predictions = dev_data.copy()
    dev_predictions[emotions] = 1
    return dev_predictions


if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="2018-E-c-En-train.txt")
    parser.add_argument("test", nargs='?', default="2018-E-c-En-dev.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_data, test_data)

    # saves predictions and creates submission zip file
    test_predictions.to_csv("E-C_en_pred.txt", sep="\t", index=False)
    with zipfile.ZipFile('submission.zip', mode='w') as submission_zip:
        submission_zip.write("E-C_en_pred.txt")

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))
