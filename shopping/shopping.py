import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    month_dict = {
        0: "Jan",
        1: "Feb",
        2: "Mar",
        3: "Apr",
        4: "May",
        5: "June",
        6: "Jul",
        7: "Aug",
        8: "Sep",
        9: "Oct",
        10: "Nov",
        11: "Dec"
    }
    data = []
    labels = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[17] == "FALSE" or row[17] == "TRUE":
                data.append([int(row[0]), float(row[1]), int(row[2]), float(row[3]), int(row[4]), [
                    float(cell) for cell in row[5:10]], list(month_dict.keys())[list(month_dict.values()).index(row[10])], [
                        int(cell) for cell in row[11:15]], 1 if row[15] == "Returning_Visitor" else 0, 1 if row[16] == "TRUE" else 0])
            labels.append(1 if row[17] == "TRUE" else 0)
    evidence = []
    for item in data:
        for value in item:
            if type(value) is list:
                temp = value.copy()
                index = item.index(value)
                item.remove(value)
                for i in temp:
                    item.insert(index, i)
                    index += 1
        evidence.append(item)
    return (evidence, labels)

    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # raise NotImplementedError


def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    fitted = model.fit(evidence, labels)
    return fitted

    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # raise NotImplementedError


def evaluate(labels, predictions):

    positive = 0
    negative = 0
    sum_positive = 0
    sum_negative = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            sum_positive += 1
            if labels[i] == predictions[i]:
                positive += 1
        elif labels[i] == 0:
            sum_negative += 1
            if labels[i] == predictions[i]:
                negative += 1
        else:
            continue
    sensitivity = float(positive/sum_positive)
    specificity = float(negative/sum_negative)
    return (sensitivity, specificity)

    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # raise NotImplementedError


if __name__ == "__main__":
    main()
