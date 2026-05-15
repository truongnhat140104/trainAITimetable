import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='line_train.csv')
    parser.add_argument('--val_file', default='line_val.csv')
    parser.add_argument('--output', default='timetable_line_classifier.joblib')
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    val_df = pd.read_csv(args.val_file)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(2, 5),
            min_df=1
        )),
        ('clf', LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            solver='lbfgs'
        ))
    ])

    model.fit(train_df['text'].astype(str), train_df['label'].astype(str))
    pred = model.predict(val_df['text'].astype(str))

    print(classification_report(val_df['label'].astype(str), pred, zero_division=0))
    print('Confusion matrix labels:', list(model.named_steps['clf'].classes_))
    print(confusion_matrix(val_df['label'].astype(str), pred, labels=list(model.named_steps['clf'].classes_)))

    joblib.dump(model, args.output)
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
