import argparse
import joblib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='timetable_line_classifier.joblib')
    parser.add_argument('--text', required=True)
    args = parser.parse_args()

    model = joblib.load(args.model)
    lines = [x.strip() for x in args.text.splitlines() if x.strip()]
    probs = model.predict_proba(lines)
    labels = model.predict(lines)
    classes = list(model.classes_)

    for line, label, prob_row in zip(lines, labels, probs):
        conf = max(prob_row)
        print(f'{label}\t{conf:.3f}\t{line}')

if __name__ == '__main__':
    main()
