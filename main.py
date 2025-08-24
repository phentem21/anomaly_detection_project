import argparse
from src.data_loader import load_data
from src.models import AnomalyModel
from src.attribution import explain_anomalies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    print("Loading data...")
    X, y = load_data("data/my_dataset.csv")

    print("Initializing model...")
    model = AnomalyModel()

    print("Training model...")
    model.train(X, epochs=args.epochs)

    print("Evaluating model...")
    scores = model.evaluate(X)

    print("Attributing anomalies...")
    explain_anomalies(scores)

if __name__ == "__main__":
    main()
