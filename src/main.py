from data_loader import download_data
from train_model import train_model
from predict import predict_next_day
from chart import create_chart
from report import create_report


def main():

    print("\n")
    print("========================================")
    print("     STOCK PREDICTION AUTOMATION")
    print("========================================")

    # STEP 1
    print("\n[1/4] DOWNLOADING STOCK DATA")
    print("----------------------------------------")

    download_data()

    # STEP 2
    print("\n[2/4] TRAINING MACHINE-LEARNING MODEL")
    print("----------------------------------------")

    train_model()

    # STEP 3
    print("\n[3/4] GENERATING STOCK PREDICTION")
    print("----------------------------------------")

    predict_next_day()

    # STEP 4
    print("\n[4/4] CREATING STOCK CHART")
    print("----------------------------------------")

    create_chart()
    
    print("\n[5/5] GENERATING PREDICTION REPORT")
    print("-----------------------------------")

    create_report()

    print("\n========================================")
    print("       AUTOMATION COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()