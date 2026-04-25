import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import sys
import PieceRecognitionModel
import EndToEndModel
import comparisonModel

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No arguments provided. -h for options")
    elif sys.argv[1] == "-h":
        print("Options:")
        print("-train: Train a model")
        print("-test: Test a model")
        print("-unseen: Test on unseen data")
        print("-predict -imgPath: Predict on a single image using full vision system. [Note ensure vision system is trained first]")
        print("Types of models:")
        print("-pipeline: Pipeline Model")
        print("-end: End-to-End Model")
        print("-combiner: Combiner Model")
        print("-full: Train complete vision system")
        print("Example usage: python ChessVisionSystem.py -train -pipeline")

    elif sys.argv[1] == "-train":
        if len(sys.argv) < 3:
            print("No model type provided. -h for options")
        elif sys.argv[2] == "-pipeline" or sys.argv[2] == "-end" or sys.argv[2] =="-combiner" or sys.argv[2] == "-full":
            option = input("0: Single training run or 1: multiple runs and keep the best model")
            if option == "0":
                print("Training Model...")
                if sys.argv[2] == "-pipeline":
                    PieceRecognitionModel.SingleTrainingRun()
                elif sys.argv[2] == "-end":
                    EndToEndModel.SingleTrainingRun()
                elif sys.argv[2] == "-combiner":
                    comparisonModel.SingleTrainingRun()
                elif sys.argv[2] == "-full":
                    PieceRecognitionModel.SingleTrainingRun()
                    EndToEndModel.SingleTrainingRun()
                    comparisonModel.SingleTrainingRun()
                print("Training Complete!")
            elif option == "1":
                count = None
                while True:
                    options = input("Enter number of training runs: ")
                    try:
                        count = int(options)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid integer for the number of training runs.")
                if count is None:
                    print("No valid number of training runs provided. Exiting.")
                    sys.exit(1)
                print("Training Model...")
                if sys.argv[2] == "-pipeline":
                    PieceRecognitionModel.MultipleTrainingRuns(count)
                elif sys.argv[2] == "-end":
                    EndToEndModel.MultipleTrainingRuns(count)
                elif sys.argv[2] == "-combiner":
                    print("Combiner model does not support multiple runs, performing single training run instead.")
                    comparisonModel.SingleTrainingRun()
                elif sys.argv[2] == "-full":
                    PieceRecognitionModel.MultipleTrainingRuns(count)
                    EndToEndModel.MultipleTrainingRuns(count)
                    comparisonModel.SingleTrainingRun()
                print("Training Complete!")
        else:
            print("Invalid model type provided. -h for options")
    elif sys.argv[1] == "-test":
        print("Testing Model...")
        if len(sys.argv) < 3:
            print("No model type provided. -h for options")
        elif sys.argv[2] == "-pipeline" or sys.argv[2] == "-end" or sys.argv[2] =="-combiner" or sys.argv[2] == "-full":
            if sys.argv[2] == "-pipeline":
                PieceRecognitionModel.getValidationAccuracy()
            elif sys.argv[2] == "-end":
                EndToEndModel.getValidationAccuracy()
            elif sys.argv[2] == "-full" or sys.argv[2] == "-combiner":
                if sys.argv[2] == "-full":
                    print("Full vision system testing is equivalent to testing the combiner model, as it evaluates the performance of the entire system.")
                comparisonModel.evaluateCombinerOnValidation()
            print("Testing Complete!")
        else:
            print("Invalid model type provided. -h for options")
    
    elif sys.argv[1] == "-unseen":
        print("Testing on unseen data...")
        if len(sys.argv) < 3:
            print("No model type provided. -h for options")
        elif sys.argv[2] == "-pipeline" or sys.argv[2] == "-end" or sys.argv[2] =="-combiner" or sys.argv[2] == "-full":
            if sys.argv[2] == "-pipeline":
                PieceRecognitionModel.UnseenTest()
            elif sys.argv[2] == "-end":
                EndToEndModel.UnseenTest()
            elif sys.argv[2] == "-full" or sys.argv[2] == "-combiner":
                print("Not supported")
            print("Testing Complete!")
        else:
            print("Invalid model type provided. -h for options")
    
    elif sys.argv[1] == "-predict":
        if len(sys.argv) < 4:
            print("No image path provided. -h for options")
        else:
            try:
                comparisonModel.prediction(sys.argv[3])
            except Exception as e:
                print(f"Error during prediction: {e}")
    else:
        print("Invalid option provided. -h for options")


