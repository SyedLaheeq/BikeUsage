from methods.RandomForest import RandomForestModel

if __name__ == '__main__':
    # Initialize and train the RandomForest model
    print("Initializing Random Forest Model...")
    rf_model = RandomForestModel()

    # Define your file paths for training and validation data
    train_file = 'training_data_fall2024.csv'  # Replace with your actual training data file path
    validation_file = 'training_data_fall2024.csv'  # Replace with your actual validation data file path

    # Train and evaluate the model
    rf_model.train_and_evaluate(train_file, validation_file)
    
    
