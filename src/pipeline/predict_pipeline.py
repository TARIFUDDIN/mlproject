## sys- Used to handle exceptions and system-related operations.
##A utility function to load saved model and preprocessor objects.
##CustomData class-This class is responsible for structuring input data into a Pandas DataFrame.Takes student-related features as input.Stores them as instance variables
     ##Converts the attributes into a dictionary.
##Creates a Pandas DataFrame, which will later be used as input for the model.
#Defines file paths for saved model and preprocessor.
#Uses load_object to load the saved ML model and preprocessor.
#Transforms the input features using the loaded preprocessor (e.g., scaling, encoding)
#Makes predictions using the trained model.
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class CustomData:
    def __init__(self, gender: str, race_ethnicity: str, parent_education: str, lunch: str, test_prep_score: str, reading_score: int, writing_score: int) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parent_education = parent_education
        self.lunch = lunch
        self.test_prep_score = test_prep_score
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        data = {
            'gender': [self.gender],
            'race_ethnicity': [self.race_ethnicity],
            'parental_level_of_education': [self.parent_education],
            'lunch': [self.lunch],
            'test_preparation_course': [self.test_prep_score],
            'reading_score': [self.reading_score],
            'writing_score': [self.writing_score]
        }
        df = pd.DataFrame(data)
        return df

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Debug: Print columns before transformation
            print("Columns in input DataFrame:", features.columns)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e, sys)