import pandas as pd

class DataCleaner:

  def __init__(self, file):
    self.file = file
    self.dataframe = None

  def load_data(self):
    try:
      self.dataframe = pd.read_csv(self.file)                                                            # Loading csv into pandas and creating a dataframe.

    except FileNotFoundError:
      print(f"File:{self.file} not found!")

    return self.dataframe

  def check_for_nulls(self):

    print(f"Null values per column:\n\n{self.dataframe.isnull().sum()}")                                # Checking for all null-values

    return self.dataframe

  def modify_datatypes(self):
    self.dataframe['TotalCharges'] = pd.to_numeric(self.dataframe['TotalCharges'], errors='coerce')     # Changing the datatype from 'object' to 'int'
    self.dataframe['Churn'] = self.dataframe['Churn'].map({"Yes":1, "No":0})                            # Label-Encoding

    return self.dataframe

  def get_new_file(self):

    new_file = self.dataframe.to_csv("Clean_Data.csv", index=False)                                      # exporting csv of clean data

    return new_file

class DataPreProcessor:

  def __init__(self, file):
    self.file = file
    self.dataframe = None

  def load_csv(self):
    try:
      self.dataframe = pd.read_csv(self.file)

    except FileNotFoundError:
      print(f"File: {self.file} not found!")

    return self.dataframe

  def pre_process(self):
    # Label-Encoding columns that have only 2 categories- 0 and 1.
    self.dataframe['gender'] = self.dataframe['gender'].map({"Female": 0, "Male": 1})
    self.dataframe['Partner'] = self.dataframe['Partner'].map({"No": 0, "Yes": 1})
    self.dataframe['Dependents'] = self.dataframe['Dependents'].map({"No": 0, "Yes": 1})
    self.dataframe["PhoneService"] = self.dataframe['PhoneService'].map({"No": 0, "Yes": 1})
    self.dataframe['PaperlessBilling'] = self.dataframe['PaperlessBilling'].map({"No": 0, "Yes": 1})

  # One-hot encoding columns that have more than 2 categories.

    self.dataframe = pd.get_dummies(self.dataframe, columns = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'],dtype=int)

    return self.dataframe

  def export_csv(self):

    preprocessed_data = self.dataframe.to_csv("Preprocessed_Data.csv", index=False)

    return preprocessed_data

if __name__ == '__main__':

  data_cleaner = DataCleaner("Telco_Customer_Churn_Raw_Data.csv")
  data_cleaner.load_data()
  data_cleaner.modify_datatypes()
  data_cleaner.get_new_file()
  data_pre_processor = DataPreProcessor("Clean_Data.csv")
  data_pre_processor.load_csv()
  data_pre_processor.pre_process()
  data_pre_processor.export_csv()








