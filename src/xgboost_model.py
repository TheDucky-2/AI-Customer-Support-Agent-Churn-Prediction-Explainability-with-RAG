from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xgboost as xgb

class Model:

  def __init__(self, model):
    self.model = model
    self.dataframe = None

  def load_data(self, file):

    try:

      self.dataframe = pd.read_csv(file)

    except FileNotFoundError:

      print(f"File not found; {file}")

    return self.dataframe

  def set_data(self, column_label = 'Churn', drop_columns = ['customerID']):

    self.X = self.dataframe.drop(columns = drop_columns+[column_label], axis =1)
    self.y = self.dataframe[column_label]

    return self.X, self.y

  def split_data(self, test_size= 0.2):

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=y, random_state=3)

    return self.X_train, self.X_test, self.y_train, self.y_test

  def separate_scale_notscale(self):

    #### This function separates columns that have to be scaled, from columns that are either text or one-hot encoded.

    to_scale_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    ### For X_train

    self.X_train_scale = self.X_train[to_scale_columns]
    self.X_train_notscale = self.X_train.drop(columns=to_scale_columns)

    ### For X_test

    self.X_test_scale = self.X_test[to_scale_columns]
    self.X_test_notscale = self.X_test.drop(columns = to_scale_columns)

    return self.X_train_scale, self.X_train_notscale, self.X_test_scale, self.X_test_notscale

  def scale_data(self):
    to_scale_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    #COLUMNS TO SCALE

    self.X_train_scaled = self.X_train_scale.copy()   # CREATING THE COPIES TO SCALE TO prevent in-place modification
    self.X_test_scaled = self.X_test_scale.copy()     # repeating same for X_test

    scaler = StandardScaler()
    # SCALING OUR DATA TO USE FOR TRAINING OUR MODEL.

    self.X_train_scaled = pd.DataFrame(scaler.fit_transform(self.X_train_scaled), columns = to_scale_columns, index=self.X_train.index)
    self.X_test_scaled = pd.DataFrame(scaler.transform(self.X_test_scaled), columns = to_scale_columns, index=self.X_test.index)

    return self.X_train_scaled, self.X_test_scaled

  def joining_scale_notscale(self):

    self.X_train_final = pd.concat([self.X_train_scaled, self.X_train_notscale], axis=1)
    self.X_test_final = pd.concat([self.X_test_scaled, self.X_test_notscale], axis=1)

    return self.X_train_final, self.X_test_final

  def export_data(self):

    processed_data = pd.concat([self.X_train_final, self.y], axis=1)
    file = processed_data.to_csv("Processed_Data.csv", index=False)

    return file

  def train_model(self):

    self.model.fit(self.X_train_final, self.y_train)      # Training the model

  def make_predictions(self):

    self.y_pred = self.model.predict(self.X_test_final)   # Making a prediction

    return self.y_pred

  def track_performance(self):

    accuracy = accuracy_score(self.y_test, self.y_pred)
    print(f"Accuracy-score: {accuracy:.2f}")
    f1 = f1_score(self.y_test, self.y_pred)
    print(f"F1-score: {f1:.2f}")
    recall = recall_score(self.y_test, self.y_pred)
    print(f"Recall-score: {recall:.2f}")

    return accuracy, f1, recall


if __name__ == '__main__':

  model = Model(xgb.XGBClassifier())
  model.load_data("Preprocessed_Data.csv")
  model.set_data()
  model.split_data()
  model.separate_scale_notscale()
  model.scale_data()
  model.joining_scale_notscale()
  model.export_data()
  model.train_model()
  model.make_predictions()
  model.track_performance()








































