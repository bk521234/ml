import re
import zipfile

import numpy as np
import pandas as pd
import pickle

"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

text_clf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
            ])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))
"""
class SpreadsheetTextExtractor:
    def get_strings_from_spreadsheet_filepath(self, filepath):
        if filepath.endswith('.xlsx'):
            with zipfile.ZipFile('sample_survey_data.xlsx') as myzip:
                with myzip.open('xl/sharedStrings.xml') as myfile:
                    shared_strings_xml = myfile.read()
                    xml_stripped_string = re.sub('<.*?(/)?>', ' ', shared_strings_xml)
            return xml_stripped_string
        elif filepath.endswith('.xls'):
            all_sheets_strings = []
            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                df = xl.parse('Sheet1', header=None, dtype=str)
                sheet_strings = get_dataframe_string_values(df)
                all_sheets_strings += sheet_strings
            return ' '.join(all_sheets_strings)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath, header=None, dtype=str)
            sheet_strings = get_dataframe_string_values(df)
            return sheet_strings
        else:
            raise InvalidFiletype('Could not read in file because it is not .xlsx, .xls, or .csv filetype.')

    def get_dataframe_string_values(self, df):
        string_values_string = ' '.join([
            val for row in df.values
                for val in row 
                if val != 'nan' 
                and val is not np.nan 
                and re.match(r'^\d+?.*?[\.\d+]?$', val) is None
            ])
        return string_values_string

class TrainingDataBuilder:
    def build_dataset(self, filepath, **kwargs):
        with open(filepath, mode='a') as f:
            df = pd.DataFrame({key:val for key, value in kwargs.items()})
            df.to_csv(f)

def save_ml_model_to_pickle(model, pickle_path='finalized_model.sav'):
    pickle.dump(model, open(pickle_path, 'wb'))

def load_ml_mmodel_from_pickle(pickle_path):
    loaded_model = pickle.load(open(pickle_path, 'rb'))
    return loaded_model


if __name__ == "__main__":
    filepath_to_survey_code_dict = []
    for filepath, survey_code in filepath_to_survey_code_dict:
        text_extractor = SpreadsheetTextExtractor()
        text_extractor.get_strings_from_spreadsheet_filepath(filepath)

        builder = TrainingDataBuilder()
        builder.build_dataset('raw_data_text_feature'=text_extractor, 'survey_code_label'=survey_code)

    # train ml model

    # save model to pickle for later use

