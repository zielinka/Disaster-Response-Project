import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
     '''
    - Load messages and categories datasets from csv files 
    - Merge the messages and categories datasets based on id
            
    '''
        
    #load messages
    messages = pd.read_csv(messages_filepath, encoding='utf-8')
    #load categories
    categories = pd.read_csv(categories_filepath, encoding='utf-8')
    # merge datasets
    df = messages.merge(categories, how='outer', on='id') 
    return df

  
def clean_data(df):
    
    '''
    Function to clean data:
       - Split values in categories column based on ; character 
       - Create column names for created categories data
       - Rename categories columns  with new column names
       - Drop duplicates
       - Replace 'related'category values from 2 to 1
    '''
    
    #create a dataframe of the 36 individual category columns
    categories= pd.concat([df[['id']],df['categories'].str.split(';', expand=True)],axis=1)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories.
    category_colnames =row[lambda x: x.iloc[1:]].index
    category_colnames = category_colnames.str[:-2]
    categories.columns.values[1:]=category_colnames
    categories = categories.reset_index(drop=True)
    
    for column in categories.columns[1:]:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column],errors='coerce')
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, how='outer', on='id')
    # drop duplicates
    df = df.drop_duplicates()
    # replace values from 2 to 1 for 'related' column
    df.related.replace(2,1,inplace=True)
    
    return df
   
#save the clean dataset into an sqlite database
def save_data(df, database_filename):
    
    '''
    Save cleaned dataset as sqlite database 
    
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('process_data', engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()