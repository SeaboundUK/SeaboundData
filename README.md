# SeaboundData

## Basic setup

Make sure you are using Python 3.9 (or Python 3.x)

    cd [path to SeaboundData]
    python -m venv myenv
    source myenv/bin/activate
    .\myenv\Scripts\activate    # *for windows users* 
    pip install -r requirements.txt

To make the virtual environment kernel available in Jupyter Notebook:

    ipython kernel install --user --name=myenv
        
In the future, always do this before running the scripts or starting Jupyter:

    source myenv/bin/activate
    .\myenv\Scripts\activate    # *for windows users*




To export to html:

    jupyter nbconvert --execute --to html --template lab --no-input DataAnalysis.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove_cell
    

    
    

## A note on data processing logic

* "exclude_metrics" in the config is used to hide data columns in the raw data files ("SensorData_YYYY-MM-DD.csv"). When something is excluded, the program will treat it as if that column did not exist.
* During data processing (when converting raw data files to processed data files i.e. "SensorData_YYYY-MM-DD_Processed.csv"):
    * columns corresponding to those in "exclude_metrics" are removed from dataframe
    * columns that contain only zeros and NaNs are removed from dataframe
* During plotting:
    * before plotting a series, the program only checks if that column name exists in the dataframe. This way the condition for plotting a metric is simplified (e.g. no need to check for zeros or whether a metric is in "exclude_metrics")

