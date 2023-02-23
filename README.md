# SeaboundData

Make sure you are using Python 3.9 (or Python 3.x)

    cd [path to SeaboundData]
    python3.10 -m venv myenv
    source myenv/bin/activate
    .\myenv\Scripts\activate    # *for windows users* 
    pip install -r requirements.txt


In the future, always do this before running the scripts:

    source myenv/bin/activate
    .\myenv\Scripts\activate    # *for windows users*

To export to html:

    jupyter nbconvert --execute --to html --template lab --no-input DataAnalysis.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags remove_cell