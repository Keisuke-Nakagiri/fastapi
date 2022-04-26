from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
import pickle

app = FastAPI()
clf = pickle.load(open("model.pickle", "rb"))

@app.post("/prediction/")
async def prediction(files: list[UploadFile] = File(...)):
    predicted_names = []
    for file in files:
        _csvdata = StringIO(str(file.file.read(), 'utf-8'))
        csvdata = pd.read_csv(_csvdata).values.tolist()
        answer = clf.predict(csvdata)
        if answer == 0:
            predicted_names.append("setosa")
        elif answer == 1:
            predicted_names.append("versicolor")
        elif answer == 2:
            predicted_names.append("virginica")
        else:
            predicted_names.append("None")
    return {
        "filenames": [file.filename for file in files],
        "predicted_names": [name for name in predicted_names]
    }