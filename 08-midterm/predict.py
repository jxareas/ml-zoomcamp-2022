import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field

model_ref = bentoml.xgboost.get("xgboost_phone_predictor:latest")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("phone_price_predictor", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def predict(application_data):
    print(application_data)
    vector = dv.transform(application_data)
    print(vector)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)

    log_result = prediction[0]

    return np.expm1(log_result)
