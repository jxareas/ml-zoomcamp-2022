import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

model_reference = bentoml.xgboost.get("credit_risk_model:latest")
dict_vectorizer = model_reference.custom_objects.get('dictVectorizer')

model_runner = model_reference.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])


class CreditApplication(BaseModel):
    seniority: int = 3
    home: str = 'owner'
    time: int = 36
    age: int = 26
    marital: str = 'single'
    records: str = 'no'
    job: str = 'freelance'
    expenses: int = 35
    income: float = 0.0
    assets: float = 6_000.0
    debt: float = 3_000.0
    amount: int = 800
    price: int = 1_000


@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON())
def classify(credit_application):
    application_data = credit_application.dict()
    vector = dict_vectorizer.transform(application_data)
    prediction = model_runner.predict.run(vector)

    result = prediction[0]
    if result > 0.5:
        return {
            "status": "Declined"
        }
    elif result > 0.25:
        return {
            "status": "Perhaps"
        }
    else:
        return {
            "status": "Approved"
        }
