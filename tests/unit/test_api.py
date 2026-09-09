from src.churn.api import main as api_main
from src.churn.api.main import ClienteInput


class ModeloFalso:
    def predict(self, df):
        return [0]

    def predict_proba(self, df):
        return [[0.9, 0.1]]


def test_predecir_logica(monkeypatch):
    monkeypatch.setattr(api_main, "modelo", ModeloFalso())

    cliente = ClienteInput(
        **{
            "Call  Failure": 2,
            "Complains": 0,
            "Subscription  Length": 10,
            "Charge  Amount": 1,
            "Seconds of Use": 500,
            "Frequency of use": 20,
            "Frequency of SMS": 5,
            "Distinct Called Numbers": 15,
            "Age Group": 2,
            "Tariff Plan": 1,
            "Status": 1,
            "Age": 30,
            "Customer Value": 200.0,
        }
    )

    resultado = api_main.predecir(cliente)

    assert resultado.churn_predicho == 0
    assert 0.0 <= resultado.probabilidad_churn <= 1.0


def test_salud_reporta_modelo_cargado(monkeypatch):
    monkeypatch.setattr(api_main, "modelo", ModeloFalso())
    resultado = api_main.salud()
    assert resultado == {"status": "ok", "modelo_cargado": True}