import os
from prefect import flow, task
import mlflow

@task
def calcular_suma():
    return sum(range(1, 6))

@flow(name="prueba", log_prints=True)
def mi_flow():
    resultado = calcular_suma()
    print(f"Suma de 1 a 5 = {resultado}")
    
    mlflow.set_experiment("prueba-validacion")
    with mlflow.start_run(run_name="prueba-mlflow"):
        mlflow.log_param("n", 5)
        mlflow.log_metric("suma", resultado)

if __name__ == "__main__":
    mi_flow()
  
