from flytekit import task, workflow
@task
def load_data() -> str:
    return "load data"
@task
def preprocess() -> str:
    return "preprocess"
    
@task
def train() -> str:
    return "train"
@task
def validate() -> str:
    return "validate"
@task
def release() -> str:
    # pushing the model to mlflow
    return "release"
@workflow
def my_wf() -> str:
    res = load_data()
    res = preprocess()
    res = train()
    res = validate()
    res = release()
    return res
if __name__ == "__main__":
    print(f"Running my_wf() {my_wf()}")