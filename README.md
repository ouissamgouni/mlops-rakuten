<h1 align="center">MLops: Rakuten prediction system for product categorisation</h1>

Short description of the project

## Design

![alt text](docs/mlops_design.jpg)

## Prerequisites

### Infra
Install:
* [Docker](https://docs.docker.com/get-docker/) and docker-compose
* [Minikube](https://minikube.sigs.k8s.io/docs/start/) or local Kubernetes cluster
* [Helm](https://helm.sh/docs/intro/install/)
``` bash
# Launch your local Kubernetes cluster
minikube start
```
Then :
* [Postgresql](https://phoenixnap.com/kb/postgresql-kubernetes)
* [Mlflow](https://github.com/bitnami/charts/tree/main/bitnami/mlflow)
* [Prometheus & Grafana](https://medium.com/@brightband/deploying-prometheus-operator-to-a-kubernetes-cluster-c2378038c79b)
* [Minio](https://medium.com/@kapincev/easy-guide-setting-up-minio-with-microk8s-kubernetes-321048d901ac)
* [Flyte](https://github.com/davidmirror-ops/flyte-the-hard-way/blob/main/docs/on-premises/single-node/002-single-node-onprem-install.md)

* [Docker registry](https://hub.docker.com/r/ogouni604/mlops-rakuten) 
### Tooling
* [Taskfile](https://taskfile.dev/installation/)

## Usage

### Run the Data/ML workflow

``` bash
task workflow:run {version: workflow docker tag}
```
Then returns Mlflow artifacts URIs

### Download a Model version
``` bash
task api:pull-model {version: Mlflow artifacts URI}
```
### Build and deploy the inference service
``` bash
task api:build push deploy {version: application version}
```

### Update monitoring stack
For operations like updating metrics scraping parameters, alerting rules, grafana dashboard, etc.
``` bash
task api:update-monitoring
```


## References
* [Configure Alert Manager with Slack Notifications on Kubernetes Cluster](https://medium.com/@phil16terpasetheo/configure-alert-manager-with-slack-notifications-on-kubernetes-cluster-helm-kube-prometheus-stack-112878c35f26)
* [Quick registration and authentication system for your FastAPI project](https://github.com/fastapi-users/fastapi-users/tree/master/examples/sqlalchemy)


