<h1 align="center">MLops: Rakuten prediction system for product categorisation</h1>

Short description of the project

## Design

![alt text](docs/mlops_design.jpg)

## Installation

### Infra

Prerequisites:

* [Docker](https://docs.docker.com/get-docker/) and docker-compose
* [Minikube](https://minikube.sigs.k8s.io/docs/start/) or local Kubernetes cluster
* [Helm](https://helm.sh/docs/intro/install/)
``` bash
minikube start
```

* [Postgresql](https://phoenixnap.com/kb/postgresql-kubernetes)
* [Flyte](https://github.com/davidmirror-ops/flyte-the-hard-way/blob/main/docs/on-premises/single-node/002-single-node-onprem-install.md)
* [Mlflow](https://medium.com/@heisash24/-84bd8496f360)
* [Monitoring stack (Prometheus & Grafana)](https://medium.com/@brightband/deploying-prometheus-operator-to-a-kubernetes-cluster-c2378038c79b)
* [Private docker registry](https://hub.docker.com/r/ogouni604/mlops-rakuten) 


## Usage

``` bash
pyflyte run --remote {path_to_workflow.py} {workflow_name}
```


## How it looks like

<p align="center">
  <img src="./dashboard.jpeg">
</p>

## References


