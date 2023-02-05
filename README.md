# Project 1 for ML1000 Course

## Installation

```sh
$ python3 -m venv venv
$ source ./bin/venv/activate
$ (venv) pip install -r requirements.txt
```

## Usage

```sh
$python3 app.py -h
usage: app.py [-h] [-t] [-p PORT] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -t, --train           Train a new model without starting the web server.
  -p PORT, --port PORT  Port to listen on for HTTP requests.
  -v, --verbose         Display additional information about execution.
```

## Portal

Available at [https://ml1000-p1.herokuapp.com/](https://ml1000-p1.herokuapp.com/)

## TODO

* When not in training mode, load the most recent model if there is any.
* Readjust the business problem
* Adjust the data to optimize the model
* Implement process/predict endpoint

## References

* [pycaret](https://pycaret.gitbook.io/docs/)