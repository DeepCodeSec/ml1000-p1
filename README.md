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

* Adjust the data to optimize the model
* Integrate latest model into training process

## References

* [pycaret](https://pycaret.gitbook.io/docs/)
* [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)