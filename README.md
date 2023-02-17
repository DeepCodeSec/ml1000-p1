# Project 1 for ML1000 Course

## Installation

To install the _Python_ program locally, first create a virtual environment:

```sh
$ python3 -m venv venv
$ source ./bin/venv/activate
$ (venv)
```

Next, install the required modules:

```sh
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

To run the server locally, simply use the following command:

```sh
$ (venv) python3 app.py
```

You can then use a browser and access the main website via `http://127.0.0.1:5000`

## Portal

Available at [https://wineplus.herokuapp.com/](https://wineplus.herokuapp.com/)

## References

* [Github](https://github.com/DeepCodeSec/ml1000-p1)
* [pycaret](https://pycaret.gitbook.io/docs/)
* [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)