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

Available at [https://ml1000-p1.herokuapp.com/](https://ml1000-p1.herokuapp.com/)

## TODO

* Adjust the data to optimize the model
* Integrate latest model into training process

## Troubleshooting

* `ModuleNotFoundError: No module named 'pycaret.internal.memory'`

When pushing the _Heroku_, sending a request to the `/` endpoint generated the following exception:

```
2023-02-15T01:55:46.150259+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/site-packages/pycaret/internal/persistence.py", line 391, in load_model
2023-02-15T01:55:46.150259+00:00 app[web.1]:     model = joblib.load(model_name)
2023-02-15T01:55:46.150260+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 658, in load
2023-02-15T01:55:46.150260+00:00 app[web.1]:     obj = _unpickle(fobj, filename, mmap_mode)
2023-02-15T01:55:46.150261+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/site-packages/joblib/numpy_pickle.py", line 577, in _unpickle
2023-02-15T01:55:46.150261+00:00 app[web.1]:     obj = unpickler.load()
2023-02-15T01:55:46.150261+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/pickle.py", line 1213, in load
2023-02-15T01:55:46.150261+00:00 app[web.1]:     dispatch[key[0]](self)
2023-02-15T01:55:46.150262+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/pickle.py", line 1538, in load_stack_global
2023-02-15T01:55:46.150262+00:00 app[web.1]:     self.append(self.find_class(module, name))
2023-02-15T01:55:46.150263+00:00 app[web.1]:   File "/app/.heroku/python/lib/python3.10/pickle.py", line 1580, in find_class
2023-02-15T01:55:46.150263+00:00 app[web.1]:     __import__(module, level=0)
2023-02-15T01:55:46.150263+00:00 app[web.1]: ModuleNotFoundError: No module named 'pycaret.internal.memory'
```

## References

* [pycaret](https://pycaret.gitbook.io/docs/)
* [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)