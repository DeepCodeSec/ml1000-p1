
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import sys
import glob
import pickle
import argparse
import logging
from datetime import datetime
#
from pycaret.classification import load_model, predict_model
import pandas as pd
from flask import Flask, request, render_template, jsonify
#
from model import WineQualityDataset
#
OPT_VERBOSE_HELP = "Display additional information about execution."
#
logger = logging.getLogger(__name__)
app = Flask(__name__)

def get_newest_file(path):
    """ Selects the most recent file created from the `models` directory. """
    files = glob.glob(os.path.join(path, '*.pkl'))
    return max(files, key=os.path.getctime)

# Check if were are currently running on Heroku
if 'DYNO' in os.environ:
    # if so load the model on execution
    model_file = get_newest_file(os.path.join(os.getcwd(), "models"))
    logger.debug(f"Latest model: {model_file}")
    current_model = load_model(model_file.split('.', maxsplit=1)[0])
else: # Otherwise let the user decide how to load the model.
    current_model = None

@app.route('/')
def home():
    """ Shows the home page using the 'templates/home.html' page """
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
    """ Endpoint processing the data to predict the quality. """
    # Extract the data from the form
    logger.debug(request.form)
    #
    fixed_acidity = float(request.form['fixed_acidity'])
    volatile_acidity = float(request.form['volatile_acidity'])
    citric_acid = float(request.form['citric_acid'])
    residual_sugar = float(request.form['residual_sugar'])
    chlorides = float(request.form['chlorides'])
    free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
    total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
    density = float(request.form['density'])
    ph = float(request.form['pH'])
    sulphates = float(request.form['sulphates'])
    alcohol = float(request.form['alcohol'])

    logger.info(f"Received request with the following data items: ")
    logger.info(f"\t fixed_acidity                    : {fixed_acidity}")
    logger.info(f"\t volatile_acidity                 : {volatile_acidity}")
    logger.info(f"\t citric_acid                      : {citric_acid}")
    logger.info(f"\t residual_sugar                   : {residual_sugar}")
    logger.info(f"\t chlorides                        : {chlorides}")
    logger.info(f"\t free_sulfur_dioxide              : {free_sulfur_dioxide}")
    logger.info(f"\t total_sulfur_dioxide             : {total_sulfur_dioxide}")
    logger.info(f"\t density                          : {density}")
    logger.info(f"\t pH                               : {ph}")
    logger.info(f"\t sulphates                        : {sulphates}")
    logger.info(f"\t alcohol                          : {alcohol}")

    # Perform guessing here
    score = 0.0
    label = None
    global current_model
    if current_model is not None:
        # Create a sample dataset for prediction
        data = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur_dioxide],
            'total sulfur dioxide': [total_sulfur_dioxide],
            'density': [density],
            'pH': [ph],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        })
        predictions = predict_model(current_model, data=data)
        logger.info(predictions)
        score = predictions['prediction_score'].iloc[0]
        label = predictions['prediction_label'].iloc[0]
        logger.info(f"Predicted quality: {label} ({score}).")
    else:
        logger.error(f"No model defined.")

    return jsonify({
        "label": label,
        "score": score
    })

def main(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="")
    parser.add_argument('-t', '--train',
                    dest="do_train",
                    action="store_true",
                    help="Train a new model without starting the web server.")
    parser.add_argument('-p', '--port',
                    dest="port",
                    default=5000,
                    help="Port to listen on for HTTP requests.")
    parser.add_argument('-m', '--model',
                    dest="model",
                    required=False,
                    help="Model file to use.")
    parser.add_argument('-v', '--verbose',
                        default=False,
                        action="store_true",
                        dest="is_debug",
                        help=OPT_VERBOSE_HELP)

    logging.basicConfig(level=logging.DEBUG)

    # Parse command-line arguents
    args = parser.parse_args(args=argv[1:])
    print("")

    # Train a new model
    if args.do_train:
        # Load the data
        datafile = os.path.abspath('./data/winequality-white.csv')
        dataset = WineQualityDataset(_file=datafile, _target_col="quality")
        logger.info(f"{dataset.nb_rows} row(s) loaded from '{datafile}'.")
        # Generate the model
        logger.info("Selecting best classifier model...")
        model = dataset.best_model
        # Print information about the best model
        print(model)
        # Generate a file name based on the current date and time
        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{now}"
        file_path = os.path.abspath(os.path.join(".", "models", file_name))
        # Save the best model to a file
        dataset.save_best_model_to(file_path)
        logger.info(f"Save model to '{file_path}.")
    # Otherwise, start the server
    else:
        # Load the latest model
        model_file = get_newest_file(os.path.join(os.getcwd(), "models"))
        logger.info(f"Selected '{model_file}' as model.")
        # For some reason, `load_model` appends `.pkl` to the file, so 
        # we need to remove it.
        global current_model
        current_model = load_model(model_file.split('.', maxsplit=1)[0])
        # Start the server
        app.run(port=int(args.port), debug=bool(args.is_debug), threaded=True)

def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))

if __name__ == '__main__':
    entry_point()

