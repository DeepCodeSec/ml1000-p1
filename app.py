
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
import os
import sys
import argparse
import logging
from datetime import datetime
#
from flask import Flask, request, url_for, redirect, render_template, jsonify
#
from model import WineQualityDataset
#
OPT_VERBOSE_HELP = "Display additional information about execution."
#
logger = logging.getLogger(__name__)
app = Flask(__name__)
#
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/process', methods=['POST'])
def process():
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
    logger.info(f"\tfixed_acidity                    : {fixed_acidity}")
    logger.info(f"\tvolatile_acidity                 : {volatile_acidity}")
    logger.info(f"\tcitric_acid                      : {citric_acid}")
    logger.info(f"\tresidual_sugar                   : {residual_sugar}")
    logger.info(f"\tchlorides                        : {chlorides}")
    logger.info(f"\tfree_sulfur_dioxide              : {free_sulfur_dioxide}")
    logger.info(f"\ttotal_sulfur_dioxide             : {total_sulfur_dioxide}")
    logger.info(f"\tdensity                          : {density}")
    logger.info(f"\tpH                               : {ph}")
    logger.info(f"\tsulphates                        : {sulphates}")
    logger.info(f"\talcohol                          : {alcohol}")

    # Perform guessing here
    quality = 0.0
    return render_template("result.html", quality=quality)


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
        datafile = os.path.abspath('./data/winquality-red.csv')
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
        # TODO: Select the most recent model, or the model specified
        # TODO: Load the model
        # Start the server
        app.run(port=int(args.port), debug=bool(args.is_debug), threaded=True)

def entry_point():
    """Zero-argument entry point for use with setuptools/distribute."""
    raise SystemExit(main(sys.argv))

if __name__ == '__main__':
    entry_point()

