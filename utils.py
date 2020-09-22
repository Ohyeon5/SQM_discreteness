# utilization functions 
import numpy as np
import pandas as pd 
import argparse
import configparser

# read configuration from the config.ini files
def get_configs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", dest="config", help="Configuration file used to run the script", required=True)
	args = parser.parse_args()

	config = configparser.ConfigParser()
	config.read(args.config)

	# initialize parameter
	param = dict()

	param['device_name'] = config.get('Device','name')
	# path
	param['data_path']   = config.get('Path','data_path')
	param['img_path']    = param['data_path'] + config.get('Path','img_path')
	param['csv_labels']  = param['data_path'] + config.get('Path','csv_labels')
	param['csv_train']   = param['data_path'] + config.get('Path','csv_train')
	param['csv_val']     = param['data_path'] + config.get('Path','csv_val')
	param['csv_test']    = param['data_path'] + config.get('Path','csv_test')
	# model
	param['model_name']  = config.get('Model','model_name')
	param['batch_size']  = config.getint('Model','batch_size')
	param['epochs']      = config.getint('Model','epochs')
	param['labels']      = config.get('Model','labels').split(',') # need to implement if empty use all labels 

	# if param['labels'] is 'all': use all the labels from csv_labels file
	if 'all' in param['labels']: 
		param['labels']  =	pd.read_csv(param['csv_labels'], index_col=False).values.squeeze().tolist()

	# mode
	param['train']  = config.getboolean('Mode','train')
	param['test']   = config.getboolean('Mode','test')

	return param
