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

	config = configparser.RawConfigParser(allow_no_value=True)
	config.read(args.config)

	# initialize parameter
	param = dict()

	param['device_name'] = config.get('Device','name',fallback='test_device')
	# path
	param['data_path']   = config.get('Path','data_path')
	param['img_path']    = param['data_path'] + config.get('Path','img_path',  fallback='20bn-jester-v1')
	param['csv_labels']  = param['data_path'] + config.get('Path','csv_labels',fallback='jester-v1-labels.csv')
	param['csv_train']   = param['data_path'] + config.get('Path','csv_train', fallback='jester-v1-train.csv')
	param['csv_val']     = param['data_path'] + config.get('Path','csv_val',   fallback='jester-v1-validation.csv')
	param['csv_test']    = param['data_path'] + config.get('Path','csv_test',  fallback='jester-v1-test.csv')
	# model
	param['model_name']  = config.get('Model','model_name',   fallback='test_model')
	param['model_path']  = config.get('Model','model_path',   fallback='./saved_models/')
	param['batch_size']  = config.getint('Model','batch_size',fallback=20)
	param['epochs']      = config.getint('Model','epochs',    fallback=300)
	# Data
	param['labels']      = config.get('Data','labels').split(',') # need to implement if empty use all labels 
	# if param['labels'] is 'all': use all the labels from csv_labels file
	if 'all' in param['labels']: 
		param['labels']  =	pd.read_csv(param['csv_labels'], index_col=False).values.squeeze().tolist()
	param['skip']        = config.getint('Data','skip',   fallback=2)
	param['im_size']     = config.getint('Data','im_size',fallback=50)

	# mode
	param['train']  = config.getboolean('Mode','train', fallback=True)
	param['test']   = config.getboolean('Mode','test' , fallback=False)

	return param
