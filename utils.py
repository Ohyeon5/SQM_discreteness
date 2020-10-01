# utilization functions 
import numpy as np
import pandas as pd 
import argparse
import configparser
import matplotlib.pyplot as plt

# read configuration from the config.ini files
def get_configs():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", dest="config", help="Configuration file used to run the script", required=True)
	parser.add_argument("--model_name", type=str, default=None, help='network type')
	parser.add_argument("-e", "--epochs", type=int, default=None, help='number of epochs')
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

	# overwrite experiment specific parameters
	if args.model_name is not None:
		param['model_name'] = args.model_name
	if args.epochs is not None:
		param['epochs'] = args.epochs
	

	return param

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

    plt.savefig('gard_flow.png')
