from networks import *
import torch
import time

# run network of choice
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running with ' + str(device) + '...')

    # discrete2(device, spatial=0, temporal=1)

    for spatial in [5, 20]:
        for temporal in [1, 2, 3]:
            for type in ['cont', 'disc1', 'disc3']:
                start = time.time()
                # try:
                network(device, type, temporal, spatial)
                # except Exception as e:
                #     print("Broke with type={}, spatial={}, temporal={}".format(type, temporal, spatial))
                #     print(e)
                # print('Time elapsed: {}'.format(time.time()-start))
