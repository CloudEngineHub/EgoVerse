import h5py
import pickle
import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import websocket_client_policy
import yaml
from lerobot_utils import prepare_training_data

TEST_TYPE = "128dim"
assert TEST_TYPE in ["128dim", "separate"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval policy and plot figures before deploy', add_help=False)
    parser.add_argument('--hdf_file_path', type=str, help='hdf file path', required=True)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=64)
    parser.add_argument('--plot', action='store_true')

    # NOTE: you should change the host and port to the one you are using.
    client = websocket_client_policy.WebsocketClientPolicy(host="169.228.49.5", port=8000)

    args = vars(parser.parse_args())

    chunk_size = args['chunk_size']

    with h5py.File(args['hdf_file_path'], 'r') as data:
        actions = np.array(data['action'])
        left_imgs = np.array(data['observation.image.left'])
        right_imgs = np.array(data['observation.image.right'])
        states = np.array(data['observation.state'])

        if len(left_imgs.shape) == 2:
            # compressed images
            assert len(right_imgs.shape) == 2
            assert left_imgs.shape[0] == right_imgs.shape[0]
            # Decompress
            left_img_list = []
            right_img_list = []
            for i in range(left_imgs.shape[0]):
                left_img = cv2.imdecode(left_imgs[i], cv2.IMREAD_COLOR)
                right_img = cv2.imdecode(right_imgs[i], cv2.IMREAD_COLOR)
                left_img_list.append(left_img.transpose((2, 0, 1)))
                right_img_list.append(right_img.transpose((2, 0, 1)))
            # BCHW format
            left_imgs = np.stack(left_img_list, axis=0)
            right_imgs = np.stack(right_img_list, axis=0)
            plain_task_text = data.attrs['description']
        
    output = None
    act = None
    act_index = 0

    if args['plot']:
        predicted_list = []
        gt_list = []
        record_list = []

    for t in tqdm(range(states.shape[0])):
            print("step", t)
            t_start = t
            # Select offseted episodes
            cur_action = actions[t_start]
            cur_left_img = left_imgs[t_start]
            cur_right_img = right_imgs[t_start]
            cur_img = np.stack([cur_left_img, cur_right_img], axis=0)  # (2, 3, H, W)

            cur_state = states[t_start]

            if output is None or act_index == chunk_size - 10:
                
                obs = {
                    "image": cur_img.astype(np.uint8)[None, ...], # [0, 255]
                    "qpos": cur_state.astype(np.float32).reshape(1, -1),  # (1, 128)
                    'cond_dict': {
                        "plain_text": [plain_task_text]
                    }
                }
                

                print("Predicted Chunk exhausted at", t_start)
                output = client.infer(obs)['action'][0] # (chunk_size,action_dim)
                act_index = 0

            act = output[act_index]
            act_index += 1

            if args['plot']:
                indices = np.arange(30, 39)[:3]
                
                predicted_list.append(act[indices])
                gt_list.append(cur_action[indices])
                

    if args['plot']:
        print("plotting")
        plt.figure()

        for i in range(3):  # x, y, z
            plt.plot([x[i] for x in predicted_list], label=f'Predicted {i}', linestyle='--')  
            plt.plot([x[i] for x in gt_list], label=f'Ground Truth {i}') 

        plt.legend()
        plt.title("Predicted vs Ground Truth")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        plt.savefig("predicted_vs_gt.png")
        # plt.show()