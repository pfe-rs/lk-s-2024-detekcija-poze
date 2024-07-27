from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import json
import os
import math


def sample_to_pil_image(sample: np.ndarray, image_shape):
    image = np.reshape(sample, newshape=image_shape)
    image = PIL.Image.fromarray(np.uint8(image*255))
    if image.size[0] < 140:
        new_size = (140, 140) 
    return image.resize(size=new_size)

def get_accuracy_train(l1, l2):
    list1 = l1.cpu()
    list1 = list1.detach().numpy()
    list2 = l2.cpu()
    list2 = list2.detach().numpy()
    size_l1 = list1.shape
    distances = []
    
    duz1 = 0
    duz2 = 0
    for o in range(size_l1[0]):
        for i in range(size_l1[1]):
            max_index1 = np.unravel_index(list1[o][i].argmax(), list1[o][i].shape)
            max_index2 = np.unravel_index(list2[o][i].argmax(), list2[o][i].shape)

            if max_index1[0]!=0 or max_index1[1]!=0:
                duz1+=1
            if max_index2[0]!=0 or max_index2[1]!=0:
                duz2+=1
                
            distancex = max_index1[1] - max_index2[1]
            distancey = max_index1[0] - max_index2[0]
            euclidian_distance = math.sqrt(distancex ** 2 + distancey ** 2)
            
            distances.append(euclidian_distance)
                 
    threshold = 7
    tp = np.sum(np.array(distances) < threshold)
    fp = duz1 - tp
    fn = duz2 - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall
            
def get_accuracy_test(l1, l2):
    list1 = l1.cpu()
    list1 = list1.detach().numpy()
    list2 = l2.cpu()
    list2 = list2.detach().numpy()
    
    size_l1 = list1.shape
    acc = 0

    koordsx1 = np.zeros((size_l1[1]), dtype=float)
    koordsy1 = np.zeros((size_l1[1]), dtype=float)
    koordsx2 = np.zeros((size_l1[1]), dtype=float)
    koordsy2 = np.zeros((size_l1[1]), dtype=float)
    
    for i in range(size_l1[0]):
        max_index1 = np.unravel_index(list1[i].argmax(), list1[i].shape)
        max_index2 = np.unravel_index(list2[i].argmax(), list2[i].shape)

    dot1 = max_index1
    dot2 = max_index2

    plt.figure(figsize=(6, 6))

    # Plot the dots
    plt.scatter(*dot1, color='blue', s=100, label='out 1')  # Blue dot
    plt.scatter(*dot2, color='red', s=100, label='input 2')   # Red dot

    # Set axis limits to match the grid size
    plt.xlim(0, 45)
    plt.ylim(0, 45)

    # Add labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Two Dots on a 45x45 Grid')
    plt.legend()

    # Add gridlines for better visibility
    plt.grid(True)

    # Show the plot
    plt.gca().invert_yaxis()  # Invert y-axis to match typical grid orientation
    plt.show()
    
    distancex = max_index1[1] - koordsx2[1]
    distancey = max_index1[0] - koordsx2[0]
    euclidian_distance = math.sqrt(distancex ** 2 + distancey ** 2)
    #np.sqrt(np.power(distancex, np.array([2] * distancex.shape[0])) + 
    #                            np.power(distancey, np.array([2] * distancex.shape[0])))
    for i in range(len(list1)):
        if euclidian_distance <= 7:
            acc += 1
    return acc / (size_l1[1] * size_l1[0])

def plot_curve(curves: Tuple[List], labels: Tuple[List], plot_name):
    
    fig, ax = plt.subplots(figsize=[6.4, 3])

    max_len = 0
    for curve in curves:
        if len(curve) > max_len:
            max_len = len(curve)
    
    x_axis = list(range(max_len))
    
    for curve, label in zip(curves, labels):
        ax.plot(x_axis, curve, label=label)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel(plot_name)
    ax.legend()
    fig.tight_layout()
    return fig


def keep_store_dict(curve, label, store_dict: Dict=None):
    
    if store_dict is None:
        store_dict = {}
    
    if not isinstance(curve, list):
        if torch.is_tensor(curve):
            curve = curve.detach().numpy()
        
        if isinstance(curve, np.ndarray):
            curve = curve.tolist()
        else:
            curve = [curve]

    if label in store_dict:    
        store_dict[label].extend(curve)
    else:
        store_dict[label] = curve
    
    return store_dict


def store_dict_to_disk(file_path: str, store_dict: Dict, overwrite: bool = False):
    # Load existing dictionary if not overwriting
    if not overwrite and os.path.isfile(file_path):
        prev_dict = load_dict_from_disk(file_path)
    else:
        prev_dict = {}

    # Merge existing dictionary with the new one
    for key, val in store_dict.items():
        if key in prev_dict:
            # Ensure the existing value is a list before extending
            if isinstance(prev_dict[key], list):
                prev_dict[key].extend(val)  # Extend the existing list
            else:
                # If it's not a list, convert it to a list
                prev_dict[key] = [prev_dict[key]] + val
        else:
            # If the key does not exist, just add it
            prev_dict[key] = val
    
    # Write the merged dictionary back to disk
    with open(file_path, 'w') as f:
        json.dump(prev_dict, f, indent=4)


def load_dict_from_disk(file_path: str) -> Dict:
    if not os.path.isfile(file_path):
        return {}

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}