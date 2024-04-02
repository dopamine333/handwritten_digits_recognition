
from os import path
import numpy as np
import neural_network
from neural_network import NeuralNetwork

INPUTNUM = 784

def label_to_array(label_number):
    label = np.zeros((10, 1))
    label[label_number] = 1
    return label


def labels_list_to_array_list(labels_list_number):
    return [label_to_array(label) for label in labels_list_number]


def image_to_input(image: np.ndarray):
    return np.array(image).reshape(INPUTNUM, 1)/255


def images_list_to_input_list(images_list):
    return [image_to_input(image) for image in images_list]


def get_digit(output: np.ndarray):
    return np.argmax(output)


def testing(neuralnetwork: NeuralNetwork, print_progress=True):
    from mnist import MNIST
    mndata = MNIST('samples')
    
    images, labels = mndata.load_testing()
    testing_num = len(images)
    average_cost = 0
    accuracy = 0
    for n in range(testing_num):
        output = neuralnetwork.get(image_to_input(images[n]))
        digit = get_digit(output)
        cost = np.sum((output-label_to_array(labels[n]))**2)
        average_cost += cost
        if print_progress:
            print(f"testing\t\toutput : {digit} answer : {labels[n]}\t\tprogress : {n+1}/{testing_num}")

        if(labels[n] == digit):
            accuracy += 1

    average_cost /= testing_num
    accuracy /= testing_num
    if print_progress:
        print(f"""
        average cost : {average_cost}
        accuracy : {accuracy*100}%
        """)
    return average_cost, accuracy


def running(network=None,save_path=None):
    from mnist import MNIST
    mndata = MNIST('samples')
    import traceback
    
    if not network:
        network = NeuralNetwork([INPUTNUM, 30, 30, 10], 0.5, 15)
    print("loading mnist dataset...")
    images, labels = mndata.load_training()
    images=images_list_to_input_list(images)
    labels=labels_list_to_array_list(labels)
    print("Create a NeuralNetwork :")
    print(network.info())
    print("Start befort_training testing")
    input("Press any key to continue :")
    start_cost, start_accuracy = testing(network)
    print("Start training")
    input("Press any key to continue :")
    try:
        for _ in range(5):
            network.training(images,labels)
    except (EOFError, KeyboardInterrupt):
        print('User stops program')
    except:
        print('Unknown programming error')
        traceback.print_exc()
    finally:
        print("Start after_training testing")
        input("Press any key to continue :")
        final_cost, final_accuracy = testing(network)
        print(f"""
            start_cost : {start_cost}
            start_accuracy : {start_accuracy*100}%
            final_cost : {final_cost}
            final_accuracy : {final_accuracy*100}%
            """)
        if not save_path:
            input("Press any key to end :")
            return
        
        input("Press any key to save :")
        network.save(save_path)
        input("Press any key to end :")
        
def load_and_running(load_path):
    running(NeuralNetwork.create_from_file(load_path),load_path)
    
if __name__=="__main__":
    #path="data/tcd_deep.json"
    #n=NeuralNetwork([INPUTNUM, 12,12,12, 12, 10], 0.8, 8)
    #path="data/tcd_heigth.json"
    #n=NeuralNetwork([INPUTNUM, 120 ,10], 0.5, 15)
    #path="data/tcd_repeat.json"
    #n=NeuralNetwork([INPUTNUM, 30, 30 ,10], 0.5, 15)
    #path="data/tcd_big.json"
    #n=NeuralNetwork([INPUTNUM, 100, 100, 100 ,10], 0.5, 15)
    #path="data/tcd_small.json"
    #n=NeuralNetwork([INPUTNUM, 8, 8 ,10], 0.5, 15)
    #path="data/tcd_small_deep.json"
    #n=NeuralNetwork([INPUTNUM, 5,5,5,5,5,5,5,5 ,10], 0.5, 15)
    #path="data/tcd_deep2.json"
    #n=NeuralNetwork([INPUTNUM, 12,12,12,12,12,12,12,12,12,12,12 ,10], 0.5, 15)
    path="data/tcd_best.json"
    n=NeuralNetwork([INPUTNUM,10],0.08, 50)

    #load_and_running(path)
    running(n,save_path=path)
    