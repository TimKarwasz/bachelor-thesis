#auxiliary modules
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime

# for printing the whole array
np.set_printoptions(threshold=sys.maxsize)

# machine learnig modules
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

#my modules
from variance_computation import save_results, compare_results, fit_testdata, return_n_smallest_variances, automation_remove_below_mean
from preprocessing_data import load_and_encode, one_hot_encode, one_hot_decode




def model_execution(data, labels):
    # this method creates the NN and executes it

    # set up loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # define neural net 
    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(name = "input_layer"),
                                tf.keras.layers.Dense(hidden_layer_size, activation='relu', name = "hidden_layer"),
                                tf.keras.layers.Dense(num_classes, activation='softmax', name= "output_layer") 
                                ])
    # compile it
    model.compile(optimizer=optimizer, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
    # run it
    history = model.fit(data, labels, epochs=epochs, verbose = 0)

    return history, model



def method_execution(x, y, n_to_remove, sequence_indices, automation = False):
    # this method contains the logic of the method, presented in the paper

    # Step one: Train the NN
    history, model = model_execution(x,y)

    # Step two: analyse the weights and find the n smallest varianz values (iterative variant) or the varianz values below the average varianz value (automatic variant)
    if automation:
        # var_to_remove holds tuples like this (42, 0.2) , where the first value is the indexpostion and the second one the variance value
        var_to_remove = automation_remove_below_mean(model)
    else:
        var_to_remove = return_n_smallest_variances(model,n_to_remove)  
    
    smaller_x = np.empty(shape=(x.shape[0],x.shape[1]-len(var_to_remove),x.shape[2]))

    # Step three: remove the input characters according to the indices returned in step two
    for elem in range(x.shape[0]): ## do 92 times
        new_row = np.delete(x[elem],[y for y,z in var_to_remove],0) # var_to_removes contains of (583, 7.863907e-05) so this deletes every row in the dataset with the indices in var_to_removes here 583 for example
        smaller_x[elem] = new_row # insert new_row in new dataset

    # for visibilty also remove them here. Sequence_indices is a list of indexpostions which keeps tracks of which postions were removed
    sequence_indices = np.delete(sequence_indices,[k for k,i in var_to_remove],0)

    return smaller_x, sequence_indices



def method(x, y, iterations, n_to_remove, automation):
    # This method orchestrates the execution of the method
    # It takes the data (x,y) and the number of iterations, as well as
    # how many features should be removed (n_to_remove)
    # the automation boolean decides if the iterative or the automatic execution is used
    # if the automatic execution is used the iterations and n_to_remove parameters will be ignored

    sequence_indices = list(range(x.shape[1])) #keep track of the indices

    #for saving the data
    list_of_data_dicts = []

    if automation:

        remaining_features = 42 
        iterations = 0

        # this is the automatic execution, which stops when the amount of remaining features is 1
        while remaining_features != 1:

            # executes the method
            x,sequence_indices = method_execution(x, y, n_to_remove, sequence_indices, automation)

            iterations += 1

            # tests the method results
            data_dict = testing_the_model(x,y, sequence_indices)

            #s save test results
            list_of_data_dicts.append(data_dict)

            print("Iteration {} finished succesfully".format(iterations))

            # prepare break condition
            remaining_features = len(sequence_indices)

    else:
        # Execute the NN and remove features for i Epochs
        for i in range(iterations):

            # execute the method
            x,sequence_indices = method_execution(x, y, n_to_remove, sequence_indices, automation)

            # tests the method results
            data_dict = testing_the_model(x,y, sequence_indices)

            # save test results
            list_of_data_dicts.append(data_dict)

            print("Iteration {} finished succesfully".format(i))

        
        if  not automation:
        # if the iterative method is used, remove the last four of five features on their own
            for i in range(4):
                x,sequence_indices = method_execution(x, y, 1, sequence_indices, automation)

                data_dict = testing_the_model(x,y,sequence_indices)
                list_of_data_dicts.append(data_dict)
        
    return list_of_data_dicts



def testing_the_model(x, y, indices):
    # this method tests the current state of features and returns a data dict

    history, model = model_execution(x, y) # train a model with the current indices

    predictions = model.predict(x) # predict with it to see how good it performs
    (correct_classifications, false_classifications, accuracy) = compare_results(predictions ,y) # evalutate

    data_dict = {   "date": datetime.now().strftime("%H:%M:%S"),
                    "correct_classifications" : correct_classifications,
                    "false_classifications" : false_classifications,
                    "accuracy" : accuracy,
                    "remaining_indices" : indices.tolist()
                }

    return data_dict


if __name__ == "__main__":

    # Hyperparameters
    learning_rate = 0.001 # standard value for adam
    hidden_layer_size = 64
    epochs = 250
    num_classes = 4 # this has to be 4 if  the boolean extended is set to False, and 6 if extended is set to True 

    # prepare data  
    x,y = load_and_encode("Path\to\SARSrCoV_spike_aln_m.fasta", extended = True) # True means 6 classes, False means 4

    list_of_all_runs = []
    ## used for executing the versuch 50 times
    for i in range(1): # 
        #False in the method() call means iterative mode here, True means automatic mode
        #data_of_single_run = method(x, y , 163, 8, False ) # this was used for Versuch 1 (163 iterations and 8 features removed per iteration (last 4 iterations excluded))
        data_of_single_run = method(x, y , 2, 1, True ) #  this was used for Versuch 2 (the second and third parameter dont matter when True (automatic mode) is used)
        list_of_all_runs.append(data_of_single_run) 
            
    # this saves the results
    #save_results("results", list_of_all_runs) # saves the results in results/results.txt



