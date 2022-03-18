import numpy as np
import math
import json
import statistics
import matplotlib.pyplot as plt

font = {'size'   : 18}

plt.rc('font', **font)



def automation_remove_below_mean(model):
    # this method selects all variances below the mean and returns them so they can be removed
    all_varianz_values = find_varianz_values(model)

    # calc mean
    mean = np.mean([x for (y,x) in all_varianz_values])

    varianz_values_below_mean = []

    # select only elemtens which are lower than mean
    for elem in all_varianz_values:
        if elem[1] < mean:
            varianz_values_below_mean.append(elem)


    return varianz_values_below_mean    



def return_n_smallest_variances(model, num_to_eliminate):
    # this method returns the n smallest varainz values
    all_varianz_values = find_varianz_values(model)

    sorted_varianz_values = sorted(all_varianz_values, key=lambda tup: tup[1]) # sorts them ascending by the second element in the tuple

    """
    # the following code was used to create the Varianzverteilung Abbildungen
    mean = np.mean([x for (y,x) in sorted_varianz_values])
    just_values = [x for (y,x) in sorted_varianz_values]
    plt.figure(figsize=(14,8))
    # matplotlib histogram
    plt.hist(just_values, color = 'grey',
             bins = 100)

    #add mean in hist plot
    plt.axvline(x=mean)
    # Add labels
    plt.title('Histogramm von Varianzwerten')
    plt.xlabel('Varianz')
    plt.ylabel('Häufigkeit')

    plt.show()

    bigger_than_mean = [x for x in just_values if x > mean]

    plt.figure(figsize=(14,8))
    # matplotlib histogram
    plt.hist(bigger_than_mean, color = 'grey',
             bins = 150)

    # Add labels
    #plt.title('Histogramm von Varianzwerten größer als der durchschnittliche Wert')
    plt.xlabel('Varianz')
    plt.ylabel('Häufigkeit')

    plt.show()
    """
    return sorted_varianz_values[:num_to_eliminate] # returns the n smallest varianz values with the corresponding index



def find_varianz_values(model):
    # this method computes the variance values and selects the maximum value for each neuron in the hidden layer (see Kapitel 4)

    # this finds the weights 
    for layer in model.layers:
            if layer.name == "hidden_layer":
                weights = layer.get_weights()[0]
                hidden_layer_size = len(layer.get_weights()[1])
    # create lists
    varianz_per_neuron = {x:None for x in list(range(hidden_layer_size))}

    #calculate varianz values and save them
    for x in range(hidden_layer_size):
        weights_to_list =  weights[:,[x]].tolist()  # gets columns from the weights
        weights_to_list = [x[0] for x in weights_to_list]      # remove unnessccary one-elemnent-lists
        recreate_input_chars = np.array_split(weights_to_list, weights.shape[0] / 22) # this recreates the input chars, so that the script knows which values belong to each other

        varianz_per_input = {}

        for index,input_char in enumerate(recreate_input_chars):
            varianz_per_input[index] = np.var(input_char) #calc varianz and save it in dic with index

        varianz_per_neuron[x] = varianz_per_input  

    

    #Choose maximum for each input from the values
    max_varianz_per_input = []
    for key in varianz_per_neuron[0].keys(): # this
        tmp = []                                         # init  tmp empty numpy array
        for k in varianz_per_neuron.keys():              # this checks each hidden layer neuron
            tmp.append(varianz_per_neuron[k][key])       
        tmp = sorted(tmp)
        max_varianz_per_input.append((key,tmp[(len(tmp)-1)])) # this selects the maximum varianz value for each input character

    
    return max_varianz_per_input
    


def compare_results(predictions, y_true):
    # This method compares the results from the model with the labels from the dataset
    mistakes = 0
    correct = 0
    for counter,row in enumerate(predictions):
            result = np.where(row == np.amax(row)) # this line finds at which postion in the row the highest value is -> 
            if result[0] != y_true[counter]:       # check if that number matches the class
                mistakes +=1
            else:
                correct +=1

    correct_classifications = (correct)
    false_classifications = (mistakes)
    accuracy = (1 - (mistakes/len(y_true)))

    return (correct_classifications, false_classifications, accuracy)
    


def fit_testdata(indices, x_test):
    # this method trims the test data so that it matches with the results from the method
    # the iterative method

    new_x = []
    for seq in x_test:
        fitted_seq = []
        for index in indices:
            fitted_seq.append(seq[index])
        new_x.append(fitted_seq)

    return np.asarray(new_x)



def save_results(filename, data_dict):
    # this method saves the results from each run into a txt file for later use
    with open("results\\" + filename + ".txt", "a", encoding="utf-8") as file:
        json.dump(data_dict, file)



def load_json_objects(filename):
    with open("results\\" + filename + ".txt", "r", encoding="utf-8") as file:
        obj = json.load(file)

    return obj

