import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

font = {'size'   : 22}

plt.rc('font', **font)


def load_json_objects(filename):
    # this method loads json data
    with open("results\\" + filename + ".txt", "r", encoding="utf-8") as file:
        obj = json.load(file)

    return obj


def examine_data(data):

        # this is how every object in the data looks like,
        # the data contains a 50 objects long list (one for each run), where every object contains these data_dicts
        """
        data_dict = {   "date": datetime.now().strftime("%H:%M:%S"),
                    "correct_classifications" : correct_classifications,
                    "false_classifications" : false_classifications,
                    "accuracy" : accuracy,
                    "remaining_indices" : indices.tolist()
                }
        """

        counter = 0
        list_of_last_4_iters = []
        list_of_durations = []

        for run in data:

            counter += 1

            run_start = datetime.strptime(run[0]["date"], '%H:%M:%S')
            run_end = datetime.strptime(run[len(run)-1]["date"], '%H:%M:%S')
            duration = (run_end - run_start)
            #print(duration)
            list_of_durations.append(duration.total_seconds())

            last_4_iterations = run[-4:]
            list_of_last_4_iters.append(last_4_iterations)

        #get_accuracy_of_feature(list_of_last_4_iters, 355)
        #eval_last_4_iters_accuracy(list_of_last_4_iters)
        eval_last_4_iters_frequency(list_of_last_4_iters)
        #get_avg_amount_of_features(list_of_last_4_iters)

        
        # prints out avg duration
        avg_duration = sum(list_of_durations)/len(list_of_durations)
        ty_res = time.gmtime(avg_duration)
        res = time.strftime("%M:%S",ty_res)
        print("Average duration : {}".format(res))



def get_accuracy_of_feature(list_of_last_4_iters, feature):
    # this method finds the accuracy, of the nn, if only one feature was used
    accurcy_of_feature_list = []
    for run in list_of_last_4_iters:
        if run[3]["remaining_indices"] == [feature]:
            print(run[3]["accuracy"])




def get_avg_amount_of_features(list_of_last_4_iters):
    # this method computes the average remaining features in the last four iterations
    # it was used for creating the tables 3 and 4
    for i in range(4):
        amount_remaining_features = []
        for run in list_of_last_4_iters:
            amount_remaining_features.append(len(run[i]["remaining_indices"]))
        avg = sum(amount_remaining_features)  / len(amount_remaining_features)
        print("Average remaining features in run {} : {}".format(i, avg))



def eval_last_4_iters_accuracy(list_of_last_4_iters):
    # this method computes the accuracy of the last four iterations, it 
    # was used for creating the tables 1-4
    for i in range(4):
        accuracy_list = []
        for run in list_of_last_4_iters:
                accuracy_list.append(run[i]["accuracy"])
        print("Accuracy list of {}th last run : {}".format((4-i),accuracy_list))
        print(len(accuracy_list))
        avg = sum(accuracy_list) / len(accuracy_list)
        print("Average: {}".format(avg))


def eval_last_4_iters_frequency(list_of_last_4_iters):
    # this method created the Abbildungen Viertletzte-Letzte Iteration for Versuch 1 and 2 
    for i in range(4):
        frequency_dict = {}
        for run in list_of_last_4_iters:
            for elem in run[i]["remaining_indices"]:
                if elem not in frequency_dict.keys():
                    frequency_dict[elem] = 1
                else:
                    frequency_dict[elem] +=1


        pd_df = pd.DataFrame(list(frequency_dict.items()))
        pd_df.columns =["Index","Frequency"]
        

        pd_df = pd_df.drop(pd_df[pd_df.Frequency < 4].index)

        # sort by Frequncy column
        pd_df = pd_df.sort_values(['Frequency']).reset_index(drop=True)


        plt.figure(figsize=(14,8))
        ax = sns.barplot(pd_df.index, pd_df.Frequency, palette="dark")
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax.set(xlabel="Index", ylabel='Häufigkeit')
        ax.set_xticklabels(pd_df.Index)

        for bar in ax.patches:
           bar.set_color('grey')  

        
    plt.show()

    

if __name__ == "__main__":
    
    # 4 Klassen manuell : results_manuell_4_klassen
    # 6 Klassen manuell : results_manuell_6_klassen
    # 4 Klassen automatisch : results_automatic_4_klassen
    # 6 Klassen automatisch : results_automatic_6_klassen

    data = load_json_objects("results_automatic_4_klassen")
    
    examine_data(data)
    #get_avg_removal_iteration(data)
    


    


    
