from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt

font = {'size'   : 22}

plt.rc('font', **font)



def load_fasta(path_to_fasta, extended):
# this method loads the fasta data
    x = []
    y = []

    for seq_record in SeqIO.parse(path_to_fasta, "fasta"):
        x.append(str(seq_record.seq)[:-1])
        y.append(seq_record.id)

    if extended:
        y = assign_classes_extended(y)
    else:
        y = assign_classes(y)


    return x,y


def one_hot_encode(sequences):
    onehot_encoded_seqs = []
    alphabet = 'MFHLVDQTIAERKSWNYPGCX-'

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    #int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    for seq in sequences:
        integer_encoded = [char_to_int[char] for char in seq]
        onehot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded.append(letter)
        onehot_encoded_seqs.append(onehot_encoded)

    return np.asarray(onehot_encoded_seqs)


def one_hot_decode(encoded_sequences):
    decoded_sequences = []

    alphabet = 'MFHLVDQTIAERKSWNYPGCX-'

    int_to_char = dict((i, c) for i, c in enumerate(alphabet))

    for seq in encoded_sequences:
        #find on which index the one is
        liste_of_ones = list(zip(*np.where(seq == 1)))
        # replace the index of the one with the right char
        decoded_chars = [int_to_char[b] for a,b in liste_of_ones]
        decoded_sequence = ''.join(decoded_chars)
        decoded_sequences.append(decoded_sequence)

    return decoded_sequences


def load_and_encode(path, extended):
    seqs, labels = load_fasta(path, extended)
    one_hot_seqs = one_hot_encode(seqs)

    return one_hot_seqs,np.asarray(labels)

"""
def add_noise(x, y, noise_len, seq_set ):
    noise = []
    # generate noise sequences with the length and chars of actual data
    for _ in range(noise_len):
        noise.append(''.join(random.choice(seq_set) for _ in range(x.shape[1])))

        # append zeros for noise data
        y = np.append(y,0)

    # one hot encode the noise
    z = one_hot_encode(noise)

    # add noise to data
    x = np.concatenate((x,z))


    return x,y
"""

def assign_classes(seqs_ids):
    # 0 -> Human
    # 1 -> Civet
    # 2 -> Rino
    # 3 -> M-Jav
    classes = []

    for seqs_id in seqs_ids:
        if "SARS" in seqs_id:
            classes.append(0)
        elif "civet" in seqs_id:
            classes.append(1)
        elif "R_" in seqs_id:
            classes.append(2)
        elif "M_jav" in seqs_id:
            classes.append(3)
        else:
            print("The following seqs ids do no match any class:")
            print(seqs_id)
            exit()

    #get_class_data(classes)
    return classes

def assign_classes_extended(seqs_ids):
    # 0 -> Human
    # 1 -> Civet
    # 2 -> R_fer
    # 3 -> R_sin
    # 4 -> R rest
    # 5 -> M-Jav
    classes = []

    for seqs_id in seqs_ids:
        if "SARS" in seqs_id:
            classes.append(0)
        elif "civet" in seqs_id:
            classes.append(1)
        elif "R_fer" in seqs_id or "R__fer" in seqs_id:
            classes.append(2)
        elif "R_sin" in seqs_id or "R__sin" in seqs_id:
            classes.append(3)
        elif "R_" in seqs_id:
            classes.append(4)
        elif "M_jav" in seqs_id:
            classes.append(5)
        else:
            print("The following seqs ids do no match any class:")
            print(seqs_id)
            exit()

    #get_class_data(classes)
    return classes

def get_class_data(class_distribution):
    distribution_dict = {}

    for value in class_distribution:
        if value in distribution_dict:
            distribution_dict[value] += 1
        else:
            distribution_dict[value] = 1
    print(distribution_dict)

    y = []
    x = []
    for value in range(len(distribution_dict)):
        y.append(distribution_dict[value])
        x.append(value)
    #y = [distribution_dict[0],distribution_dict[1],distribution_dict[2],distribution_dict[3]]
    print(y)
    if len(y) == 6:
        class_names = ["Mensch", "Zibetkatze", "1", "2", "3", "Schuppentier"]
    else:
        class_names = ["Mensch", "Zibetkatze", "Hufeisennasen", "Schuppentier"]
    pos = np.arange(len(class_names))
    #plt.figure(figsize=(14,8))
    plt.bar(x, y, color='grey')
    plt.xlabel("Klassen")
    plt.ylabel("Häufigkeit")
    plt.title("Klassenverteilung")

    #plt.xticks(x, x)
    plt.xticks(pos, class_names)

    plt.show()





