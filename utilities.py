import numpy as np     
    
def calc_vote(results: np.ndarray):
    """
    VoteClassifier 

    Args:
        results: Target matrix - results for all Decision Trees

    Returns:
        Majority classification vote for each instance (sample)

    Raises:

    """
    results = results.sum(axis=0)

    output = []
    for item in results:
        counters = np.unique(list(item), return_counts=True)
        counters_dict = dict(zip(counters[0],counters[1]))
        output += max(counters_dict, key=counters_dict.get)

    return np.array(output)

    
def correct_class(target: np.ndarray,
                  predicted: np.ndarray,
                  sample_size: int
                 ):
    """
    Get correctly classified results

    Args:
        target: Target values
        result: Classified values

    Returns:
        Array of values where match (1) occurs and no match (0) occurs
    """
    intersection_list = []
    for i in range(sample_size):
        if target[i]==predicted[i]:
            intersection_list.append(1)
        else:
            intersection_list.append(0)

    return np.array(intersection_list)

def binary_encode(target: np.ndarray,
                  classifier
                 ):
    """
    Encode two target classifications into 1 and -1

    Args:
        target: Array to convert
        classifier: Instance of classifier (self in many cases)

    Returns:
        Encoded array (-1,1)

    Raises:
        ValueError
    """
    sample_size = classifier._sample_size

    labels = np.unique(target)
    if len(labels) > 2:
        raise ValueError("Boosting only supports binary classificaton")

    encoded = np.empty(shape=target.shape,dtype=int)
    if not hasattr(classifier, "_labels"):
        unqiue_vals = np.unique(target)
        labels = {}
        labels[-1] = str(unqiue_vals[0])
        labels[1] = str(unqiue_vals[1])

    for i in range(sample_size):
        if str(target[i]) == labels[-1]:
            encoded[i] = -1
        else:
            encoded[i] = 1

    return encoded, labels

def binary_decode(target: np.ndarray,
                  classifier
                 ):
    """
    Decode classification from (-1,1) into labels defined in binary_encode
    Binary_encode must have been called

    Args:
        target: Array to convert
        classifier: Instance of classifier (self in many cases)

    Returns:
        Decoded binary array
    """
    sample_size = len(target)
    labels = classifier._labels

    decoded = np.empty(shape=target.shape,dtype=object)
    for i in range(sample_size):
        if target[i] == -1:
            decoded[i] = labels[-1]
        else:
            decoded[i] = labels[1]

    return decoded