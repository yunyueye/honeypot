# honeypot
An algorithm about detecting  ICS honeypot

The dataset(data_no_ip.csv) contains three common ICS honeypots, Modbus, ATG, and S7.

The processed data is located in the 'data' folder, containing data for three different protocols. It can be directly used for running the proposal_algorithm_1.py script.(These files can be obtained by running process_*.py）

This program needs to be run in a Python3 or higher environment, and requires the libraries numpy and pandas.

example:
    """
    feature_data = np.loadtxt("data/numpy_array_data_atg")
    label_data = np.loadtxt("data/numpy_label_data_atg")
    """
    The above two lines of code indicate the loading of data from the ATG honeypot.
    
    The final test results are stored in the variable 'test_recognition'.:
        The i-th element in the list: the sample that was identified as a honeypot in the i-th iteration.
        The i-th element: (index of sample, p(y=1|X), iteration epoch)
        
    """
    feature_data = np.loadtxt("data/numpy_array_data_xx")
    label_data = np.loadtxt("data/numpy_label_data_xx")
    """
    you can load other protocol ICS honeypot data

-------------------------------
process_*.py :
After running the file, you will get two files: numpy_array_data_* and numpy_label_data_*. Put these two files into the data folder and then you can run proposal_algorithm_1.py. (I have put all the processed files into the "data" file)

-----------------------------
selectKfeature.py:
You can run selectKfeature.py to test whether your python environment contains the required packages

----------------------
Explanation of part of the code in proposal_algorithm_1.py:

    # Process test dataset
    test_output = [[] for i in range(len(selected_feature_index))] 
    for i in range(len(selected_feature_index)):   # Iterating over the optimal feature list
        optimal_i = selected_feature_index[0:i+1]  # Combining the first i features: (f1, f2, ..,fi)
        feature_data = test_feature_array[:, optimal_i]  # Fetching data for the corresponding feature
        feature_size = feature_data.shape          
        for sample_i in range(feature_size[0]):    # index of sample; Calculating the probability for the i-th sample  
            f_sample = list(feature_data[sample_i])   # Feature data for the i-th sample      
            for f_i in range(len(poster_p_list[i])):  # Looking up the table in the calculated posterior probability table
                if f_sample == poster_p_list[i][f_i][0]: # Finding the posterior probability for the current feature combination
                    f_p_0 = poster_p_list[i][f_i][1][0]  # p(y=0|(f1, f2, ..,fi))  
                    f_p_1 = poster_p_list[i][f_i][1][1]  # p(y=1|(f1, f2, ..,fi))  
                    break
            # Using Bayes' formula to calculate the probability of p(y=1) for the sample
            sample_p_1 = prior_p_1 * f_p_1 / (prior_p_1 * f_p_1 + prior_p_0 * f_p_0)                           
            test_output[i].append(sample_p_1)  # Saving the results
            """test_outout: The i-th sublist of the result represents the probability 
            of each sample being a honeypot in the i-th calculation; 
            The i-th calculation of probability uses the first i features (f1, f2, ..,fi)"""