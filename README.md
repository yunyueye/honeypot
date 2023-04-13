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
