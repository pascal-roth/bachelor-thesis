| name                              | creator     | decription                                   
|-----------------------------------|-------------|----------------------------------------------
| abs_enthalpy.py                   | Pascal Roth | show evolution of the absolute enthalpy inside the reactor and that pressure adjusted enthalpy is constant 
| generate_data_comparison.sh       | Pascal Roth | generate the thermochemical states dataframe using the progress variable definition of the comparison between MLP and GRM model   
| generate_data_exp.sh              | Pascal Roth | generate the thermochemical states dataframe with similar initial conditions                    
| generate_data_test_p20.sh         | Pascal Roth | generate the thermochemical states dataframe for test data set with an initial pressure of 20bar                       
| generate_data_test_p40.sh         | Pascal Roth | generate the thermochemical states dataframe for test data set with an initial pressure of 20bar
| generate_data_train.sh            | Pascal Roth | generate the thermochemical states dataframe of the training data set
| Homogeneous_Reactor.py            | Pascal Roth | constant volume homogeneous reactor model implemented using the cantera software kit
| Homogeneous_Reactor_Multi.py      | Pascal Roth | constant volume homogeneous reactor model implemented using the cantera software kit (run on multiple CPU cores)
| iterator.py                       | Pascal Roth | execute homogeneous reactor model with different initial states
| iterator.sh                       | Pascal Roth | shell script to start iterator
| iterator_multi.py                 | Pascal Roth | execute homogeneous reactor model with different initial states (run on multiple CPU cores)
| iterator_multi.sh                 | Pascal Roth | shell script to start iterator multi
| pre_process_fc.py                 | Pascal Roth | functions called by iterator