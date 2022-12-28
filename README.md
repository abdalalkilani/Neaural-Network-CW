This repository only includes Part 2 of the CourseWork as it is what I worked on:

[**PART 2**](part2_house_value_regression.py)
------------

The training and testing of the neural network is done here. The full regressor is stored in the part2_model.pickle file.
In order to obtain the best model use the command for the California housing prices use:

`python3 part2_house_value_regression.py`

Changing the hyperparamters in the line(in the example_main function):

```python
regressor = Regressor(x_train, nb_epoch = number_of_epochs, neurons = desired_hidden_layer, learning_rate = desired_starting learning_rate, batch_size = desired_batch_size, early_stop = desired_early_stopping_value)
```

The best hyperparamters we determined are already set in this line - so no need to change it yoursellf.

If you want to test on another set of data, please remove the data in housing.csv and put your own data into this file.


Contributors for whole CW
------------
- AbdelQader AlKilany (aqa20)
- Luc Jones (lxj19)
- Nicholas Mytilineos (nem20)
- Paul Vangerow (pjv20)