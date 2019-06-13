# Data preprocessing for traffic management

## Variable
1. geohash6
2. day
3. timestamp
4. demand 


## Collecting data to dump into a pickle file
```python
python data.py
```

Data is stored in 2 list.

1. Demand matrix
2. Timing

Both the indexes of list correspond with each other. They are sequential.

## Reading data from pickle file
```python
python readdata.py
```

## Fake data

Tried increasing the dataset by generating a new middle value between timestep using moving average. It did not turn out well as the loss is higher.
