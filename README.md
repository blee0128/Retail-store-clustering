# Retail Store Clustering

# Prerequisites

- Python 3.7

In your virtual environment, install the requirements,
```bash
pip install -r requirements.txt
```

# Project
It reads data from sample [input](input.json) file, does clustering for the retail store, and save the results into [output](output.json) file.
```bash
python main.py
```

# Integrate as a Module

Import the class and instantiate. The expected input and output are as follow.

```python
from clustering import ProfileClustering

# update the number of component and number of cluster
number_of_component = 3
number_of_cluster = 3

# run detection
pc = ProfileClustering(number_of_component,number_of_cluster)  # any config for clustering???
output_data = pc.process(input_data)  # see below for input and output data format
```

## Input Data

```
[
    {
        "Id": "123",
        "Feature1": "string",
        "Feature2": 10000,
        "Feature3": -0.56,
        "Feature4": true,
        ...
    },
    {
        "Id": "123",
        "Feature1": "string",
        "Feature2": 10000,
        "Feature3": -0.56,
        "Feature4": true,
        ...
    },
    ...
]
```

## Output Data

```
[
    {
        "Cluster": 0,    # Outliers or unclustered entities
        "Name": "Unclustered",
        "Ids": [
        ],
        "Similarities": [
        ],
        "Differences": [
        ]
    },
    {
        "Cluster": 1,
        "Name": "Low activity stores",
        "Ids": [
            "123",
            "456",
            ...
        ],
        "Similarities": [    # top 5 most similar features in the cluster, in desc order
            {
                "Feature": "Feature2",
                "Mean": 12344,
                "Variance": 5.66
            },
            {
                "Feature": "Feature4",
                "Mean": true,
                "Variance": null
            },
            ...
        ],
        "Differences": [    # top 5 most different features in the cluster, in desc order
            {
                "Feature": "Feature3",
                "Mean": -5.67,
                "Variance": 102.45
            },
            {
                "Feature": "Feature1",
                "Mean": null,
                "Variance": null
            },
            ...
        ]
    },
    ...
]
```

