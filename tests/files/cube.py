import pandas


def cube_test():
    return pandas.DataFrame({
        "x": [0, 0, 0, 0, 1, 1, 1, 1, ],
        "y": [0, 0, 1, 1, 0, 0, 1, 1, ],
        "z": [0, 1, 1, 0, 0, 1, 1, 0, ]}
    )