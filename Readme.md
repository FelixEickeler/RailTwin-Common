## RailTwin-Common
### Setup
RailTwin-Common (rtc) was tested with python 3.9 and anaconda environment. The requirements are given by the requirements.txt.
```
conda env create --file requirements.yml python=3.9
```

Windows:
This module relies on spdlog. On Windows this is not available and needs to be installed separately. 
A version for python 3.9 was compiled and can be found on:

[Precompile spdlog py3.9](https://filedn.eu/l4hiESSdAeuuEoSLE7Uolr4/binary_supply/spdlog.cp39-win_amd64.pyd)

Linux:
Use the spdlog version from pip. The anaconda package seems broken !
```bash
pip install pybind11 spdlog
```

### Common

#### IO - module
All data will be read to a pandas dataframe with appropriate naming (see common.io._allowed_headers).
This is a unified interface that can be used in your programming ! 

Normally operations should be carried out on the data to normalize it. See the options in IOOperations and header.py !

Currently, the RailTwin-Common supports the following formats for reading and writing. 

| #   | Fileformat | read                    | write               |
|-----|---------|-------------------------|---------------------|
| 0   | *.ply | :heavy_check_mark:      | :heavy_check_mark:  |
| 1   | *.feather | :heavy_check_mark:      | :heavy_check_mark:  |
| 2   | *.laz | :heavy_check_mark:(1)   | :x: |
| 3   | *.xyz | :heavy_check_mark:(1,2) | :x: |

(1) no metadata can be read
(2) Format must be register, but reader is available

It is possible to register additional reader & writer by adding them to the
common.io.registered_reader or the common.io.registered_writer dictionary. 
```
common.io.registered_reader[".my_format"] = your_custom_read_function
common.io.registered_reader[".xyz"] = common.io.writer.csv.create_csv_reader(HeliosFormat)  
```

#### Logger
RailTwin-Common uses spdlog for logging. The logger can be reused in any part of your code by:
```python
logger = common.logger.RailTwinLogger.create()
```
Please see splog settings for more information

### utilites & plotting
1) RailTwin-Common comes with a tool to divide any supported infrastructure asset in subassets. This tool resides in utils track_divider.py
    ```
    track_divider.py
    ```
    and can be called directly.
2) RailTwin-Common provides some plot functions:
    - plane_plot
    - plot2d
    - plot3d
    


### algorithms

Currently two helpful tools are supported:
1.) QuadIndex, which creates a way of indexing similar to a quadtree, without creating a new datastructure.
    This is usefull if tools should be applied a in a certain way (like a checkerboard with 1x1m tiles)

2.) TrackDivider, which splits a point cloud along the first pca vector. This is continued until a maximal given size is reaches.
    This tool might be useful to localize operation in certain orders for lengthily shapes (like railway tracks, roads, pipes).
     
    algorithms.track_divider



#### PCA / SVD
---pending---


