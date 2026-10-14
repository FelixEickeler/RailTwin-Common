## RailTwin-Common
### Setup
RailTwin-Common (rtc) is the package `railtwin_common` (Python >= 3.9, numpy 1.22 - 2.x, pandas 1.4 - 2.x).
Use it as git submodule named `railtwin_common`, or install it:
```bash
pip install -e .                    # IO only: numpy, pandas, plyfile, laspy, pyarrow
pip install -e ".[algorithms,plot]" # also scipy / scikit-learn and matplotlib / plotly
python -m pytest                    # tests
```
```python
from railtwin_common.common.io import read, write
```

### Common

#### IO - module
All data will be read to a pandas dataframe with appropriate naming (see common/io/header.py). Known
properties get their internal name and type, other properties are kept as they are.
This is a unified interface that can be used in your programming !

The writers never change the dataframe they are given. Coordinates keep their type unless `xyz_accuracy` is set
(e.g. `"<f4"`): float32 is not precise enough for georeferenced coordinates, use `local=True` (centred, the shift
is stored as `attrs["metadata"]["offset"]`) or keep float64.

Normally operations should be carried out on the data to normalize it. See the options in IOOperations and header.py !

Currently, the RailTwin-Common supports the following formats for reading and writing. 

| #   | Fileformat | read                    | write               |
|-----|---------|-------------------------|---------------------|
| 0   | *.ply | :heavy_check_mark:      | :heavy_check_mark:  |
| 1   | *.feather | :heavy_check_mark:      | :heavy_check_mark:  |
| 2   | *.las / *.laz | :heavy_check_mark:(1)   | :x: |
| 3   | *.xyz | :heavy_check_mark:(1,2) | :x: |

(1) point formats 0-5, no metadata apart from the offset
(2) Format must be register, but reader is available

It is possible to register additional reader & writer by adding them to the
common.io.registered_reader or the common.io.registered_writer dictionary. 
```
common.io.registered_reader[".my_format"] = your_custom_read_function
common.io.registered_reader[".xyz"] = common.io.writer.csv.create_csv_reader(HeliosFormat)  
```

#### Logger
RailTwin-Common logs with Python's `logging` ("RT-Logger", to stderr). The logger can be reused in any part of
your code by:
```python
logger = railtwin_common.common.logger.RailTwinLogger.create()   # a logging.Logger
RailTwinLogger.register_file_logger(Path("logs"))                 # optional: also into logs/<date>-RailTwin.log
```

### Labels
Point clouds keep fine component labels; a mapping turns them into the classes a task needs, at load time.
`labels/obm_mapping.json` (from RailTwin-Virtualizer `ecppm2022_simple_mapping.json`, component ids made unique) has
64 `component_labels`, the `systems` they belong to, `system_labels`, and the names used in design models
(`synthetic`) and annotations (`physical`).
```python
from railtwin_common.labels import LabelMapping
mapping = LabelMapping.load()                                    # labels/obm_mapping.json, or any path
system_ids = mapping.to_systems(component_ids, default="instance") # numpy lookup; default for unmapped components
mapping.component_id("Schwelle", "synthetic")                     # id of a source name
```
Without `default`, `to_systems` raises and lists the components that have no system.

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


