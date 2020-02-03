# Optical Flow
Example of a basic Optical Flow application.


Usage
====

In the example, there are two type of Optical Flow algorithms implemented. 

1. Sparse (Lucas Kanade)
2. Dense (Farneback)

To test each usolated implementation, run:

#### Sparse
```
python app_sparse.py
```

#### Dense
```
python app_dense.py
```

Alternatively, you can run the standalone application containing both implementations and argument parser. 

```
python app_optical_flow --input videos\cars.mp4 --mode sparse
```

The will yield the following output in video form. Press ESC key to close.

<a href="https://raw.githubusercontent.com/valbertoenoc/optical_flow/images/sparse.png"><img src="https://github.com/valbertoenoc/optical_flow/blob/master/images/sparse.png" width=320></a>


```
python app_optical_flow --input videos\cars.mp4 --mode dense
```

<a href="https://raw.githubusercontent.com/valbertoenoc/optical_flow/images/dense.png"><img src="https://raw.githubusercontent.com/valbertoenoc/optical_flow/images/dense_flowgrid.png" align="left" width="128"></a>

  

<a href="https://raw.githubusercontent.com/valbertoenoc/optical_flow/images/dense_colorcoded.png"><img src="https://raw.githubusercontent.com/valbertoenoc/optical_flow/images/dense_colorcoded.png" align="left" width="128"></a>

