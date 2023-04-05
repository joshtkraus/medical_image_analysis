# medical_image_analysis
Medical Image Analysis for the NIH Chest X-ray Data Set

## NIH Chest X-ray Dataset: Array Format

*See code in process_data.py*

<ins> NumPy Array File (.npy)

Data are stored in NumPy array file (.npy) for reduced storage costs (>50% reduction in size compared to .csv & .txt files). Writing arrays to each file proved to be simpler using this approach and addressed size constraints when reading in X- ray data. Moreover, loading the data matrix in this format reduced computational cost and time, making it a reasonable choice for this context. To read the data in, the following code can be used (see [here](https://numpy.org/devdocs/reference/generated/numpy.load.html)):

`np.load(file, allow_pickle=True)`

<ins> data.npy

The file data.npy contains a 112,120 Chest X-rays. Each image is given as a size of 64 X 64, hence represented as a vector in	R<sup>4096</sup>. This vector is stored as a row in the file, while each column corresponds to a unique X-ray image.

<ins> labels.npy

The file labels.npy contains the disease classifications for the 112,120 Chest X-rays. The array is one-dimensional and contains the disease classification corresponding to each unique X-ray image.
