# CellDetectorPreprocessing

### Work in Progress - Use with Caution

#### What This Package Does
This repository provides preprocessing functions essential for performing cell classification on immunofluorescent whole slide images (WSI) stored on an OMERO server.

#### Nuclei Detection
The core of the CellDetector algorithm is nuclei detection, which can be performed using any suitable algorithm. Currently only the **Cellpose** algorithm is supported. The program performs nuclei detection on the whole slide scan and uploads a list of detected nuclei to the OMERO server and links the list to the processed image.

#### Prepare Lazy Loading Locally
To enable lazy loading of images (e.g., using Dask), the program stores a local copy of the images as a Zarr array. This approach facilitates efficient data handling and processing during training and prediction of the CellDetector algorithm.

#### Cumulative Density Function - Calculation
Lastly, the program calculates a cumulative density function of the fluorescence channel within the tissue ROI ("tissue_0") for classification purposes. This data is then uploaded to the OMERO server.


## Acknowledgements

The authors would like to thank the Federal Ministry of Education and Research
and the state governments (www.nhr-verein.de/unsere-partner) for supporting this
work/project as part of the joint funding of National High Performance Computing
(NHR).
