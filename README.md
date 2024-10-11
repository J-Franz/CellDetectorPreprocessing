# CellDetectorPreprocessing

### WORK in Progress - be careful reusing it

#### What this package does
To perform cell classification of immunofluorescent images on whole slide scans stored on an OMERO server this repository contains relevant helping preprocessing functions. 

#### Nuclei Detection
The basis of the CellDetector algorithm is nuclei detection which can be performed by any algorithm. Currently only the cellpose algorithm is promoted. The program will perform nuclei detection on the whole slide scan and upload a list of detected nuclei with the selected channel of interest to the omero server linked to the processed image.

#### Prepare lazy loading locally
Additionally to enable lazy loading of images e.g. using dask a local copy of the images will be stored as a zarr array.

#### Cumulative density function - calculation
Lastly a cumulative density function of the fluorescence channel for classification within the tissue ROI ("tissue_0") will be calculated and uploaded to the OMERO server as well.
