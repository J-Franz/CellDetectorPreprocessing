CODEPATH=/share/Work/Neuropathologie/CellDetector/code/CellDetectorPreprocessing/
ANALYSISPATH=/share/Work/Neuropathologie/CellDetectorPlayground/
IMAGENAME=myimage_cellpose
PYTHONINDOCKER=/miniconda3/envs/cellpose/bin/python
PYTHONPROGRAM=/CellDetectorPreprocessing/Code/main_zarr.py
PASSWORD=ome
IMAGEID=253891
GPU=False

docker run -v $CODEPATH:/CellDetectorPreprocessing -v $ANALYSISPATH:/Analysis/ -it $IMAGENAME $PYTHONINDOCKER $PYTHONPROGRAM $PASSWORD $IMAGEID /Analysis/ $GPU
