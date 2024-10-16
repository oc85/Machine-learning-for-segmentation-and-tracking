# Machine-learning-for-segmentation-and-tracking


# Project Overview

This project involves using **Detectron2** for **image segmentation** on datasets containing amoebas and yeast. It includes data preprocessing, model configuration, and segmentation processing. The results are visualized and saved for further analysis.

## Key Components


- **Libraries and Setup**: The code imports necessary libraries such as **OpenCV**, **NumPy**, **PyTorch**, and **Detectron2**. It configures logging and sets up **Detectron2** for image segmentation tasks.


- **Data Preprocessing**: The script loads and preprocesses datasets by resizing images and matching histograms to reference images. This ensures that input data is consistent for model processing.


- **Model Configuration**: Two separate configurations for **Detectron2** are used. The first configuration is for initial dataset processing, and the second is for yeast segmentation.


- **Segmentation Process**: The segmentation is performed using **DefaultPredictor** from **Detectron2**. It processes each image, applies masks to identify regions of interest, and performs amoeba and yeast segmentation.


- **Data Cleaning and Risk Analysis**: Post-segmentation, the code cleans datasets by removing small regions and performs a risk analysis using morphological operations to highlight critical areas.


- **Visualization and Output**: The results are visualized using **Matplotlib** and saved as images. The code handles multiple regions of interest and generates labeled output for further analysis.

## Usage


1. **Setup Environment**: Ensure all required libraries are installed, and paths to data sources are correctly specified.
2. **Run Script**: Execute the script to perform segmentation and analysis on the dataset.

3. **Review Output**: Check the processed images and results saved in the specified output directory.

This project provides a comprehensive approach to image segmentation using state-of-the-art deep learning techniques with **Detectron2**.

![e4ea0d_0472d909880147b9b9a19e06307a0a59~mv2](https://github.com/user-attachments/assets/f158a8e6-7b43-43b8-b485-b292a2fdc5a3)



