# Unraveling the cytoskeletal architecture of cancer cells: a novel computational approach to predict cell fate - CODE

This code is divided in 8 chapters:

  1. **Dataset**: Inspect the dataset with RGB and 2D deconvoluted cytoskeleton/nuclei images.
  2. **Nuclei Preprocessing**: Adjust the parameters for a given image to segment and preprocess nuclei and visualize the results.
  3. **Cytoskeleton Preprocessing**: Adjust the parameters for a given image to indentify individual filaments and visualize the results.
  4. **Cell Segmentation**: Use ROIpoly to draw a polygonal mask around the cell of interest.
  5. **Processing**: Extract line segments and prepare cells for feature extraction.
  6. **Feature Extraction**: Extract the desired features (DCFs, LSFs or CNFs).
  7. **Results Analysis**: Compare feature values between single or a population of cells.

# Additional notes:

   - Use the ```environment.yaml``` file to setup a dedicated environment with the required packages to run the code.
 
 
 
