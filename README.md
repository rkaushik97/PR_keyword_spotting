## Key Word Spotting: Algorithm and Results Summary

This report summarizes the algorithm and results of the Key Word Spotting notebook **"KeyWordSpotting.ipynb"**. The primary goal is to identify handwritten keywords in images.

### Algorithm

The core of the keyword spotting system relies on feature extraction and matching.

1.  **Data Preparation and Preprocessing:**
    * The system utilizes datasets like IAM, George Washington, Bentham, and Modern.
    * It involves parsing XML files for transcriptions and word segmentations.
    * Images are typically converted to grayscale.
    * **Image Augmentation (Implicit):** While not explicitly detailed as a separate pre-processing step in the provided snippets, image manipulation functions (like `ImageEnhance` and `cv2` operations) are imported, suggesting potential use for augmentation or normalization.
    * **Stroke Width Transform (SWT) and Connected Components (CC):** Functions related to SWT and CC are defined, indicating their use in segmenting text or pre-processing images to isolate textual regions. `swt`, `connected_components`, `find_letters`, `crop_image_cc` point to a sophisticated segmentation pipeline.
    * **Synthetic Data Generation:** The notebook includes code to generate synthetic keyword images using `svgpathtools` to parse SVG paths of characters and assemble them into words. This is likely used to augment the training set or create specific test cases.

2.  **Feature Extraction:**
    * **Histogram of Oriented Gradients (HOG):** The primary feature descriptor used is HOG. The `skimage.feature.hog` function is employed to extract these features from the segmented word images. Parameters such as `orientations`, `pixels_per_cell`, and `cells_per_block` are crucial for HOG feature computation.
    * **Descriptor Storage:** The extracted HOG features, along with their corresponding labels (keywords) and source file names, are stored. The notebook snippet shows features being collected in a `defaultdict(list)`.

3.  **Keyword Spotting (Matching):**
    * **Querying:** The system takes a query keyword (either from the dataset or synthetically generated).
    * **Feature Comparison:** The HOG features of the query image are compared against the HOG features of all images in the dataset.
    * **Distance Metric:** Euclidean distance (`np.linalg.norm`) is used to measure the similarity between HOG feature vectors. A lower distance indicates higher similarity.
    * **Ranking:** For a given query, all dataset images are ranked based on their distance to the query image.

### Results

The results section of the notebook focuses on evaluating the performance of the keyword spotting algorithm.

1.  **Evaluation Metric (Implicit):** While a specific metric like Mean Average Precision (mAP) or precision-recall curves are not explicitly named in the provided visualization code, the system evaluates performance by:
    * Retrieving the top-k most similar images from the dataset for a given query keyword.
    * Visually inspecting these top-k matches.

2.  **Visualization of Top Matches:**
    * The `show_top_matches` function visualizes the results. For a selected query image (keyword), it displays the query itself and the `top_k` images from the dataset that have the smallest Euclidean distance in the HOG feature space.
    * **Correct vs. Incorrect Matches:** The visualization uses border colors (green for a correct match where the label of the retrieved image matches the query label, and red for an incorrect match) to provide an immediate visual assessment of the retrieval quality.
    * The distance value is also displayed for each matched image.
    * The notebook snippet shows a plan to display results for 25 different query words, showing the top 10 matches for each.

**Summary of Performance (Inferred from Visualization Approach):**
The performance is primarily assessed qualitatively through visual inspection of the ranked retrieval lists. The border colors help in quickly identifying how many of the top results are actual instances of the query keyword. The distances provide a quantitative measure of similarity, where lower values are better. The notebook aims to show if the algorithm can consistently rank true matches higher than non-matches.



### Team Members

1. Kaushik Raghupathruni
2. Nathan Wegmann
3. Raunak Pillai
4. Richmond Djwerter
5. Yi-Shiun Alan Wu
