**Report: Keyword Spotting in Historical Documents using Feature Engineering and Dynamic Time Warping**

**Abstract:**
This report details a "classic" feature engineering approach combined with Dynamic Time Warping (DTW) for keyword spotting in the George Washington historical document dataset. The system focuses on extracting robust visual features from word images and using DTW to measure similarity between these feature sequences. The methodology includes SVG-based word localization, image preprocessing (binarization, contrast enhancement), extraction of sliding window and Histogram of Oriented Gradients (HOG) features, and finally, sequence matching using DTW with a Sakoe-Chiba band. The system was evaluated on a subset of the validation data, achieving an average F1-score of 0.455 for Top-10 retrieval, demonstrating the viability of traditional feature-based methods for this task.

**1. Introduction**
The objective of this project is to develop a system for spotting keywords within images of historical documents. This task, often referred to as Query-by-Example (QbE) keyword spotting, aims to retrieve visually similar word images to a given query word image. Such systems are particularly valuable for collections where full Optical Character Recognition (OCR) is challenging or unreliable. This implementation employs a traditional pattern recognition pipeline, emphasizing handcrafted feature extraction and sequence alignment techniques, specifically Dynamic Time Warping (DTW), to compare word images. The dataset used is an excerpt from the George Washington Database, providing document images, word location SVGs, transcriptions, and keyword lists.

**2. Methodology**

The implemented system follows a multi-stage process:

**2.1. Data Preparation and Word Localization**
* **Dataset Paths:** Standardized paths were defined for accessing images (`PAGES_DIR`), SVG locations (`SVG_DIR`), transcriptions (`INDEX_TSV`), keywords (`KEYWORDS_TSV`), and output directories (`DEST_DIR`).
* **Transcription and Keyword Loading:**
    * Keywords from `KEYWORDS_TSV` were loaded. A `clean_word` function was applied to both the provided keywords and the transcriptions. This function converts text to lowercase and removes hyphens (e.g., "c-a-p-t-a-i-n" becomes "captain").
    * The main transcription index (`INDEX_TSV`) was processed. Only word instances whose cleaned transcription matched one of the cleaned keywords were retained in an `index` list. This `index` (containing `{'id': locator, 'keyword': cleaned_keyword}`) was primarily used for extracting and saving images of keyword instances.
* **Word Image Cropping:**
    * For each keyword instance in the `index`:
        * The corresponding full-page image (`.jpg`) and SVG annotation file (`.svg`) were identified.
        * The `bbox_from_svg` function, utilizing `xml.etree.ElementTree` and `svgpathtools.parse_path`, extracted the bounding box (`xmin, ymin, width, height`) of the word from the `<path>` element's `d` attribute in the SVG file.
        * The `svg_to_pixel_coords` function converted these SVG bounding box coordinates to pixel coordinates relative to the JPG image, accounting for potential scaling differences between SVG and image dimensions.
        * The word image was cropped from the full-page image (converted to grayscale using Pillow `Image.open().convert("L")`) using the calculated pixel bounding box.
        * Contrast enhancement (`ImageEnhance.Contrast(crop).enhance(2.0)`) was applied to the cropped grayscale image.
        * The processed cropped images were saved into a destination directory (`DEST_DIR`), organized into subfolders named after the `cleaned_keyword` text (e.g., `words/captain/270-09-01.png`). A total of 246 such keyword images were processed and saved.

**2.2. Image Preprocessing for Feature Extraction**
Before feature extraction, word images loaded for processing (from the `DEST_DIR`) undergo further preprocessing:
* **Binarization:** Images are converted to binary (black and white) using `cv2.adaptiveThreshold` (Gaussian method, `THRESH_BINARY_INV`). This step aims to create a clean representation of the ink strokes.
* **Morphological Closing:** A small closing operation (`cv2.morphologyEx` with a 2x2 kernel) is applied to the binary image to fill small holes and connect nearby components, potentially cleaning noise.

**2.3. Feature Extraction**
For each preprocessed (binary) word image, two types of features are extracted and combined to form a feature sequence:
* **Sliding Window Features (`extract_window_features`):**
    * The binary image is scanned column by column (window width of 1 pixel).
    * For each column (window), a set of 7 features is calculated:
        1.  Lower contour position (`lc`).
        2.  Upper contour position (`uc`).
        3.  Number of black-white transitions (`transitions`).
        4.  Fraction of black pixels in the window (`black_frac`).
        5.  Fraction of black pixels between `lc` and `uc` (`black_between`).
        6.  Gradient of lower contour (`grad_lc` - note: implementation seems to result in 0).
        7.  Gradient of upper contour (`grad_uc` - note: implementation seems to result in 0).
    * This results in a sequence of 7-dimensional feature vectors, where the sequence length corresponds to the width of the word image.
* **Histogram of Oriented Gradients (HOG) Features (`extract_hog_features`):**
    * The binary image is resized to a fixed size (64x64 pixels).
    * HOG features are computed using `skimage.feature.hog` with 8 orientations, 8x8 pixels per cell, and 1x1 cells per block. The `feature_vector=True` option is used, and the result is reshaped to `(-1, 8)`, treating each block's HOG features as a step in a sequence.
* **Concatenation and Normalization:**
    * The sliding window feature sequence and HOG feature sequence are concatenated horizontally after truncating to the minimum of their lengths.
    * `z_normalize_features`: The combined feature sequence is Z-normalized (standardized) column-wise. **Note:** The `StandardScaler` is instantiated and fit *within* this function for each sequence individually.
    * `pad_or_trim_sequence`: The normalized feature sequence is padded (by repeating the last feature vector) or trimmed to a fixed `target_len` of 100 steps.

**2.4. Similarity Measurement and Matching**
* **Dynamic Time Warping (DTW):** The similarity (or rather, distance) between two word image feature sequences is calculated using `dtw_distance_with_sakoe_chiba`.
    * This function implements DTW with a Sakoe-Chiba band constraint (`w=10` in the main loop, `DTW_BAND_RATIO` parameter was defined but not directly used in this function call) to speed up computation.
    * The cost between individual feature vectors in the sequences is the Euclidean norm (`np.linalg.norm`).
    * The final DTW distance is normalized by the sum of the lengths of the two sequences. A lower DTW distance indicates higher similarity.
* **Matching Process:**
    * The feature extraction and DTW matching are performed on `val_data`. This data is constructed by iterating through the saved keyword images in `DEST_DIR`, filtering them by `val_pages` (document IDs belonging to the validation set), and then extracting their features.
    * `val_data` is further filtered to only include keywords (labels) that appear at least 5 times, resulting in 22 unique word images being used for the DTW matching evaluation based on the progress bar (`22/22`).
    * For each word in this filtered `val_data` (acting as a query), its DTW distance to all other words in `val_data` is computed. Matches are sorted by this distance.

**2.5. Evaluation Metrics**
* Standard information retrieval metrics: Precision, Recall, and F1-score are calculated for Top-K retrieval (K=10).
* `tp` (true positives): Correctly retrieved instances of the query label within Top-K.
* `fp` (false positives): Incorrectly retrieved instances within Top-K.
* `fn` (false negatives): Instances of the query label not retrieved within Top-K but present in the dataset.

**3. Results and Discussion**

**3.1. Quantitative Results**
The system was evaluated by performing DTW matching on the filtered validation set. The reported average performance metrics for Top-10 retrieval are:
* **Average Precision:** 0.332
* **Average Recall:** 0.730
* **Average F1-score:** 0.455

A sample printout for individual queries (e.g., for "captain" and "letters") shows varying P/R/F1 scores:
* `304-12-03.png | Label: captain    | P: 0.10  R: 0.25  F1: 0.14`
* `301-07-06.png | Label: captain    | P: 0.30  R: 0.75  F1: 0.43`
* `303-02-01.png | Label: letters    | P: 0.50  R: 1.00  F1: 0.67`

The relatively high average recall suggests that true matches for a query (if they exist in the evaluated set) are often found, though not necessarily all within the top K. The precision indicates that a fair portion of the Top-10 results are indeed correct matches.

**3.2. Qualitative Results**
The `show_top_matches` function provides visual examples of the keyword spotting results.

*  For query "captain" (`304-12-03.png`), the top results include other instances of "captain" (green border) as well as visually similar but different words.
*  For query "letters" (`303-02-01.png`), several instances of "letters" are retrieved with high ranking.

These visualizations confirm that the DTW approach on the extracted feature sequences can successfully identify and rank visually similar word images. The green borders highlight correct retrievals based on the textual label.

**4. Discussion and Comparison**

This implementation represents a "classic" pipeline for keyword spotting, contrasting with modern deep learning approaches.
* **Strengths:**
    * Relies on well-understood image processing techniques and feature descriptors (contours, HOG).
    * DTW is effective for comparing sequences of varying lengths and handling non-linear distortions common in handwriting.
    * Does not require GPU for its core computations (though feature extraction can be parallelized).
    * The features are, to some extent, interpretable.
* **Limitations:**
    * **Feature Engineering:** The performance is highly dependent on the quality and robustness of the handcrafted features. Developing these features can be an iterative and time-consuming process. The `grad_lc` and `grad_uc` features in the sliding window appeared to be zero, which might indicate an issue or an area for refinement.
    * **`StandardScaler` Usage:** Applying `StandardScaler` by fitting it to each individual sequence during feature extraction (`z_normalize_features`) is not standard. Typically, the scaler should be fit on a representative training set and then used to transform all data (train, validation, test) consistently. This could impact the stability and comparability of feature vectors.
    * **Computational Cost of DTW:** While banded DTW is used, computing pairwise DTW distances for a large gallery of words can still be computationally expensive during query time. The evaluation loop (`DTW Matching: 100%|██████████| 22/22 [00:02<00:00, 10.98it/s]`) was performed on a very small filtered set (22 items). Scaling this to thousands of words would be significantly slower than embedding-based cosine similarity searches.
    * **Data Scope:** The initial `index` was filtered to only include keywords for image cropping. The `val_data` for DTW matching also seems to be built only from these keyword images (those present in `DEST_DIR` and `val_pages`). This means the evaluation might be assessing how well it distinguishes between different keyword instances, rather than finding keywords among a larger set including non-keywords, depending on how `val_data` is constructed and used. The exercise goal implies finding keywords within a broader document context.

**5. Conclusion**
This project successfully implemented a traditional feature-based keyword spotting system using a combination of image processing, sliding window features, HOG features, and Dynamic Time Warping. The system demonstrated its ability to retrieve visually similar word images, achieving an average F1-score of 0.455 on the evaluated validation subset. While the handcrafted feature engineering and DTW approach has its merits, particularly in interpretability and lower reliance on massive training datasets for the core algorithms, considerations around feature robustness, scaling of `StandardScaler`, and the computational cost of DTW for large-scale retrieval are important. The results provide a valuable benchmark for "classic" KWS techniques on this dataset.
