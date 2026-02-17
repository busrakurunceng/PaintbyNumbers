# Paint by Numbers

A modular image processing pipeline that transforms any photograph into a paint-by-numbers template. The system analyzes pixel-level color data, reduces millions of colors down to a manageable palette using K-Means clustering, segments the image into paintable regions, and generates a numbered canvas ready for painting.

## How It Works

```
Photo → Bilateral Filter → K-Means Clustering → Region Detection → Contour Extraction → Number Placement → Canvas + Legend
```

The pipeline runs in 10 sequential steps:

| Step | Description | Module |
|------|-------------|--------|
| 1 | Load image | `image_io.py` |
| 2 | Smooth with bilateral filter (preserve edges, reduce noise) | `preprocessing.py` |
| 3 | Flatten pixels to 2D matrix for clustering | `pixel_analysis.py` |
| 4 | K-Means clustering to reduce colors to K groups | `clustering.py` |
| 5 | Segment image + create label map | `segmentation.py` |
| 6 | Name each color using LAB color space distance | `color_categorization.py` |
| 7 | Detect connected regions + remove small noise areas | `region_detection.py` |
| 8 | Extract contours and draw boundary lines | `contour_extraction.py` |
| 9 | Calculate centroids and place color numbers | `number_placement.py` |
| 10 | Render reference image and color legend | `rendering.py` |

## Outputs

| File | Description |
|------|-------------|
| `canvas_numbered.png` | White canvas with outlines and numbers (painting template) |
| `reference.png` | Colored regions with outlines and numbers (painting guide) |
| `color_legend.png` | Number-to-color mapping table |
| `canvas_outline.png` | White canvas with outlines only (no numbers) |
| `comparison_cleanup.png` | Before/after noise removal comparison |
| `segmented.png` | Posterized image after K-Means |
| `cleaned_segmented.png` | Posterized image after small region cleanup |
| `smoothed.png` | Image after bilateral filter |

## Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your image in the `data/` folder
2. Update `IMAGE_PATH` in `config.py`
3. Run the pipeline:

```bash
python main.py
```

4. Check the `outputs/` folder for results

## Configuration

All parameters are centralized in `config.py`:

```python
K_CLUSTERS = 16           # Number of colors (paint tubes)
MIN_REGION_AREA = 500     # Regions smaller than this get merged
LINE_COLOR = (80, 80, 80) # Dark gray lines (paint can cover them)
LINE_THICKNESS = 2        # Outline width in pixels
MIN_AREA_FOR_LABEL = 800  # Regions smaller than this get no number
```

## Project Structure

```
PaintbyNumbers/
├── main.py                         # Pipeline orchestrator
├── config.py                       # All configurable parameters
├── requirements.txt                # Python dependencies
├── src/
│   ├── image_io.py                 # Image load/save with BGR-RGB conversion
│   ├── preprocessing.py            # Bilateral filter smoothing
│   ├── pixel_analysis.py           # Pixel matrix extraction
│   ├── clustering.py               # K-Means color clustering
│   ├── segmentation.py             # Image segmentation + label map
│   ├── color_categorization.py     # LAB-based color naming
│   ├── region_detection.py         # Connected components + noise removal
│   ├── contour_extraction.py       # Boundary line extraction
│   ├── number_placement.py         # Centroid calculation + labeling
│   ├── rendering.py                # Reference image + color legend
│   └── visualization.py            # Comparison visualizations
├── data/                           # Input images (git-ignored)
├── outputs/                        # Generated results (git-ignored)
└── docs/                           # Project documentation
```

## Tech Stack

- **OpenCV** — Image I/O, bilateral filter, connected components, contours, drawing
- **NumPy** — Pixel matrix operations
- **scikit-learn** — K-Means clustering
- **matplotlib** — Color legend and comparison visualizations

## Key Design Decisions

- **Bilateral filter over Gaussian blur** — Preserves edges (region boundaries stay sharp) while smoothing flat areas
- **LAB color space for naming** — LAB is perceptually uniform; Euclidean distance in LAB correlates with how humans perceive color difference
- **Dark gray lines instead of black** — Allows paint to cover the outlines more easily
- **Dynamic font scaling** — Number size adapts to region area; tiny regions get no label at all
- **Connected component analysis** — Same color at different locations becomes separate paintable regions
