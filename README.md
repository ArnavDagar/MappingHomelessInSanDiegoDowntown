### Introduction

The Downtown San Diego Partnership (DSDP) conducts a monthly count of 10 regions in
San Diego downtown. The process is manual – from data collection to data tallying and tabulation.
While the paper maps have the geo spatial information of the counts, it is not digitized and mapped
on digital maps.

This project aims to use computer vision and machine learning methods to develop a
solution to generate GIS data from these paper maps for further visualization and analysis.

### Raw Data

```
Template Maps:
```
The Downtown San Diego Partnership provided the 10 template map images that they
use for counting. These are essentially empty maps that they print and use for the count every
month. These maps span different regions of downtown that DSSSDP serves – East Village
(EV), East Village South (EVS), Sherman Heights (SH), Golden Hills (GH), Gaslamp (GL),
Barrio Logan (BL), Cortez (TZ), City Center (CC) and Columbia (CO).

```
Data Maps:
```

I downloaded monthly count map data pdfs from 2018 to Jan 2025 from the DSDP
website and converted into jpgs. Here is a sample from the 2024 Dec data collection

# Figure 1 : Sample Data Maps – 10 for each month. Source: Downtown San Diego Partnership

DSDP capture different count information using symbols and digits. Below images show
samples of the count data. They are zoomed out significantly to be visible. Count information
includes individuals, tents and vehicles.

Individual counts are shown as either signle digit numbers or when 10 or more people are
obsevred, a circle is marked around the number to indicate a double digit number. Vechicels are
shown with a rectangle symbol with count of vehciles included in the symbol. Similarly tents are
shown with triangular symbols and have count of tents included in the symbol. Figures below
show some samples.


_Figure 2 : Zoomed Sample Single Digit count data from maps_

_Figure 3 : Zoomed Sample Double Digit count data from maps_

```
Figure 4 : Zoomed Sample Vehicle Count data from maps
```

```
Figure 5 : Zoomed Sample Tent count data from maps
```
### Preparing Template Maps – one time process.

We use template maps to identify key information present on the maps – legends, symbols,
tallied data, map region, 4 Ground Control Points (GCPs) in Image Pixel Coordinates (IPC),
Information Outside Map Neat Lines (IOMNL). I annotated 10 Template Maps (TMs) in
Roboflow with Bounding Boxes (BBs) for Map Title, 4 GCPs, IOMNL, Polygons for map,
Polygons for shade map region. A sample annotated map is shown below.


## Figure 7 : Steps in matching and aligning data maps with template maps

_Figure 6 : Template with Neat Lines (Purple), IOMNLs (Cyan), GCPs (4 intersections in red green, blue and orange) annotated
manually in roboflow_

```
Thus from roboflow I get the following 55 identifying classes corresponding to the 10
```
## template maps. These class identifiers are stores in a yaml file for use in the program.

nc: 55
names: ['CC', 'CV', 'Date-Field', 'Event-Field', 'Event-Field-2',
'GCP0_BL', 'GCP0_CC', 'GCP0_CO', 'GCP0_EV', 'GCP0_EVS', 'GCP0_GH',
'GCP0_GL', 'GCP0_M', 'GCP0_SH', 'GCP0_TZ', 'GCP1_BL', 'GCP1_CC',
'GCP1_CO', 'GCP1_EV', 'GCP1_EVS', 'GCP1_GH', 'GCP1_GL', 'GCP1_M',
'GCP1_SH', 'GCP1_TZ', 'GCP2_BL', 'GCP2_CC', 'GCP2_CO', 'GCP2_EV',
'GCP2_EVS', 'GCP2_GH', 'GCP2_GL', 'GCP2_M', 'GCP2_SH', 'GCP2_TZ',
'GCP3_BL', 'GCP3_CC', 'GCP3_CO', 'GCP3_EV', 'GCP3_EVS', 'GCP3_GH',
'GCP3_GL', 'GCP3_M', 'GCP3_SH', 'GCP3_TZ', 'IC', 'IV', 'Rain-Field',
'Shaded', 'TC', 'TV', 'Temperature-Field', 'Title', 'Total', 'map']

nC = number of classes
‘CC’, ‘CO’,’EV’,’EVS’, ‘TZ’, ‘BL’,’M’,’GL’,’GH’, ‘SH’ correspond to the 10 regions – City
Center, Columbia, East Village, East Village South, Cortez, Barrio Logan, Marina, Gaslamp,
Golden Hills, and Sherrman Heights respectively. There are 4 GCPs (GCP0-3) for each region.
Rest of the items correspond to different fields within the map.


For the Cortez template map, roboflow also generated the bounding box or polygon pixel
coordinates for each of the classes pertinent to the map. This is stored in a text file. The first
number is the class from the list above followed by the normalized pixel y,x coordinates.

The 4 GCPs are easily identifiable intersections on the map. I looked up their
corresponding latitude longitude coordinates in Google maps and created a xls file (Link to
Latest_gcp.xlx). Below is a sample of the information from the file. GCPs 0-3 are used for
converting image pixel coordinates into latitude longitude coordinates. -1, 4 are used to identify
the maximum area covered by the map. In the table Geo corresponds to one of 10 geo regions.
GCP id 0- 3 corresponds to 4 intersections on the map. -1 and 4 are used to identify the corners of
the rectangle defining the region. gcpY and gcpX are the latitude and longitude respectively.

```
Geo gcp_id gcp_address gcpY^ gcpX^
TZ - 1 5 - Ramp 32.72345280256031^ - 117.^
TZ 0 1st-Cedar 32.722008227696165^ - 117.^
TZ 1 9th-Cedar 32.72200934455702^ - 117.^
TZ 2 9th-Ash 32.71991000872872^ - 117.^
TZ 3 1st-Ash 32.71987664466721^ - 117.^
TZ 4 10th-A 32.71888013779527 - 117.
```

### Machine Learning models for count data

```
I have trained six multistage machine learning models in roboflow to read the count data and classify
it as Individual, Tent and Vehicle. Below steps, flowchart describes what the model do, what they were
trained on and the performance metrics of each of the models:
```
1. model_id: count-annotations-4/3 → Object detection (hand marked count data)
This model find bounding boxes for all of the hand marked counts – Individual Single Digit (SD),
Individual Double Digit (DD), Single Tent (ST), Multi Tent (MT). Single Vehicle (SV), Multi Vehicle
(MV). This stage returns the center coordinates of bounding box and its width and height (cx,cy,wx,wy)
2. model_id: digitclassification/2 → Single Digit Classification
This model is a digit classification model that identifies the image within the Single Digit bounding box
as 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
3. model_id : doubledigit/2 → Object detection (Digits)
A Double Digit Object Detection Model processes the image within the Double Digit bounding box and
produces bounding boxes for the Tens and Units digits. A digit classification model then classifies the
Tens an Unit digit images to 0-9 and the tens and units digits are combined to produce the count.
4. model_id: tent-classification/1 classifies the Tent bounding in three classes – Single Tent, Single
Digit Tent, Single Tent and Multi Digit Tent. The Single Tent bounding box is just a tent, while
others have digits marked in them.
5. tentdigits-sd/1 gives bounding box for the digit within the tent symbol
6. dd-tdd/1 gives bounding boxes for the Unit and Tens digit within the tent symbol

```
Below tables provide more details on the input, output, type of models, training data used and
performance metric of each of the models.
Sr
No.
```
```
Model ID Type Model Type mAP@50 Precision Recall Accuracy # Training
Items
```
1 count-annotations-4/3 Object
Detection

#### YOLOV12-X 92.4% 88.7% 88.6% N/A 12664

2 digitclassification/2 Classification ViT N/A N/A N/A 97.2% 6325

3 dd-tdd/1 Object
Detection

#### YOLOV12-X 96.1% 93.5% 89.0% N/A 686

4 tentdigits-sd/1 Object
Detection

#### YOLOV12-X 92.7% 89.9% 88.9% N/A 1104

5 doubledigit/2 Object
Detection

#### YOLOV12-X 94.7% 91.3% 92.8% N/A 348

6 tent-classification/1 Classification ViT N/A N/A N/A 91.4% 3020

```
Table 1 : Machine Learning models and performance metrics
```

Sr. No. Model ID Input Output Sample Input and Annotations

1 count-annotations-4/3 (^) Image of
Data
Map of
A
Region
Bounding Boxes of
each count -
Individual Single
Digit (SD),
Individual Double
Digit (DD), Single
Tent (ST), Multi
Tent (MT). Single
Vehicle (SV), Multi
Vehicle (MV).
2 digitclassification/2 Image of
A Single
Digit
Digit Value
0,1,2,3,4,5,6,7,8,
3 dd-tdd/1 (^) Image of
A
Double-
Digit
Tent
2 Bounding Boxes
of Digits (Ten’s
and Unit’s)
4 tentdigits-sd/1 Image of
A Single-
Digit
Tent
1 Bonding Box of
Digit
5 doubledigit/2 Image of
A Double
Digit
Number
2 Bounding Boxes
of Digits (Ten’s
and Unit’s)
6 tent-classification/1 (^) Image of
A Tent
One of 3 Classes:
Single Tent,
Single-Digit Tent,
Double-Digit Tent
_Table 2 : Machine learning with inputs and outputs_


Below flow chart shows how an input data map is processed with these 6 machine learning models.
Starting with a data map aligned to its template, we run object identification models to identify relevant
boundaries boxes and then either run classification or object detection models on relevant bounding box
images till we get a count information and one of 3 classes – Individual, Tent, Vehicle. In the flow char
below the object detections and classification steps are color coded.

_Figure 7 : Flowchart for processing a data map_

### Program to process each month’s data maps

This section describes the python script used to process each month’s data maps. The
script assumes the following directory structure (root directories and subdirectories).

#### PROJECT DIRECTORY STRUCTURE

/content/drive/MyDrive/
│

```
Aligned Data
Map
```
```
Count
Annotations
```
```
SD
```
```
Digit
Classification
```
```
I, #
```
```
DD
```
```
DD-Model
```
```
Ten
```
```
Digit
Classification
```
```
I, 10 * #
```
```
Unit
```
```
Digit
Classification
```
```
I, #
```
```
ST
```
```
Digit
Classification
```
```
T, #
```
```
MT
```
```
DD-T-Model
```
```
Ten
```
```
Digit
Classification
```
```
T, 10 * #
```
```
Unit
```
```
Digit
Classification
```
```
T, #
```
```
SV
```
```
V, 1
```
```
MV
```
```
V, 2
```

├── HomelessMappingFolder/
│ ├── Config/
│ │ ├── T_geo.txt # Template configuration text files
│ │ └── T_geo.yaml # Template configuration YAML files
│ │
│ ├── Results/
│ │ ├── yyyyMMM_homeless_count_data.csv # Processed count data
│ │ ├── yyyyMMMgeobbInMap.jpg # Bounding box visualizations per region
│ │ └── yyyyMMMDT.jpg # Downtown consolidated map
│ │
│ ├── Templates/
│ │ └── T_geo.jpg # Template images for each geographic region
│ │
│ └── LatestGCP.xlsx # Ground Control Points spreadsheet
│
└── Maps/
└── yyyy/ # Year folder (e.g., 2025)
└── yyyy_MMM/ # Month folder (e.g., 2025_MAY)
├── [Data Downloaded from DSDP website]
└── geo.jpg # Converted PDF images for each region
# (where geo = EV, EVS, GL, M, TZ, CO, BL, GH, CC, SH)

NOTES:

- yyyy = 4-digit year (e.g., 2025)
- MMM = 3-letter month abbreviation (e.g., MAY, JUN, JUL)
- geo = Geographic region code (10 regions total):
- EV: East Village
- EVS: East Village South
- GL: Gaslamp
- M: Marina
- TZ: Cortez
- CO: Colombia
- BL: Barrio Logan
- GH: Golden Hill
- CC: City Center
- SH: Sherman Heights

FILE DESCRIPTIONS:

- Config files: Template matching and annotation configurations
- Results: Processed data outputs and visualization maps
- Templates: Reference images for image alignment
- LatestGCP.xlsx: Geographic coordinate reference points
- Maps: Source imagery organized by year and month


The HomelessMappingFolder has a Config, Results and Template folder. The Config folder
has yaml and txt files for each geo region and were described in earlier sections. The Results
folder is where all the results will be stored after the processing. The above notation describes
the format of the results. The Template folder stores the template map images. This folder does
NOT need to be modified.

The Maps folder stores the raw data downloaded from the DSDP website using the above
notation. As a new month data is available, we need to download and generate 10 jpgs from the
pdf file. The pdf is available on the DSDP website, and an online tool is used to separate each
page of the pdf into jpg image files.

Below is a snapshot of all of the code and areas to modify for processing a given month’s
data

Cell 1:

This code installs and imports a Python environment combining geospatial data processing
libraries (geopandas, contextily, mapclassify) with computer vision packages (cv2, imutils),
machine learning inference capabilities (inference_sdk), and data manipulation tools (matplotlib,
numpy, pandas, pyxlsb). The imports establish functionality for reading/writing geographic data
formats, rendering web map tiles as basemaps, performing image processing operations,
executing ML model inference, handling various file formats including Excel binary files, and
creating visualizations with geographic coordinate system support and annotation capabilities
through matplotlib.offsetbox.

Cell 2:

This code implements a computer vision pipeline using Roboflow's inference SDK to
detect and classify objects in images, specifically extracting numerical values from different
object categories. The get_all_bounding_boxes function performs object detection using


configurable confidence thresholds and returns bounding box coordinates with class labels, while
get_classification provides image classification results. The pipeline processes six object types:
single-digit (SD) and double-digit (DD) numbers using digit classification models, single-tent
(ST) and multi-tent (MT) objects requiring tent classification and digit extraction, and single-
vehicle (SV) and multi-vehicle (MV) categories with hardcoded return values. The analyze_bb
function serves as the main processor that crops detected regions from the source image, applies
the appropriate classification model based on object type, and returns structured data containing
coordinates, dimensions, object type (Individual/Tent/Vehicle), and extracted numerical count.



Cell 3:

This code implements a computer vision pipeline for georeferencing and spatial analysis
of images using SIFT feature detection and homography transformations. The match_images
function performs keypoint detection and matching between template and target images using
SIFT descriptors with ratio testing, while align_imgs handles image alignment through
perspective warping based on computed homography matrices. The process_sections function
serves as the main processor that combines image alignment, YAML annotation parsing, ground
control point (GCP) extraction, and coordinate transformation by matching image pixel
coordinates to real-world coordinates using RANSAC-based homography estimation. Additional
utility functions include bb_within_poly for polygon-based bounding box filtering,
generate_poly_mask for creating binary masks from polygon vertices, draw_marker for
visualization overlays, and add_image_marker for embedding images as markers in matplotlib


plots, collectively enabling the transformation of detected objects from image coordinates to
georeferenced coordinate systems.



Cell 4:

This code implements a comprehensive geospatial analysis pipeline that processes
multiple geographic regions by loading ground control points from Excel files, performing image
alignment and object detection on images of paper maps, and transforming detected objects from
pixel coordinates to real-world geographic coordinates using homography matrices. The pipeline
iterates through predefined geographic regions, applies the process_sections and analyze_bb
functions to extract and classify bounding boxes, performs perspective transformation using
OpenCV to convert image coordinates to latitude/longitude pairs, and generates annotated output
images with bounding box overlays and classification labels. The results are systematically
stored in a structured DataFrame containing spatial coordinates, object classifications,
confidence scores, and metadata, while simultaneously creating georeferenced visualizations
using GeoPandas and contextily basemaps that display detected objects as image markers
overlaid on web map tiles, with all outputs saved as CSV files and georeferenced map images for
spatial analysis and reporting.

**In this cell, the month (mm), year (yy), path need to be modified corresponding to the data
being processed.**




### Sample Data Generated

```
Here we look at some of the data generated through the processing of individual maps.
```
1. Aligned to the template of the image to the data map with bounding boxes and machine learning
    count outputs. Below is a sample for the East Village South map from January 2025.

Some observations:

1. Bounding boxes around hand marked symbols are drawn in green
2. Above each bounding box, the number in blue represents a sorted count ID (seen in spreadsheet
    below)
3. Below each bounding box, there letter and number in red represent the type and the count of the
    hand marked symbol. I = Individual, T = Tent, V = Vehicle
4. This image can be used for manual inspection and performance evaluation of the performance of
    the machine leaning models used
       a. Most of the bounding boxes have been identified correctly, but there are some false
          negatives and incorrect identifications
       b. The “1” at the intersection of J and 15th St. was missed. This could be added manually
       c. The 30 individuals at the intersection of Imperial and 17th St is misidentified as 50
          individuals. This could be corrected manually with inspection of each of the datasets. The
          map and IDs provide a natural way of doing it.

_Figure 8 : East Village South Datamap processed using the machine learning models_


2. Below is a consolidated image for each month that takes the hand marked symbols and places
    them at the appropriate location on a map (created with the geopandas and contextily python
    libraries)

_Figure 9 : January 2025 maps processed and combined, transporting handmarked symbols onto maps_


3. Below is a csv file created that consolidates data for all of the 10 regions into one file for one
    month into one file. This csv file is used to create ArcGIS maps.

Year, Month: Date of data
Region: Location where data was collected
ID: Manual number associated with the count
BBx, BBy, BBw, BBh: Bounding box center pixel coordinates and dimensions
Longitude, Latitutde: Real world location of bounding box
Confidence: Likelihood of correct representation of the bounding box by the machine learning model
Type: Individual, Tent, Vehicle
Count: Number of individuals, tents, or vehicles
Valid: Flag that can be set to determine accuracy through manual inspection. Represents correct data
Missed: Flag that can be set to determine accuracy through manual inspection. Represents missed data

4. Using this csv file, we can create an ArcGIS map to visualize the data


_Figure 10 : ArcGIS map showing homeless in January 2025 in San Diego Downtown_

### Conclusion

This project presents a fully automated, end-to-end pipeline for digitizing and georeferencing
observational DSDP homelessness survey data using a combination of deep learning, computer vision,
and geospatial analysis techniques. The system extracts hand-annotated count data from legacy paper
maps, classifies the observed entities into three semantic classes (Individual, Tent, Vehicle), and
transforms their image-space coordinates into real-world geographic coordinates using homography
matrices constructed from manually annotated ground control points (GCPs).

The approach leverages a modular machine learning architecture composed of six models, combining
object detection (YOLOv12-X) and classification (ViT-based) networks. These models, trained on over
20,000 manually labeled samples, demonstrate strong performance in accurately detecting and
interpreting symbolic annotations under varying imaging conditions and handwriting styles. The image
registration pipeline uses SIFT keypoint detection with Lowe’s ratio test and RANSAC-based
homography estimation for robust template-data alignment, enabling consistent geospatial
transformations across diverse map samples.

Post-processing routines convert pixel coordinates to WGS84 (EPSG:4326) geographic coordinates
using transformation matrices derived from annotated GCPs. The output includes structured CSV files
with spatial metadata, classification confidence scores, and validation flags, alongside visualization
artifacts generated using GeoPandas and contextily in Web Mercator (EPSG:3857) for cartographic
rendering. These outputs are further integrated into ArcGIS workflows to support longitudinal
spatiotemporal analysis of homelessness distributions.


Future development will focus on improving the end data quality by manual inspection and updating
the valid and missed columns within all the monthly csvs.


