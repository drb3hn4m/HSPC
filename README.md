## Spectral Processing Challenge (HSP C)

### Intro

#### Scenario 1
There is an excel workbook which contains 2 sheets. Each sheet acts as a standalone dataset. You can find them in the [data](data/) folder and are required to use both of these datasets.

Each dataset contains **161** columns, including:

- Unique identifier column: ***Sample ID***
- **11** Metadata columns
- **1** Response Variable column (***Total Organic Carbon [TOC]***)
- One ***Background*** indicator column
- **146** spectral columns (wavelengths spanning from *1350nm* ~ *2550nm* expressed in raw intensities)

Each row can be of two types: *Sample Scan* or *Background Scan* and can be identified using the *Background* column. Each sample has a repeat scan of 10 counts and can be identified using *ScanIndex*.

The data associated with the dependent variable and spectral data are required to be transformed to be suitable inputs for subsequent modeling of the dependent variables using the spectral data. You are expected to work with data from both datasets, between the wavelength range of **1400nm ~ 2400nm** and a granularity of **1nm**.

- Perform general EDA on the given datasets and present your findings.
- The following is to be performed in a sequence to convert raw intensities to absorbance:

> All downstream tasks operate only on the absorbance spectra for any modelling tasks

  1. Convert raw intensity to absorbance by dividing sample intensity with its corresponding background intensity.
  ```math
  \displaylines{
    \text{Given}\\ \text{Spectral Intensity Matrix: }\underset{N \times M}{\mathrm{SX}}|\text{ N samples, M wavelengths} \\ \text{ \& Background Spectral Intensity Matrix} \underset{N \times M}{\mathrm{BX}}\\
    A_{xs_i|W_1...W_j} = \frac{I_{xs_i|W_1...W_j}}{I_{xb|W_1...W_j}}\\
    \text{Where}\\
    xs_i: i^{th}\text{ Sample }\|\ i = 1...M\\
    xs_i: i^{th}\text{ Background }\|\ i = 1...M\\
    w_j: j^{th}\text{ Wavelength }\|\ j = 1...N\\
    {[ A | I ]}_{xs_i|W_j}: \text{Absorbance or Intensity of }i^{th}\text{ sample at }j^{th}\text{ Wavelength }
  }
  ```
  2. Inverse and apply logarithm (base 10)
  ```math
  \displaylines{
    AR_{xs_i|W_1...W_j} = \frac{1}{A_{xs_i|W_1...W_j}}\\
    AL_{xs_i|W_1...W_j} = \log{_{10}}{AR_{xs_i|W_1...W_j}}
  }
  ```
  3. Apply a smoothing technique to smooth the absorbance spectrum
  4. Normalize the smoothed absorbance using any normalization methods
- Detect spectral outliers and report their Sample ID's and ScanIndex. There is no restriction on the type of outlier analysis to perform. The preformance would be assesed based on the general classification metrics.
  5. Use any robust spectral outlier removal technique to identify and remove outliers (also report the outliers in a separate file)
  6. Build a predictive model (Eg. linear regression, partial least square, random forest, Neural Networks etc.) for each of the dependent variables, presenting suitable metrics for assessing the performance of the predictive model. Additionally, perform comparative analysis with a baseline predictor (be sure to implement hyperparam tuning strategies and reasoning).
> P.S. You can choose to add any other preprocessing steps as you see fit with an explanation or reasoning behind those additions.

You can choose to present your work as standalone scripts or as a jupyter notebook or any other pertinent technology.


### Appendix
#### Stat
- [Smoothing](https://en.wikipedia.org/wiki/Smoothing) In statistics and image processing, to **smooth** a data set is to create an approximating function that attempts to capture important patterns in the data.
- [Interpolation](https://en.wikipedia.org/wiki/Interpolation) In the mathematical field of numerical analysis, interpolation is a type of estimation, a method of constructing (finding) new data points based on the range of a discrete set of known data points.
- [Principal component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) is a popular technique for analyzing large datasets containing a high number of dimensions/features per observation.
- [Normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)) In the simplest cases, normalization of ratings means adjusting values measured on different scales to a notionally common scale.
- [chemometrics](https://chemometrics.readthedocs.io/en/stable/) is a free and open source library for visualization, modeling, and prediction of multivariate data.
- [NirPy](https://nirpyresearch.com/) Research is an educational space dedicated to Python chemometrics, where we take data science concepts down to the language of spectroscopic science.

#### For Data Apps:
- [Anvil](https://anvil.works/) Build web apps with nothing but Python.
- [Streamlit](https://streamlit.io/) turns data scripts into shareable web apps in minutes.
- [Gradio](https://gradio.app/) Build & Share Delightful Machine Learning Apps.
- [Dash](https://dash.plotly.com/) is the original low-code framework for rapidly building data apps in Python.

#### CLI:
- [Python Fire](https://github.com/google/python-fire) is a library for automatically generating command line interfaces (CLIs) from absolutely any Python object.
- [Click](https://github.com/pallets/click) is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary.
- [Cement](https://builtoncement.com/) CLI application framework for Python.
- [PyQt5](https://riverbankcomputing.com/software/pyqt/intro) is more than a GUI toolkit. It includes abstractions of network sockets, threads, Unicode, regular expressions, SQL databases, SVG, OpenGL, XML, a fully functional web browser, a help system, a multimedia framework, as well as a rich collection of GUI widgets.
