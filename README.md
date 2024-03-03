## Hone Spectral Processing Challenge (HSP C)

### Intro

[Hone's](https://honeag.com/about-hone/) product offering is dependent on the continuous integration of new data to deploy predictive models, generate actionable insights to our customers in Australia and Internationally. Our fundamental product<sup>[[1](https://honeag.com/hone-lab-red/),[2](https://honeag.com/hone-carbon/)]</sup> offering is a hardware-agnostic software platform that enables the rapid development and deployment of predictive models. Your role within this team will be to expedite and scale our means of developing and deploying our models and to tackle complex data logistics problems.

### Challenge

Thank you for agreeing to participate in this component of the Hone Carbon recruitment process!

This is a two part scenario and is designed to test your ability to transform and manipulate data, apply statistical methods to filter it and test basic understanding of data presentation and software design.

Both tasks are expected to be solved using Python. There is no restriction on the packages that can be used. Useful links are provided in the [appendix](#appendix) section.

#### Scenario 1

You are provided with an excel workbook which contains 2 sheets. Each sheet acts as a standalone dataset. You can find them in the [data](data/) folder and are required to use both of these datasets in conjunction to solve the presented challenges.

Each dataset contains **161** columns, including:

- Unique identifier column: ***Sample ID***
- **11** Metadata columns
- **1** Response Variable column (***Total Organic Carbon [TOC]***)
- One ***Background*** indicator column
- **146** spectral columns (wavelengths spanning from *1350nm* ~ *2550nm* expressed in raw intensities)

Each row can be of two types: *Sample Scan* or *Background Scan* and can be identified using the *Background* column. Each sample has a repeat scan of 10 counts and can be identified using *ScanIndex*.

You have been requested by a stakeholder that the data associated with the dependent variable and spectral data are transformed to be suitable inputs for subsequent modeling of the dependent variables using the spectral data. You are expected to work with data from both datasets, between the wavelength range of **1400nm ~ 2400nm** and a granularity of **1nm**.

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

#### Scenario 2
Imagine you are presenting the findings from above scenario to your colleagues with limited understanding of statistics or data science. It falls on you to present them in a way laymen can understand.

Create a simple python based application that can take in a dataset and produce visualizations from each step you perform in the above scenario to show how each decision you made influenced the preprocessing and how outlier detection algorithms removed outliers along with ability to make a prediction on any trained model (remember, the exact same preprocessing needs to be performed on any prediction target).

There is no restriction on the type of application you choose to create. A few useful links are attached in the [appendix](#appendix) section.

#### Bonus Scenario
This is not a mandatory scenario but you can choose to attempt this section and show off your impressive modelling skills and awe us.

1. Explore various normalization techniques of the dependent variables themselves and the effect of that on the predictive capability. If using such techniques, how do you account for or include these in a productionized model?
2. Develop architecture to productionize the entire training and prediction flow. Make reasonable assumptions where required but be sure to account for most real world cases.

### Submission

You can choose to fork this repository or create your own repository to show your work.

We are most interested in your thought process behind all the decisions made, so leave as many comments as possible either as inline comments or as a separate report. For any application / notebook created, provide clear instructions / examples that can help us run your application at our end in a markdown.

Once you have completed all the given tasks, shoot an email to [sri@honeag.com](mailto:sri@honeag.com) stating that you have finished your tasks and that the repository is ready for evaluation.

**kōun o inorimasu (I wish you good luck)**

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
