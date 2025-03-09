# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was a Logistic Regression model.
## Intended Use
The intended use of the model is to predict if a persons annual income exceeds $50k USD using data available via the Census.
## Training Data
All training data is derived from the Census. The categories included in census data are: age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary.
## Evaluation Data
The evaluation data was from the same census data. 20% of the data was used for testing data.
## Metrics
The metrics used were Precision (returned .6301), Recall (returned 0.3176), and F1 (returned 0.4223)
## Ethical Considerations
Ethical considerations to make are biases based on historic, systemic, or cultural biases. Effort should be made to help ensure these are not exaggerated in the model.
## Caveats and Recommendations
Outdated data may cause newer data to be increasingly inaccurate. With changes in geopolitical powers, inflation, and societal norms, adjustments should be made to the training data to more accurately represent potential changes.