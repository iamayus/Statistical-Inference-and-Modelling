# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(randomForest)
library(corrplot)
library(reshape2)

# Initial Data Exploration
# Load the dataset
health_insurance <- read_csv("Health_insurance.csv")

# Display the structure of the dataset
str(health_insurance)

# Display summary statistics of the dataset
summary(health_insurance)

# Check for missing values
colSums(is.na(health_insurance))

# Display the first few rows of the dataset
head(health_insurance)

# EDA
# Compare data based on region
ggplot(health_insurance, aes(x = region, fill = region)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Distribution of Data Based on Region", x = "Region", y = "Count")

# Compare data based on gender
ggplot(health_insurance, aes(x = sex, fill = sex)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Distribution of Data Based on Gender", x = "Gender", y = "Count")

# Visualize gender distribution in smoker category
ggplot(health_insurance, aes(x = smoker, fill = sex)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(title = "Gender Distribution in Smoker Category", x = "Smoker", y = "Count")

# Age distribution
ggplot(health_insurance, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  theme_minimal() +
  labs(title = "Age Distribution", x = "Age", y = "Frequency")

# Compare the data based on children count
ggplot(health_insurance, aes(x = factor(children), fill = factor(children))) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Distribution Based on Children Count", x = "Number of Children", y = "Count")

# Scatter plots to visualize relationships between numeric variables and charges
ggplot(health_insurance, aes(x = age, y = charges)) +
  geom_point() +
  theme_minimal() +
  labs(title = "Age vs Charges", x = "Age", y = "Charges")

ggplot(health_insurance, aes(x = bmi, y = charges)) +
  geom_point() +
  theme_minimal() +
  labs(title = "BMI vs Charges", x = "BMI", y = "Charges")

ggplot(health_insurance, aes(x = children, y = charges)) +
  geom_point() +
  theme_minimal() +
  labs(title = "Number of Children vs Charges", x = "Number of Children", y = "Charges")

# Correlation analysis before encoding
numeric_cols_before <- health_insurance %>% select(age, bmi, children, charges)
cor_matrix_before <- cor(numeric_cols_before)

# Transform correlation matrix to long format
melted_cor_matrix_before <- melt(cor_matrix_before)

# Plot heatmap before encoding
ggplot(data = melted_cor_matrix_before, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 4) +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Matrix Heatmap Before Encoding")

# Label Encoding for sex and smoker
health_insurance <- health_insurance %>%
  mutate(
    sex = ifelse(sex == "male", 1, 0),
    smoker = ifelse(smoker == "yes", 1, 0)
  )

# One-Hot Encoding for region
dummies <- dummyVars(" ~ region", data = health_insurance)
region_encoded <- predict(dummies, newdata = health_insurance)

# Combine the original dataset with the one-hot encoded columns and remove the original region column
health_insurance <- cbind(health_insurance, region_encoded) %>%
  select(-region)

# Display the first few rows of the transformed dataset
head(health_insurance)

# Correlation analysis after encoding
cor_matrix_after <- cor(health_insurance)

# Transform correlation matrix to long format
melted_cor_matrix_after <- melt(cor_matrix_after)

# Plot heatmap after encoding
ggplot(data = melted_cor_matrix_after, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab", 
                       name = "Correlation") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3) +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  coord_fixed() +
  labs(title = "Correlation Matrix Heatmap After Encoding")

# Set seed for reproducibility
set.seed(123)

# Split the data into training (80%) and testing (20%) sets
trainIndex <- createDataPartition(health_insurance$charges, p = 0.8, list = FALSE)
train_data <- health_insurance[trainIndex, ]
test_data <- health_insurance[-trainIndex, ]

# Display the dimensions of the training and testing sets
dim(train_data)
dim(test_data)

# Standardize the features (excluding the target variable)
preProcValues <- preProcess(train_data, method = c("center", "scale"))

train_data_scaled <- predict(preProcValues, train_data)
test_data_scaled <- predict(preProcValues, test_data)

# Display the first few rows of the scaled training data
head(train_data_scaled)

# Train a linear regression model
linear_model <- train(charges ~ ., data = train_data_scaled, method = "lm")

# Display the model summary
summary(linear_model)

# Predict on the testing set
predictions <- predict(linear_model, newdata = test_data_scaled)

# Calculate performance metrics (e.g., RMSE, R-squared)
linear_performance <- postResample(predictions, test_data_scaled$charges)

print(linear_performance)

# Train a Decision Tree model
set.seed(123)
tree_model <- train(charges ~ ., data = train_data_scaled, method = "rpart")

# Display the model summary
print(tree_model)

# Predict on the testing set using the Decision Tree model
tree_predictions <- predict(tree_model, newdata = test_data_scaled)

# Calculate performance metrics for the Decision Tree model
tree_performance <- postResample(tree_predictions, test_data_scaled$charges)

print(tree_performance)

# Train a Random Forest model
set.seed(123)
rf_model <- train(charges ~ ., data = train_data_scaled, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Display the model summary
print(rf_model)

# Predict on the testing set using the Random Forest model
rf_predictions <- predict(rf_model, newdata = test_data_scaled)

# Calculate performance metrics for the Random Forest model
rf_performance <- postResample(rf_predictions, test_data_scaled$charges)

print(rf_performance)

# Train a Gradient Boosting model
set.seed(123)
gb_model <- train(charges ~ ., data = train_data_scaled, method = "gbm", 
                  trControl = trainControl(method = "cv", number = 5), verbose = FALSE)

# Display the model summary
print(gb_model)

# Predict on the testing set using the Gradient Boosting model
gb_predictions <- predict(gb_model, newdata = test_data_scaled)

# Calculate performance metrics for the Gradient Boosting model
gb_performance <- postResample(gb_predictions, test_data_scaled$charges)

print(gb_performance)

# Display performance metrics for all models
model_performance <- data.frame(
  Model = c("Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"),
  RMSE = c(linear_performance[1],
           tree_performance[1],
           rf_performance[1],
           gb_performance[1]),
  Rsquared = c(linear_performance[2],
               tree_performance[2],
               rf_performance[2],
               gb_performance[2])
)

print(model_performance)


# Plot RMSE comparison
ggplot(model_performance, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison - RMSE", x = "Model", y = "RMSE") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "none")

# Plot R-squared comparison
ggplot(model_performance, aes(x = Model, y = Rsquared, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison - R-squared", x = "Model", y = "R-squared") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "none")
