# =============================================================================
# SECTION 1: SETUP AND PACKAGE INSTALLATION
# =============================================================================

# Clear environment
rm(list = ls())

# Set seed for reproducibility
set.seed(123)

# Install required packages if not already installed
required_packages <- c(
  "tidyverse",      # Data manipulation and visualization
  "mlbench",        # Access to Pima Indians Diabetes dataset
  "caret",          # Machine learning and cross-validation
  "pROC",           # ROC curves and AUC
  "glmnet",         # LASSO and Ridge regression
  "corrplot",       # Correlation plots
  "mice",           # Missing data imputation
  "gridExtra",      # Arranging plots
  "reshape2",       # Data reshaping
  "GGally"          # Extended ggplot2 functionality
)

# Function to install packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

# Install and load packages
install_if_missing(required_packages)

# =============================================================================
# SECTION 2: DATA LOADING AND INITIAL EXPLORATION
# =============================================================================

cat("\n=== LOADING DATA ===\n")

# Load the Pima Indians Diabetes dataset from mlbench
# This dataset contains 768 observations of Pima Indian women aged 21+
# Target: Predicting diabetes onset within 5 years
data("PimaIndiansDiabetes2", package = "mlbench")

# Create a working copy
diabetes_data <- PimaIndiansDiabetes2

# Calculate percentage of missing values
missing_pct <- colSums(is.na(diabetes_data)) / nrow(diabetes_data) * 100
cat("\nPercentage of Missing Values:\n")
print(round(missing_pct, 2))

# =============================================================================
# SECTION 3: DATA CLEANING AND PREPROCESSING
# =============================================================================

cat("\n=== DATA PREPROCESSING ===\n")

# 3.1: Handle Missing Values using Multiple Imputation
cat("\nPerforming multiple imputation for missing values...\n")

# Separate outcome variable for imputation
diabetes_features <- diabetes_data[, -ncol(diabetes_data)]
diabetes_outcome <- diabetes_data$diabetes

# Perform multiple imputation using mice
# Using predictive mean matching (pmm) method
imputed_data <- mice(diabetes_features, 
                     m = 5,           # Number of imputations
                     method = "pmm",  # Predictive mean matching
                     seed = 123,
                     printFlag = FALSE)

# Complete the dataset with imputed values
diabetes_complete <- complete(imputed_data, 1)

# Add outcome variable back
diabetes_complete$diabetes <- diabetes_outcome

cat("Missing values after imputation:\n")
print(colSums(is.na(diabetes_complete)))

# 3.2: Feature Engineering
cat("\nCreating additional features...\n")

# BMI Categories (WHO Classification)
diabetes_complete <- diabetes_complete |>
  mutate(
    bmi_category = cut(mass,
                       breaks = c(0, 18.5, 25, 30, 35, 40, Inf),
                       labels = c("Underweight", "Normal", "Overweight", 
                                 "Obese Class I", "Obese Class II", "Obese Class III"),
                       include.lowest = TRUE),
    
    # Age groups
    age_group = cut(age,
                   breaks = c(0, 30, 40, 50, 60, Inf),
                   labels = c("21-30", "31-40", "41-50", "51-60", "60+"),
                   include.lowest = TRUE),
    
    # Glucose categories (ADA Guidelines)
    glucose_category = cut(glucose,
                          breaks = c(0, 100, 126, Inf),
                          labels = c("Normal", "Prediabetes", "Diabetes"),
                          include.lowest = TRUE),
    
    # Diabetes risk score (composite)
    risk_score = scale(glucose) + scale(mass) + scale(age) + scale(pedigree)
  )

# =============================================================================
# SECTION 4: EXPLORATORY DATA ANALYSIS
# =============================================================================

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# 4.1: Class Distribution
cat("\nDiabetes Outcome Distribution:\n")
print(table(diabetes_complete$diabetes))
cat("\nProportions:\n")
print(prop.table(table(diabetes_complete$diabetes)))

# 4.2: Statistical Tests for Predictors

# T-test for continuous variables
continuous_vars <- c("pregnant", "glucose", "pressure", "triceps", 
                     "insulin", "mass", "pedigree", "age")

cat("\n=== T-tests: Diabetes vs No Diabetes ===\n")
for (var in continuous_vars) {
  test_result <- t.test(diabetes_complete[[var]] ~ diabetes_complete$diabetes)
  cat(sprintf("\n%s:\n", var))
  cat(sprintf("  Mean (No Diabetes): %.2f\n", test_result$estimate[1]))
  cat(sprintf("  Mean (Diabetes): %.2f\n", test_result$estimate[2]))
  cat(sprintf("  p-value: %.4f\n", test_result$p.value))
  cat(sprintf("  Significant: %s\n", ifelse(test_result$p.value < 0.05, "YES", "NO")))
}

# 4.3: Correlation Analysis
cat("\n=== Correlation Analysis ===\n")

# Calculate correlation matrix for numeric variables
numeric_data <- diabetes_complete |>
  select(all_of(continuous_vars)) |>
  cor()

cat("\nCorrelation Matrix:\n")
print(round(numeric_data, 2))

# =============================================================================
# SECTION 5: DATA VISUALIZATION
# =============================================================================

cat("\n=== GENERATING VISUALIZATIONS ===\n")

# Create output directory for plots
dir.create("diabetes_plots", showWarnings = FALSE)

# 5.1: Distribution plots for all continuous variables
for (var in continuous_vars) {
  p <- ggplot(diabetes_complete, aes_string(x = var, fill = "diabetes")) +
    geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
    scale_fill_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
    labs(title = paste("Distribution of", var, "by Diabetes Status"),
         x = var, y = "Count", fill = "Diabetes") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(paste0("diabetes_plots/dist_", var, ".png"), p, width = 8, height = 5)
}

# 5.2: Correlation heatmap
png("diabetes_plots/correlation_heatmap.png", width = 800, height = 800)
corrplot(numeric_data, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         addCoef.col = "black", number.cex = 0.7,
         col = colorRampPalette(c("#2E86AB", "white", "#A23B72"))(200),
         title = "Correlation Matrix of Predictors")
dev.off()

# 5.3: Box plots comparing diabetes groups
p1 <- ggplot(diabetes_complete, aes(x = diabetes, y = glucose, fill = diabetes)) +
  geom_boxplot() +
  scale_fill_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
  labs(title = "Glucose Levels by Diabetes Status", y = "Glucose (mg/dL)") +
  theme_minimal() +
  theme(legend.position = "none")

p2 <- ggplot(diabetes_complete, aes(x = diabetes, y = mass, fill = diabetes)) +
  geom_boxplot() +
  scale_fill_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
  labs(title = "BMI by Diabetes Status", y = "BMI (kg/m²)") +
  theme_minimal() +
  theme(legend.position = "none")

p3 <- ggplot(diabetes_complete, aes(x = diabetes, y = age, fill = diabetes)) +
  geom_boxplot() +
  scale_fill_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
  labs(title = "Age by Diabetes Status", y = "Age (years)") +
  theme_minimal() +
  theme(legend.position = "none")

p4 <- ggplot(diabetes_complete, aes(x = diabetes, y = pedigree, fill = diabetes)) +
  geom_boxplot() +
  scale_fill_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
  labs(title = "Diabetes Pedigree by Diabetes Status", y = "Pedigree Function") +
  theme_minimal() +
  theme(legend.position = "none")

combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)

# 5.4: Age vs BMI scatter plot
scatter_plot <- ggplot(diabetes_complete, aes(x = age, y = mass, color = diabetes)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_manual(values = c("neg" = "#2E86AB", "pos" = "#A23B72")) +
  geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Age vs BMI by Diabetes Status",
       x = "Age (years)", y = "BMI (kg/m²)", color = "Diabetes") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

# =============================================================================
# SECTION 6: MODEL PREPARATION
# =============================================================================

cat("\n=== PREPARING DATA FOR MODELING ===\n")

# Select only numeric predictors for modeling
model_data <- diabetes_complete |>
  select(all_of(continuous_vars), diabetes)

# Split data into training (70%) and testing (30%)
set.seed(123)
train_index <- createDataPartition(model_data$diabetes, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

cat(sprintf("\nTraining set size: %d observations\n", nrow(train_data)))
cat(sprintf("Testing set size: %d observations\n", nrow(test_data)))

# Check class distribution in splits
cat("\nTraining set distribution:\n")
print(prop.table(table(train_data$diabetes)))
cat("\nTesting set distribution:\n")
print(prop.table(table(test_data$diabetes)))

# =============================================================================
# SECTION 7: MODEL BUILDING - LOGISTIC REGRESSION
# =============================================================================

cat("\n=== MODEL 1: LOGISTIC REGRESSION ===\n")

# 7.1: Full Logistic Regression Model
log_model_full <- glm(diabetes ~ ., 
                      data = train_data, 
                      family = binomial(link = "logit"))

cat("\nFull Logistic Regression Model Summary:\n")
print(summary(log_model_full))

# Calculate odds ratios
odds_ratios <- exp(coef(log_model_full))
cat("\nOdds Ratios:\n")
print(round(odds_ratios, 3))

# 7.2: Stepwise Selection
log_model_step <- step(log_model_full, direction = "both", trace = 0)

cat("\nStepwise Selected Model:\n")
print(summary(log_model_step))

# 7.3: Predictions and Evaluation
# Training set predictions
train_pred_prob <- predict(log_model_step, train_data, type = "response")
train_pred_class <- ifelse(train_pred_prob > 0.5, "pos", "neg")
train_pred_class <- factor(train_pred_class, levels = c("neg", "pos"))

# Testing set predictions
test_pred_prob <- predict(log_model_step, test_data, type = "response")
test_pred_class <- ifelse(test_pred_prob > 0.5, "pos", "neg")
test_pred_class <- factor(test_pred_class, levels = c("neg", "pos"))

# Confusion Matrix - Training
cat("\n=== Training Set Performance ===\n")
train_cm <- confusionMatrix(train_pred_class, train_data$diabetes, positive = "pos")
print(train_cm)

# Confusion Matrix - Testing
cat("\n=== Testing Set Performance ===\n")
test_cm <- confusionMatrix(test_pred_class, test_data$diabetes, positive = "pos")
print(test_cm)

# ROC Curve and AUC
train_roc <- roc(train_data$diabetes, train_pred_prob)
test_roc <- roc(test_data$diabetes, test_pred_prob)

cat(sprintf("\nTraining AUC: %.4f\n", auc(train_roc)))
cat(sprintf("Testing AUC: %.4f\n", auc(test_roc)))

# Plot ROC curves
png("diabetes_plots/roc_logistic.png", width = 800, height = 600)
plot(test_roc, col = "#A23B72", lwd = 2, main = "ROC Curve - Logistic Regression")
lines(train_roc, col = "#2E86AB", lwd = 2, lty = 2)
legend("bottomright", 
       legend = c(paste("Test AUC =", round(auc(test_roc), 3)),
                 paste("Train AUC =", round(auc(train_roc), 3))),
       col = c("#A23B72", "#2E86AB"), lwd = 2, lty = c(1, 2))
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()


# =============================================================================
# SECTION 8: MODEL BUILDING - LASSO REGRESSION
# =============================================================================

cat("\n=== MODEL 2: LASSO REGRESSION ===\n")

# Prepare data for glmnet (requires matrix format)
x_train <- model.matrix(diabetes ~ ., train_data)[, -1]
y_train <- train_data$diabetes
x_test <- model.matrix(diabetes ~ ., test_data)[, -1]
y_test <- test_data$diabetes

# Convert outcome to numeric (0/1) for glmnet
y_train_numeric <- ifelse(y_train == "pos", 1, 0)
y_test_numeric <- ifelse(y_test == "pos", 1, 0)

# Cross-validation to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train_numeric, 
                      alpha = 1, 
                      family = "binomial",
                      nfolds = 10)

cat("\nOptimal Lambda (min):", cv_lasso$lambda.min, "\n")
cat("Optimal Lambda (1se):", cv_lasso$lambda.1se, "\n")

# Plot cross-validation curve
png("diabetes_plots/lasso_cv.png", width = 800, height = 600)
plot(cv_lasso, main = "LASSO Cross-Validation")
dev.off()

# Fit LASSO model with optimal lambda
lasso_model <- glmnet(x_train, y_train_numeric, 
                      alpha = 1, 
                      family = "binomial",
                      lambda = cv_lasso$lambda.min)

# Display coefficients
cat("\nLASSO Coefficients:\n")
lasso_coef <- coef(lasso_model)
print(lasso_coef)

# Count non-zero coefficients
non_zero <- sum(lasso_coef != 0) - 1  # Exclude intercept
cat(sprintf("\nNumber of selected features: %d out of %d\n", non_zero, ncol(x_train)))

# Predictions
lasso_pred_prob_train <- predict(lasso_model, x_train, type = "response")
lasso_pred_class_train <- ifelse(lasso_pred_prob_train > 0.5, "pos", "neg")
lasso_pred_class_train <- factor(lasso_pred_class_train, levels = c("neg", "pos"))

lasso_pred_prob_test <- predict(lasso_model, x_test, type = "response")
lasso_pred_class_test <- ifelse(lasso_pred_prob_test > 0.5, "pos", "neg")
lasso_pred_class_test <- factor(lasso_pred_class_test, levels = c("neg", "pos"))

# Evaluation
cat("\n=== LASSO Training Set Performance ===\n")
lasso_train_cm <- confusionMatrix(lasso_pred_class_train, train_data$diabetes, positive = "pos")
print(lasso_train_cm)

cat("\n=== LASSO Testing Set Performance ===\n")
lasso_test_cm <- confusionMatrix(lasso_pred_class_test, test_data$diabetes, positive = "pos")
print(lasso_test_cm)

# ROC and AUC
lasso_train_roc <- roc(train_data$diabetes, as.numeric(lasso_pred_prob_train))
lasso_test_roc <- roc(test_data$diabetes, as.numeric(lasso_pred_prob_test))

cat(sprintf("\nLASSO Training AUC: %.4f\n", auc(lasso_train_roc)))
cat(sprintf("LASSO Testing AUC: %.4f\n", auc(lasso_test_roc)))

# =============================================================================
# SECTION 9: MODEL BUILDING - RANDOM FOREST
# =============================================================================

cat("\n=== MODEL 3: RANDOM FOREST ===\n")

# Train Random Forest with cross-validation
rf_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

rf_model <- train(
  diabetes ~ .,
  data = train_data,
  method = "rf",
  trControl = rf_control,
  metric = "ROC",
  ntree = 500,
  importance = TRUE
)

cat("\nRandom Forest Model:\n")
print(rf_model)

# Variable Importance
var_imp <- varImp(rf_model)
cat("\nVariable Importance:\n")
print(var_imp)

# Plot variable importance
png("diabetes_plots/rf_importance.png", width = 800, height = 600)
plot(var_imp, main = "Random Forest Variable Importance")
dev.off()

# Predictions
rf_pred_train <- predict(rf_model, train_data)
rf_pred_prob_train <- predict(rf_model, train_data, type = "prob")

rf_pred_test <- predict(rf_model, test_data)
rf_pred_prob_test <- predict(rf_model, test_data, type = "prob")

# Evaluation
cat("\n=== Random Forest Training Set Performance ===\n")
rf_train_cm <- confusionMatrix(rf_pred_train, train_data$diabetes, positive = "pos")
print(rf_train_cm)

cat("\n=== Random Forest Testing Set Performance ===\n")
rf_test_cm <- confusionMatrix(rf_pred_test, test_data$diabetes, positive = "pos")
print(rf_test_cm)

# ROC and AUC
rf_train_roc <- roc(train_data$diabetes, rf_pred_prob_train$pos)
rf_test_roc <- roc(test_data$diabetes, rf_pred_prob_test$pos)

cat(sprintf("\nRandom Forest Training AUC: %.4f\n", auc(rf_train_roc)))
cat(sprintf("Random Forest Testing AUC: %.4f\n", auc(rf_test_roc)))

# =============================================================================
# SECTION 10: MODEL COMPARISON
# =============================================================================

cat("\n=== COMPREHENSIVE MODEL COMPARISON ===\n")

# Create comparison dataframe
model_comparison <- data.frame(
  Model = c("Logistic Regression", "LASSO", "Random Forest"),
  Train_Accuracy = c(
    train_cm$overall["Accuracy"],
    lasso_train_cm$overall["Accuracy"],
    rf_train_cm$overall["Accuracy"]
  ),
  Test_Accuracy = c(
    test_cm$overall["Accuracy"],
    lasso_test_cm$overall["Accuracy"],
    rf_test_cm$overall["Accuracy"]
  ),
  Train_Sensitivity = c(
    train_cm$byClass["Sensitivity"],
    lasso_train_cm$byClass["Sensitivity"],
    rf_train_cm$byClass["Sensitivity"]
  ),
  Test_Sensitivity = c(
    test_cm$byClass["Sensitivity"],
    lasso_test_cm$byClass["Sensitivity"],
    rf_test_cm$byClass["Sensitivity"]
  ),
  Train_Specificity = c(
    train_cm$byClass["Specificity"],
    lasso_train_cm$byClass["Specificity"],
    rf_train_cm$byClass["Specificity"]
  ),
  Test_Specificity = c(
    test_cm$byClass["Specificity"],
    lasso_test_cm$byClass["Specificity"],
    rf_test_cm$byClass["Specificity"]
  ),
  Train_AUC = c(
    auc(train_roc),
    auc(lasso_train_roc),
    auc(rf_train_roc)
  ),
  Test_AUC = c(
    auc(test_roc),
    auc(lasso_test_roc),
    auc(rf_test_roc)
  )
)
# IMPORTANT #
cat("\nModel Performance Comparison:\n")
print(model_comparison)

# Plot comparison
png("diabetes_plots/model_comparison.png", width = 1000, height = 600)
par(mfrow = c(1, 2))

# Test Accuracy Comparison
barplot(model_comparison$Test_Accuracy,
        names.arg = model_comparison$Model,
        main = "Test Set Accuracy Comparison",
        ylab = "Accuracy",
        ylim = c(0, 1),
        col = c("#2E86AB", "#F18F01", "#A23B72"),
        las = 2)
abline(h = 0.5, lty = 2, col = "gray")

# Test AUC Comparison
barplot(model_comparison$Test_AUC,
        names.arg = model_comparison$Model,
        main = "Test Set AUC Comparison",
        ylab = "AUC",
        ylim = c(0, 1),
        col = c("#2E86AB", "#F18F01", "#A23B72"),
        las = 2)
abline(h = 0.5, lty = 2, col = "gray")

dev.off()

# Combined ROC Curve
png("diabetes_plots/roc_comparison.png", width = 800, height = 800)
plot(test_roc, col = "#2E86AB", lwd = 2, main = "ROC Curve Comparison - All Models")
lines(lasso_test_roc, col = "#F18F01", lwd = 2)
lines(rf_test_roc, col = "#A23B72", lwd = 2)
legend("bottomright",
       legend = c(
         paste("Logistic Regression (AUC =", round(auc(test_roc), 3), ")"),
         paste("LASSO (AUC =", round(auc(lasso_test_roc), 3), ")"),
         paste("Random Forest (AUC =", round(auc(rf_test_roc), 3), ")")
       ),
       col = c("#2E86AB", "#F18F01", "#A23B72"),
       lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# =============================================================================
# SECTION 11: PREDICTION FUNCTION
# =============================================================================

cat("\n=== CREATING PREDICTION FUNCTION ===\n")

# Function to predict diabetes risk for new patients
predict_diabetes_risk <- function(
    pregnant,
    glucose,
    pressure,
    triceps,
    insulin,
    mass,
    pedigree,
    age,
    model_choice = "logistic"
) {
  
  # Create data frame with input values
  new_patient <- data.frame(
    pregnant = pregnant,
    glucose = glucose,
    pressure = pressure,
    triceps = triceps,
    insulin = insulin,
    mass = mass,
    pedigree = pedigree,
    age = age
  )
  
  # Select model and make prediction
  if (model_choice == "logistic") {
    prob <- predict(log_model_step, new_patient, type = "response")
    model_name <- "Logistic Regression"
  } else if (model_choice == "lasso") {
    new_x <- model.matrix(~ ., new_patient)[, -1]
    prob <- predict(lasso_model, new_x, type = "response")[1, 1]
    model_name <- "LASSO"
  } else if (model_choice == "rf") {
    prob <- predict(rf_model, new_patient, type = "prob")$pos
    model_name <- "Random Forest"
  } else {
    stop("Invalid model choice. Choose 'logistic', 'lasso', or 'rf'")
  }
  
  # Determine risk level
  risk_level <- if (prob < 0.3) {
    "Low Risk"
  } else if (prob < 0.7) {
    "Moderate Risk"
  } else {
    "High Risk"
  }
  
  # Return results
  result <- list(
    model = model_name,
    probability = prob,
    prediction = ifelse(prob > 0.5, "Diabetes", "No Diabetes"),
    risk_level = risk_level,
    input_data = new_patient
  )
  
  return(result)
}

# =============================================================================
# SECTION 12: EXAMPLE PREDICTIONS
# =============================================================================

cat("\n=== EXAMPLE PREDICTIONS ===\n")

# Example 1: Low risk patient
cat("\nExample 1: Young, healthy individual\n")
example1 <- predict_diabetes_risk(
  pregnant = 2,
  glucose = 85,
  pressure = 70,
  triceps = 20,
  insulin = 80,
  mass = 22.5,
  pedigree = 0.2,
  age = 25,
  model_choice = "rf"
)
print(example1)

# Example 2: Moderate risk patient
cat("\nExample 2: Middle-aged with elevated glucose\n")
example2 <- predict_diabetes_risk(
  pregnant = 5,
  glucose = 140,
  pressure = 85,
  triceps = 30,
  insulin = 150,
  mass = 28.5,
  pedigree = 0.5,
  age = 45,
  model_choice = "rf"
)
print(example2)

# Example 3: High risk patient
cat("\nExample 3: High-risk profile\n")
example3 <- predict_diabetes_risk(
  pregnant = 8,
  glucose = 180,
  pressure = 90,
  triceps = 40,
  insulin = 200,
  mass = 35.0,
  pedigree = 0.8,
  age = 55,
  model_choice = "rf"
)
print(example3)

################################################################################
# END OF SCRIPT
################################################################################
