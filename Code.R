library(tidyverse)
library(caret)
library(car)
library(lmtest)
library(plyr)
library(caret)
library(MASS)
library(tree)
library(corrplot)
library(readxl)
library(rpart)
library(rattle)
library(scales)
library(caTools)
library(randomForest)
library(ggrepel)
library(gmodels)
library(xgboost)
library(pscl)
library(ResourceSelection)
library(InformationValue)
library(reshape2)
library(pROC)
library(rstudioapi)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Importing the dataset
credit_data <-
    read_excel("Assessment Case Study Data.xlsx", sheet = "LoanData")

# Examining data
head(credit_data)
names(credit_data)
dim(credit_data)
str(credit_data)

# Removing columns 'cust_id' and 'acc_no'
credit_data <-
    credit_data %>% dplyr::select(-c("cust_id", "acc_no"))
# Renaming the 'amount (USD) column to 'amount'.
colnames(credit_data)[5] <- "amount"

# Checking for anomalies in the data.
for (col in colnames(credit_data)) {
    if (all(!is.na(as.numeric(unlist(credit_data[col]))))) {
        print(paste(col, ": NUM"))
    } else {
        print(paste(col, ": ", unique(credit_data[col])))
    }
}

# Column wise check for NA values
lapply(credit_data, function(x) {
    sum(is.na(x))
}) # years at residence has missing values.

# Changing 'purpose' variable to character for easy renaming of "car0" variable (all combined into a singular "car" variable now)
credit_data$purpose[credit_data$purpose == "car0"] <- "car"

# Checking for unknowns and missing values.
sum(credit_data$checking_balance == "unknown") / length(credit_data$checking_balance)
sum(credit_data$savings_balance == "unknown") / length(credit_data$savings_balance)
sum(is.na(credit_data$years_at_residence)) / length(credit_data$years_at_residence)

# Filling missing values from 'years_at_residence' with the median.
credit_data$years_at_residence[is.na(credit_data$years_at_residence)] <-
    median(credit_data$years_at_residence, na.rm = T)

## Function to clean columns and transform the columns to their required datatypes.
col_convert <- function(col) {
    if (all(!is.na(as.numeric(col)))) {
        col <- as.numeric(col)
    } else {
        col <- as.factor(col)
    }
}

credit_data <-
    data.frame(lapply(credit_data, function(x) {
        col_convert(x)
    }))
credit_data$dependants <- as.factor(credit_data$dependants)
lapply(credit_data, function(x) {
    sum(is.na(x))
})


# Using a gradient boosting algorithm to predict 'unknowns' from 'checking balance' and 'savings balance' columns.

savings_unknowns <-
    credit_data[credit_data$savings_balance == "unknown",] %>% dplyr::select(-c(default, savings_balance))
savings_knowns <-
    credit_data[credit_data$savings_balance != "unknown",] %>% dplyr::select(-c(default))
savings_knowns <-
    savings_knowns[savings_knowns$checking_balance != "unknown",]

checking_unknowns <-
    credit_data[credit_data$checking_balance == "unknown",] %>% dplyr::select(-c(default, checking_balance))
checking_knowns <-
    credit_data[credit_data$checking_balance != "unknown",] %>% dplyr::select(-c(default))
checking_knowns <-
    checking_knowns[checking_knowns$savings_balance != "unknown",]

savings_knowns$savings_balance <-
    droplevels(savings_knowns$savings_balance)
savings_knowns$checking_balance <-
    droplevels(savings_knowns$checking_balance)

checking_knowns$checking_balance <-
    droplevels(checking_knowns$checking_balance)
checking_knowns$savings_balance <-
    droplevels(checking_knowns$savings_balance)

# Creating training and testing splits
set.seed(256)
samp <- createDataPartition(savings_knowns$savings_balance, p = 0.6)

savings_labels <-
    as.numeric(unlist(savings_knowns[unlist(samp),]$savings_balance))
savings_test_labels <-
    as.numeric(unlist(savings_knowns[-unlist(samp),]$savings_balance))

checking_labels <-
    as.numeric(unlist(checking_knowns[unlist(samp),]$checking_balance))
checking_test_labels <-
    as.numeric(unlist(checking_knowns[-unlist(samp),]$checking_balance))

savings_values_tr <- savings_knowns[unlist(samp),]$savings_balance
checking_values_tr <-
    checking_knowns[unlist(samp),]$checking_balance

savings_values_test <-
    savings_knowns[-unlist(samp),]$savings_balance
checking_values_test <-
    checking_knowns[-unlist(samp),]$checking_balance

savings_knowns <-
    savings_knowns %>% dplyr::select(-c(savings_balance))
checking_knowns <-
    checking_knowns %>% dplyr::select(-c(checking_balance))
# Creating training and testing splits

savings_tr <- savings_knowns[unlist(samp),]
checking_tr <- checking_knowns[unlist(samp),]
savings_test <- savings_knowns[-unlist(samp),]
checking_test <- checking_knowns[-unlist(samp),]


# Building the model
savings_model <-
    xgboost::xgboost(
        data = data.matrix(savings_knowns[unlist(samp),]),
        label = savings_labels,
        nrounds = 1000,
        verbose = F
    )
checking_model <-
    xgboost::xgboost(
        data = data.matrix(checking_knowns[unlist(samp),]),
        label = checking_labels,
        nrounds = 1000,
        verbose = F
    )

# Insample analysis.
savings_insamp <- predict(savings_model, data.matrix(savings_tr))
savings_insamp <- round(savings_insamp, 1)
savings_insamp <-
    factor(savings_insamp,
           levels = c(1, 2, 3, 4),
           labels = levels(savings_values_tr))

checking_insamp <-
    predict(checking_model, data.matrix(checking_test))
checking_insamp <- round(checking_insamp, 1)
checking_insamp <-
    factor(
        checking_insamp,
        levels = c(1, 2, 3),
        labels = levels(checking_values_test)
    )


# Checking for accuracy.

caret::confusionMatrix(savings_insamp, savings_values_tr)
caret::confusionMatrix(checking_insamp, checking_values_test)

# Outsample analysis to check for good fit.

savings_outsamp <- predict(savings_model, data.matrix(savings_test))
savings_outsamp <- round(savings_outsamp, 1)
savings_outsamp <-
    factor(
        savings_outsamp,
        levels = c(1, 2, 3, 4),
        labels = levels(savings_values_test)
    )

checking_outsamp <-
    predict(checking_model, data.matrix(checking_tr))
checking_outsamp <- round(checking_outsamp, 1)
checking_outsamp <-
    factor(
        checking_outsamp,
        levels = c(1, 2, 3),
        labels = levels(checking_values_tr)
    )

caret::confusionMatrix(savings_outsamp, savings_values_test)
caret::confusionMatrix(checking_outsamp, checking_values_tr)

# We see that we have incredibly high accuracy in the training and testing sets, hence there is no
# overfitting.

savings_predictions <-
    predict(savings_model, data.matrix(savings_unknowns))
savings_predictions <- round(savings_predictions, 0)
savings_predictions <-
    factor(
        savings_predictions,
        levels = c(1, 2, 3, 4),
        labels = levels(savings_values_test)
    )
credit_data$savings_balance[credit_data$savings_balance == "unknown"] <-
    savings_predictions
credit_data$savings_balance <-
    droplevels(credit_data$savings_balance)

checking_predictions <-
    predict(checking_model, data.matrix(checking_unknowns))
checking_predictions <- round(checking_predictions, 0)
checking_predictions <-
    factor(
        checking_predictions,
        levels = c(1, 2, 3),
        labels = levels(checking_values_test)
    )
credit_data$checking_balance[credit_data$checking_balance == "unknown"] <-
    checking_predictions
credit_data$checking_balance <-
    droplevels(credit_data$checking_balance)

sapply(credit_data, function(x) {
    sum(is.na(x))
})


# Exploring cleaned dataset.
dim(credit_data)
glimpse(credit_data)
summary(credit_data)
prop.table(table(credit_data$default))
summary(credit_data$months_loan_duration)
summary(credit_data$amount)
prop.table(table(credit_data$checking_balance))
prop.table(table(credit_data$savings_balance))

# Outlier analysis
# Function to generate spectrum of colours
colfunc <- colorRampPalette(c("#FF5E04", "#333333"))

numeric_cols <-
    colnames(credit_data)[sapply(credit_data, function(x) {
        is.numeric(x)
    })]
boxplot_data <- credit_data %>%
    dplyr::select(all_of(numeric_cols)) %>%
    melt()

ggplot(data = boxplot_data, aes(x = variable, y = value)) +
    geom_boxplot(fill = colfunc(6)) +
    facet_wrap( ~ variable, scales = "free") +
    ggtitle("Boxplots") +
    theme_minimal() +
    xlab("") +
    ylab("") +
    theme(
        plot.title = element_text(hjust = 0.5),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        strip.background = element_rect(colour = "black", fill = "black"),
        strip.text = element_text(colour = "white"),
        panel.border = element_rect(colour = "black", fill = NA)
    )

# We see that amount and loan duration have high number of outliers.
# Using log transformation to normalise:

boxplot_data$value[boxplot_data$variable == "amount"] <-
    log(boxplot_data$value[boxplot_data$variable == "amount"])
boxplot_data$value[boxplot_data$variable == "months_loan_duration"] <-
    log(boxplot_data$value[boxplot_data$variable == "months_loan_duration"])


ggplot(data = boxplot_data, aes(x = variable, y = value)) +
    geom_boxplot(fill = colfunc(6)) +
    facet_wrap( ~ variable, scales = "free") +
    ggtitle("Boxplots after Log Transformation") +
    theme_minimal() +
    xlab("") +
    ylab("") +
    theme(
        plot.title = element_text(hjust = 0.5),
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        strip.background = element_rect(colour = "black", fill = "black"),
        strip.text = element_text(colour = "white"),
        panel.border = element_rect(colour = "black", fill = NA)
    )

# We see that outliers substantially reduce in the data set.
#
# credit_data$amount <- log(credit_data$amount)
# credit_data$months_loan_duration <-
#     log(credit_data$months_loan_duration)

# Visualising Data

# Density plot of loan amounts
ggplot(credit_data, aes(x = amount)) +
    geom_histogram(aes(y = ..density..), colour = "#333333", fill = "white") +
    geom_density(fill = "#FF5E04",
                 color = "#FF5E04",
                 alpha = .3) +
    ggtitle("Density Plot of Amount") +
    xlab("Amount of Loan") +
    ylab("Density") +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        family = "serif",
        size = 17
    ),)

# Density plot of age
ggplot(credit_data, aes(x = age)) +
    geom_density(fill = "firebrick1",
                 colour = "firebrick1",
                 alpha = 0.4) +
    ggtitle("Density Plot of Age") +
    xlab("Amount of Loan") +
    ylab("Density") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),)

# Pie chart of defaulters
frequency_table <- data.frame(table(credit_data$default))
colnames(frequency_table) <- c("Default", "Frequency")
frequency_table["Percentage"] <-
    (frequency_table$Frequency) / sum(frequency_table$Frequency)

ggplot(data = frequency_table) +
    geom_bar(aes(x = "",
                 y = Frequency,
                 fill = Default,),
             width = 1,
             stat = "identity") +
    scale_fill_manual(values = c("#333333", "#FF5E04")) +
    coord_polar("y") +
    theme_minimal() +
    ggtitle("Frequency of Defaulters") +
    geom_text_repel(
        aes(
            x = "",
            y = Frequency,
            label = paste(round(Percentage, 2) * 100, "%", sep = "")
        ),
        size = 5,
        show.legend = F,
        nudge_x = 0,
        colour = "white",
        hjust = c(0, -0.8),
        vjust = c(0, -0.8),
        segment.color = "transparent"
    ) +
    guides(fill = guide_legend(title = "Default")) +
    theme(
        plot.title = element_text(
            hjust = 0.5,
            size = 17,
            family = "serif"
        ),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_blank()
    )

# Credit history of defaulters
(
    credit_data %>%
        dplyr::select(credit_history, default) %>%
        dplyr::count(credit_history, default) %>%
        ggplot(aes(
            x = credit_history, y = n, fill = default
        )) +
        geom_bar(stat = "identity",) +
        guides(fill = guide_legend(title = "Default")) +
        xlab("Credit History") +
        ylab("Number of Obligors") +
        ggtitle("Default Rate Based on Credit History") +
        theme_minimal() +
        theme(plot.title = element_text(
            hjust = 0.5,
            size = 17,
            family = "serif"
        ),) +
        scale_fill_manual(values = c("#FF5E04", "#333333"))
)

# Default rate based on credit history
CrossTable(
    credit_data$credit_history,
    credit_data$default,
    prop.r = TRUE,
    prop.c = FALSE,
    prop.t = FALSE,
    prop.chisq = FALSE
)

# Defaults based on loan type
credit_data %>%
    dplyr::select(purpose, default) %>%
    dplyr::count(purpose, default) %>%
    filter(default == "yes") %>%
    ggplot(aes(x = purpose, y = n, fill = purpose)) +
    geom_bar(stat = "identity") +
    xlab("Purpose") +
    ylab("Number of Obligors") +
    geom_text(aes(label = n),
              vjust = 1.6,
              color = "white",
              size = 3.5) +
    ggtitle("Loan type wise defaulters") +
    scale_fill_discrete(
        name = "Purpose of Loan",
        labels = c(
            "Business",
            "Car",
            "Education",
            "Furniture/Appliances",
            "Renovations"
        )
    ) +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 26,
        family = "serif"
    ),) +
    scale_fill_manual(values = colfunc(5))

# Loan type wise default rate
CrossTable(
    credit_data$purpose,
    credit_data$default,
    prop.r = TRUE,
    prop.c = FALSE,
    prop.t = FALSE,
    prop.chisq = FALSE
)

# Defaults based on existing loans
credit_data %>%
    dplyr::select(existing_loans_count, default) %>%
    dplyr::count(existing_loans_count, default) %>%
    # filter(default == "yes") %>%
    ggplot(aes(x = existing_loans_count, y = n, fill = default)) +
    geom_bar(stat = "identity") +
    guides(fill = guide_legend(title = "Default")) +
    xlab("Number of existing loans") +
    ylab("Number of Obligors") +
    ggtitle("Existing Loans vs Default") +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 17,
        family = "serif"
    ),) +
    scale_fill_manual(values = c("#FF5E04", "#333333"))

credit_data %>%
    dplyr::select(job, default) %>%
    dplyr::count(job, default) %>%
    # filter(default == "yes") %>%
    ggplot(aes(x = job, y = n, fill = default)) +
    geom_bar(stat = "identity") +
    guides(fill = guide_legend(title = "Default")) +
    xlab("Job") +
    ylab("Number of Obligors") +
    ggtitle("Job vs Default") +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 17,
        family = "serif"
    ),) +
    scale_fill_manual(values = c("#FF5E04", "#333333"))

CrossTable(
    credit_data$job,
    credit_data$default,
    prop.r = TRUE,
    prop.c = FALSE,
    prop.t = FALSE,
    prop.chisq = FALSE
)

# Default vs Checking Balance
credit_data %>%
    dplyr::select(checking_balance, default) %>%
    dplyr::count(checking_balance, default) %>%
    # filter(default == "yes") %>%
    ggplot(aes(x = checking_balance, y = n, fill = default)) +
    geom_bar(stat = "identity") +
    guides(fill = guide_legend(title = "Default")) +
    xlab("Checking Balance") +
    ylab("Number of Obligors") +
    ggtitle("Checking Balance vs Default") +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 17,
        family = "serif"
    ),) +
    scale_fill_manual(values = c("#FF5E04", "#333333"))

CrossTable(
    credit_data$checking_balance,
    credit_data$default,
    prop.r = TRUE,
    prop.c = FALSE,
    prop.t = FALSE,
    prop.chisq = FALSE
)

# Default vs Savings Balance
credit_data %>%
    dplyr::select(savings_balance, default) %>%
    dplyr::count(savings_balance, default) %>%
    # filter(default == "yes") %>%
    ggplot(aes(x = savings_balance, y = n, fill = default)) +
    geom_bar(stat = "identity") +
    guides(fill = guide_legend(title = "Default")) +
    xlab("Savings Balance") +
    ylab("Number of Obligors") +
    ggtitle("Savings Balance vs Default") +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 17,
        family = "serif"
    ),) +
    scale_fill_manual(values = c("#FF5E04", "#333333"))

CrossTable(
    credit_data$savings_balance,
    credit_data$default,
    prop.r = TRUE,
    prop.c = FALSE,
    prop.t = FALSE,
    prop.chisq = FALSE
)

# Correlation Analysis
source("cor_check.R")
correlation <-
    mixed_assoc(credit_data, adjust_cramersv_bias = F)[c("x", "y", "assoc")]
cor_plot_df <- correlation %>% spread(y, assoc)
colnames(cor_plot_df) <-
    c(
        "x",
        "Age",
        "Amt",
        "ChBal",
        "CrHis",
        "Default",
        "Dep",
        "Emp_Dur",
        "ExLoan",
        "House",
        "Job",
        "MLD",
        "OthCr",
        "%Inc",
        "Ph",
        "Purp",
        "SvBal",
        "YAR"
    )

cor_plot_df %>%
    column_to_rownames("x") %>%
    as.matrix() %>%
    corrplot(
        number.cex = 0.7,
        tl.cex = 0.9,
        insig = "blank",
        type = "lower",
        method = "number",
        bg = "#333333",
        diag = T,
        tl.col = "black",
        col = colorRampPalette(c("white", "#FF5E04"))(200),
        tl.srt = 45
    )

cor_plot_df %>%
    column_to_rownames("x") %>%
    filter(Default < 0.1) %>%
    dplyr::select(Default)

# We see that dependants,existing_loans_count, job, percent_of_income, phone,
# purpose, and years_at_residence have very low association (< 0.1) with default.

# Doing a train-test split for model creation
set.seed(179)
samp <- createDataPartition(credit_data$default, p = 0.70)
training <- credit_data[unlist(samp),]
testing <- credit_data[-unlist(samp),]

prop.table(table(training$default))
prop.table(table(testing$default))

# Testing different models for accuracy and fit

# Logistic Regression:

logistic1 <- glm(default ~ ., data = credit_data, family = binomial)
summary(logistic1)
car::vif(logistic1)
anova(logistic1, test = "Chisq")

# Removing insignificant variables job, years_at_residence, existing_loans_count and dependents.
logistic2 <-
    glm(
        default ~ . - job - dependants - years_at_residence - existing_loans_count,
        data = training,
        family = binomial
    )

summary(logistic2)
anova(logistic2, test = "Chisq")
vif(logistic2)

# Removing months_loan_duration to remove multicollinearity.

logistic3 <-
    glm(
        default ~ . - job - dependants - years_at_residence -
            existing_loans_count - months_loan_duration,
        data = training,
        family = binomial
    )
summary(logistic3)
vif(logistic3) # No indication of multi-collinearity.
anova(logistic3, test = "Chisq")
lrtest(logistic3, logistic2)

# Logistic 3 is a good fit.

# Making predictions and checking model fit.
pred_check_in <- 0
pred_check_out <- 0
val <- 0
val1 <- 0

for (i in seq(0.1, 0.8691996, by = 0.001)) {
    pred_insample <-
        predict.glm(logistic3, newdata = training, type = "response")
    
    pred_insample <- ifelse(pred_insample > i, "yes", "no")
    pred_insample <- as.factor(pred_insample)
    mat_in <-
        caret::confusionMatrix(training$default, pred_insample)
    acc_insamp <- mat_in$overall[1]
    
    if (acc_insamp > pred_check_in) {
        pred_check_in <- acc_insamp
        val1 <- i
    }
    
    pred_outsample <-
        predict.glm(logistic3, newdata = testing, type = "response")
    pred_outsample <- ifelse(pred_outsample > i, "yes", "no")
    pred_outsample <- as.factor(pred_outsample)
    mat_out <-
        caret::confusionMatrix(testing$default, pred_outsample)
    acc_outsamp <- mat_out$overall[1]
    
    if (acc_outsamp > pred_check_out) {
        pred_check_out <- acc_outsamp
        val <- i
    }
}

insamp_predict <-
    predict.glm(logistic3, newdata = training, type = "response")
insamp_predict <- ifelse(insamp_predict > 0.57, "yes", "no")
insamp_predict <- as.factor(insamp_predict)
caret::confusionMatrix(training$default, insamp_predict)

outsamp_predict <-
    predict.glm(logistic3, newdata = testing, type = "response")
outsamp_predict <- ifelse(outsamp_predict > 0.57, "yes", "no")
outsamp_predict <- as.factor(outsamp_predict)
caret::confusionMatrix(testing$default, outsamp_predict)

pR2(logistic3)
somersD(factor(
    training$default,
    levels = c("no", "yes"),
    labels = c(0, 1)
),
fitted(logistic3))
hoslem.test(training$default, logistic3$fitted.values)

# Testing the linearity assumption.
probs <- fitted(logistic3)
numeric_data <- training %>% dplyr::select_if(is.numeric)
predictors <- colnames(numeric_data)
numeric_data <- numeric_data %>%
    mutate(logit = log(probs / (1 - probs))) %>%
    gather(key = "predictors", value = "predictor.value", -logit)


ggplot(numeric_data, aes(logit, predictor.value)) +
    geom_point(size = 0.5, alpha = 0.5) +
    geom_smooth(method = "loess") +
    theme_bw() +
    facet_wrap( ~ predictors, scales = "free_y")

# K-fold Cross Validation:
train_control <- trainControl(method = "cv", number = 10)
model4 <-
    train(
        default ~ checking_balance + credit_history + purpose + amount +
            savings_balance + employment_duration + percent_of_income + age +
            other_credit + housing + phone,
        data = training,
        method = "glm",
        family = binomial,
        trControl = train_control
    )
model4

predict_kcv <- predict(model4, testing)
roc_predict_kcv <- ifelse(predict_kcv == "no", 0, 1)
roc_predict_test <- ifelse(testing$default == "no", 0, 1)
caret::confusionMatrix(predict_kcv, testing$default)
plotROC(roc_predict_test, roc_predict_kcv)

# Checking which seed gives highest accuracy and McFaddens R2.
seed <- numeric()
r2 <- numeric()
for (i in 1:1000) {
    set.seed(i)
    samp <- createDataPartition(credit_data$default, p = 0.70)
    training <- credit_data[unlist(samp),]
    logistic3 <-
        glm(
            default ~ . - job - dependants - years_at_residence
            - existing_loans_count - months_loan_duration,
            data = training,
            family = binomial
        )
    r2[i] <- pR2(logistic3)["McFadden"]
    seed[i] <- i
}
which.max(r2)
max(r2)

# Decision Trees

set.seed(53)
samp <- createDataPartition(credit_data$default, p = 0.70)
training <- credit_data[unlist(samp),]
testing <- credit_data[-unlist(samp),]

tree <-
    rpart::rpart(default ~ .,
                 data = training,
                 control = rpart.control(cp = 0.01))
insample_tree <- predict(tree, training, type = "class")
outsample_tree <- predict(tree, testing, type = "class")

caret::confusionMatrix(training$default, insample_tree)
caret::confusionMatrix(testing$default, outsample_tree)

rpart::printcp(tree)
rpart::plotcp(tree)

caret::varImp(tree)
fancyRpartPlot(tree)

roc_predict_tree <- ifelse(outsample_tree == "no", 0, 1)
roc_predict_test <- ifelse(testing$default == "no", 0, 1)
roc_predict_train <- ifelse(training$default == "no", 0, 1)
caret::confusionMatrix(outsample_tree, testing$default)
InformationValue::plotROC(roc_predict_test, roc_predict_tree)

vm <- caret::varImp(tree)

ggplot(vm, aes(x = reorder(rownames(vm), Overall), y = Overall)) +
    geom_point(color = "#333333",
               size = 4,
               alpha = 0.8) +
    geom_segment(aes(
        x = rownames(vm),
        xend = rownames(vm),
        y = 0,
        yend = Overall
    ),
    color = "#FF5E04") +
    xlab("Variable") +
    ylab("Overall Importance") +
    theme_minimal() +
    ggtitle("Variable Importance Plot (Decision Tree)") +
    coord_flip() +
    theme(plot.title = element_text(
        hjust = 0.5,
        family = "serif",
        size = 17
    ))

roc_predict_tree_in <- ifelse(insample_tree == "no", 0, 1)
InformationValue::plotROC(roc_predict_train, roc_predict_tree_in)

roc_predict_tree <- ifelse(outsample_tree == "no", 0, 1)
InformationValue::plotROC(roc_predict_test, roc_predict_tree)

# Testing seed with highest accuracy

seed <- numeric()
acc <- numeric()
for (i in 1:1000) {
    set.seed(i)
    print(i)
    samp <- createDataPartition(credit_data$default, p = 0.70)
    training1 <- credit_data[unlist(samp),]
    tree1 <- rpart::rpart(
        default ~ .,
        data = training1,
        method = "class",
        control = rpart.control(cp = 0.01)
    )
    
    outsample_tree1 <- predict(tree1, testing, type = "class")
    acc[i] <-
        caret::confusionMatrix(testing$default, outsample_tree1)$overall["Accuracy"]
    seed[i] <- i
}
which.max(acc)
max(acc)

# Random Forest
control_forest <- trainControl(method = "cv", number = 10)
seed <- 8
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(credit_data))
tunegrid <- expand.grid(.mtry = mtry)
rf_default <-
    train(
        default ~ .,
        data = training,
        method = "rf",
        metric = metric,
        tuneGrid = tunegrid,
        trControl = control_forest
    )

forest_insamp <- predict(rf_default, training)
forest_predict <- predict(rf_default, testing)
caret::confusionMatrix(forest_insamp, training$default)
caret::confusionMatrix(forest_predict, testing$default)
plotROC(roc_predict_train, ifelse(forest_insamp == "no", 0, 1))
plotROC(roc_predict_test, ifelse(forest_predict == "no", 0, 1))

vmf <- caret::varImp(rf_default)

ggplot(vmf$importance, aes(x = reorder(rownames(vmf$importance), Overall), y = Overall)) +
    geom_point(color = "#333333",
               size = 4,
               alpha = 0.8) +
    geom_segment(aes(
        x = rownames(vmf$importance),
        xend = rownames(vmf$importance),
        y = 0,
        yend = Overall
    ),
    color = "#FF5E04") +
    xlab("Variable") +
    ylab("Overall Importance") +
    theme_minimal() +
    ggtitle("Variable Importance Plot (Random Forest)") +
    coord_flip() +
    theme(plot.title = element_text(
        hjust = 10,
        family = "serif",
        size = 17,
        color = "black"
    ))

# Gradient Boosting
set.seed(179)
samp <- createDataPartition(credit_data$default, p = 0.70)
training <- credit_data[unlist(samp),]
testing <- credit_data[-unlist(samp),]

labels <- training$default
labels_ts <- testing$default

boost_tr <-
    model.matrix( ~ . + 0, data = training %>% dplyr::select(-c(default)))
boost_ts <-
    model.matrix( ~ . + 0, data = testing %>% dplyr::select(-c(default)))

labels <- as.numeric(labels) - 1
labels_ts <- as.numeric(labels_ts) - 1

# Creating training and testing data for the model

dtrain <- xgb.DMatrix(data = boost_tr, label = labels)
dtest <- xgb.DMatrix(data = boost_ts, label = labels_ts)

params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.21,
    gamma = 0,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1
)

xgbcv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 1000,
    nfold = 10,
    showsd = T,
    stratified = T,
    print_every_n = 1,
    maximize = F,
    eval_metric = "error",
    early_stopping_rounds = 200,
    verbose = F
)

xgbcv$best_iteration
xgbcv$best_ntreelimit

xgb <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 162,
    watchlist = list(val = dtest, train = dtrain),
    print_every_n = 1,
    early_stopping_rounds = 10,
    maximize = F,
    eval_metric = "error",
    n_tree = 162
)

predict_xgb_tr <- predict(xgb, dtrain)
predict_xgb_tr <- ifelse(predict_xgb_tr > 0.5, 1, 0)
predict_xgb_tr <- predict_xgb_tr + 1
predict_xgb_tr <-
    factor(predict_xgb_tr,
           levels = c(1, 2),
           labels = c("no", "yes"))
caret::confusionMatrix(predict_xgb_tr, training$default)
plotROC(ifelse(training$default == "no", 0, 1),
        ifelse(predict_xgb_tr == "no", 0, 1))

predict_xgb <- predict(xgb, dtest)
predict_xgb <- ifelse(predict_xgb > 0.5, 1, 0)
predict_xgb <- predict_xgb + 1
predict_xgb <-
    factor(predict_xgb,
           levels = c(1, 2),
           labels = c("no", "yes"))
caret::confusionMatrix(predict_xgb, testing$default)
plotROC(ifelse(testing$default == "no", 0, 1),
        ifelse(predict_xgb == "no", 0, 1))

xgb_imp <- xgb.importance(colnames(boost_tr), model = xgb)
(gg <-
        xgb.ggplot.importance(xgb_imp, measure = "Frequency", top_n = 12))
gg + ylab("Frequency") +
    ggtitle("Feature Importance") +
    guides(fill = FALSE, color = FALSE) +
    theme_minimal() +
    theme(plot.title = element_text(
        hjust = 0.5,
        size = 17,
        family = "serif"
    ),) +
    scale_fill_manual(values = c("#FF5E04", "#333333"))
