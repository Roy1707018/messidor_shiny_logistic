# app.R

library(shiny)
library(foreign)   # for read.arff

#--------------------------
# Load and prepare data
#--------------------------
data_raw <- read.arff("messidor_features.arff")

# Rename columns to valid R names
colnames(data_raw) <- c(paste0("X", 0:18), "Class")

# Make Class a factor: 0 = No_DR, 1 = DR
data_raw$Class <- factor(data_raw$Class,
                         levels = c(0, 1),
                         labels = c("No_DR", "DR"))

#--------------------------
# UI
#--------------------------
ui <- fluidPage(
  titlePanel("Diabetic Retinopathy Classification (Logistic Regression)"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Model Parameters"),
      
      # Train / test split
      sliderInput("train_ratio", "Train / Test Split (Train %):",
                  min = 0.5, max = 0.9, value = 0.8, step = 0.05),
      
      # Threshold slider
      sliderInput("threshold", "Classification Threshold (P(DR) cutoff):",
                  min = 0.1, max = 0.9, value = 0.5, step = 0.05),
      
      actionButton("train_btn", "Train / Update Model")
    ),
    
    mainPanel(
      tabsetPanel(
        
        # ----------------------------------------------------------
        # Introduction Tab
        # ----------------------------------------------------------
        tabPanel("Introduction",
                 h3("About the Dataset"),
                 p("This app uses the Messidor Features dataset. Each row is a patient, and the columns are numeric features extracted from retinal fundus images."),
                 p("The target variable is Class: 0 = No_DR (no diabetic retinopathy), 1 = DR (presence of diabetic retinopathy)."),
                 
                 h3("Modeling Methodology"),
                 p("We use logistic regression to model the probability that a patient has diabetic retinopathy based on image-derived features."),
                 p("The model outputs P(DR) between 0 and 1. We then choose a classification threshold."),
                 p("If P(DR) is greater than or equal to the threshold, we classify the patient as DR; otherwise as No_DR."),
                 p("By changing the threshold and the train/test split in the left panel, you can see how the confusion matrix, accuracy, sensitivity, and specificity change. This demonstrates the impact of model parameters according to the data science life cycle.")
        ),
        
        # ----------------------------------------------------------
        # Data Overview Tab
        # ----------------------------------------------------------
        tabPanel("Data Overview",
                 h4("Summary of Variables"),
                 verbatimTextOutput("data_summary"),
                 h4("First 10 Rows"),
                 tableOutput("data_head")
        ),
        
        # ----------------------------------------------------------
        # Model Results Tab
        # ----------------------------------------------------------
        tabPanel("Model Results",
                 h4("Confusion Matrix"),
                 tableOutput("conf_matrix"),
                 
                 h4("Performance Metrics"),
                 verbatimTextOutput("metrics_text"),
                 
                 h4("Logistic Regression Summary"),
                 verbatimTextOutput("model_summary")
        ),
        
        # ----------------------------------------------------------
        # NEW TAB: Full Report PDF
        # ----------------------------------------------------------
        tabPanel("Full Report",
                 h3("Case Study Report"),
                 p("Below is the complete case study report as a PDF."),
                 
                 tags$iframe(
                   src = "case_study_3.pdf",   # MUST BE IN www/ FOLDER
                   style = "width:100%; height:700px; border:none;"
                 )
        )
        
      )
    )
  )
)

#--------------------------
# SERVER
#--------------------------
server <- function(input, output, session) {
  
  # Reactive: split data into train/test according to train_ratio
  split_data <- reactive({
    set.seed(123)  # reproducible
    n <- nrow(data_raw)
    train_size <- floor(input$train_ratio * n)
    idx <- sample(seq_len(n), size = train_size)
    
    list(
      train = data_raw[idx, ],
      test  = data_raw[-idx, ]
    )
  })
  
  # Train logistic regression model when button is clicked
  model_fit <- eventReactive(input$train_btn, {
    dat <- split_data()
    train_data <- dat$train
    test_data  <- dat$test
    
    # Fit logistic regression
    logit_model <- glm(Class ~ ., data = train_data, family = binomial)
    
    # Predict probabilities of DR on test set
    prob_dr <- predict(logit_model, newdata = test_data, type = "response")
    
    # Convert probabilities to classes using current threshold
    pred_class <- ifelse(prob_dr >= input$threshold, "DR", "No_DR")
    pred_class <- factor(pred_class, levels = levels(train_data$Class))
    
    list(
      model      = logit_model,
      test_data  = test_data,
      prob_dr    = prob_dr,
      pred_class = pred_class
    )
  })
  
  #--------------------------
  # Data overview outputs
  #--------------------------
  output$data_summary <- renderPrint({
    summary(data_raw)
  })
  
  output$data_head <- renderTable({
    head(data_raw, 10)
  })
  
  #--------------------------
  # Model result outputs
  #--------------------------
  output$conf_matrix <- renderTable({
    mf <- model_fit()
    table(True = mf$test_data$Class, Predicted = mf$pred_class)
  })
  
  output$metrics_text <- renderPrint({
    mf <- model_fit()
    cm <- table(True = mf$test_data$Class, Predicted = mf$pred_class)
    
    # Accuracy
    accuracy <- sum(diag(cm)) / sum(cm)
    
    TP <- ifelse("DR" %in% rownames(cm) && "DR" %in% colnames(cm), cm["DR", "DR"], 0)
    FN <- ifelse("DR" %in% rownames(cm) && "No_DR" %in% colnames(cm), cm["DR", "No_DR"], 0)
    TN <- ifelse("No_DR" %in% rownames(cm) && "No_DR" %in% colnames(cm), cm["No_DR", "No_DR"], 0)
    FP <- ifelse("No_DR" %in% rownames(cm) && "DR" %in% colnames(cm), cm["No_DR", "DR"], 0)
    
    sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
    specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
    
    cat("Threshold:", input$threshold, "\n")
    cat("Accuracy  :", round(accuracy * 100, 2), "%\n")
    cat("Sensitivity (TPR, DR correctly detected):",
        ifelse(is.na(sensitivity), "NA", paste0(round(sensitivity * 100, 2), "%")), "\n")
    cat("Specificity (TNR, No_DR correctly detected):",
        ifelse(is.na(specificity), "NA", paste0(round(specificity * 100, 2), "%")), "\n")
  })
  
  output$model_summary <- renderPrint({
    mf <- model_fit()
    summary(mf$model)
  })
}

#--------------------------
# Run the app
#--------------------------
shinyApp(ui = ui, server = server)
