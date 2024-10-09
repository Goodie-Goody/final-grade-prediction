# End-to-End Machine Learning Pipeline for Academic Performance Prediction in Nigerian Secondary Schools

## Overview
Our team embarked on a comprehensive project to develop an end-to-end machine learning pipeline aimed at predicting student academic performance in Nigerian senior secondary schools. Leveraging a dataset sourced from Kaggle, we utilized AWS services to create a robust and scalable solution. This project not only highlights our collective technical expertise in data engineering, data science, and web development but also demonstrates our ability to integrate and deploy complex workflows seamlessly.

## Problem Statement
Nigerian senior secondary schools often lack effective warning systems to identify students at risk of underperforming, relying primarily on teachers' and parents' intuition. This project addresses this critical gap by employing machine learning to predict student performance, enabling educators to provide timely and targeted support to improve overall academic outcomes.

## Unique Value Add
### Data Engineering
- **Data Ingestion**: We ingested the raw dataset into an S3 bucket (studentperformance).
- **Data Transformation**: Using AWS Glue, we created an ELT pipeline to clean and transform the data.
- **Data Storage**: The processed data was stored in Amazon Redshift, ensuring efficient querying and analysis.

### Data Analysis
- **Power BI Reporting**: Our data analyst developed a comprehensive Power BI report to analyze various factors affecting student performance, providing actionable insights to support our predictions.

### Machine Learning
- **Model Development**: We experimented with multiple models and chose logistic regression for its balance of performance and interpretability. The model achieved the following scores:
  - **PASS**: Precision 0.88, Recall 0.91, F1-Score 0.89
  - **FAIL**: Precision 0.71, Recall 0.62, F1-Score 0.66
  - **Overall Accuracy**: 0.84
  - **Macro Average**: Precision 0.79, Recall 0.77, F1-Score 0.78
  - **Weighted Average**: Precision 0.83, Recall 0.84, F1-Score 0.84

### Web Application
- **User Interface**: We developed a Flask web application that allows users to upload Excel files and download predictions.
- **Additional Features**: Users can download a data dictionary and sample data, enhancing the overall user experience.

## ETL and Data Warehouse Process
- **Data Extraction**: We extracted sample data from Kaggle and engaged with school administrators to understand factors affecting student performance.
- **Data Development**: Collaboratively ensured data quality and efficiency using Excel for initial transformations and enhancements.
- **AWS S3**: Uploaded our four tables into four folders within the S3 bucket (studentperformance).
- **AWS Redshift**: Created a serverless database and workgroup (studentperformance-group), setting up necessary components like VPC, namespace, and IAM roles. Created four tables in Redshift according to S3 preferences.
- **AWS Glue**: Set up a crawler (student-performance) to load data from S3, attached the ETL-role, created the database and Redshift connection, and configured a job (student) to transfer data from S3 to Redshift.

This setup supports incremental loads, allowing updates, additions, or removals of columns in the Excel file to be reflected in the data warehouse by simply re-uploading to S3 and rerunning the job.

## Impact
This project represents a significant improvement over the previous situation where schools relied solely on intuition. With an accuracy of 84%, our logistic regression model provides a reliable tool for predicting student performance, allowing for timely interventions. The model's performance, particularly in identifying students likely to pass (Precision 0.88, Recall 0.91), ensures that at-risk students receive the support they need.

Moreover, as more data is ingested and the system is continuously monitored and tweaked, the model's accuracy and reliability are expected to improve further. This project showcases our team's ability to manage the entire data lifecycle, from ingestion and transformation to analysis and deployment. It highlights our proficiency with AWS services, machine learning, and web development, making it a valuable addition to our portfolio. Potential employers and collaborators can see our capability to deliver end-to-end solutions that drive meaningful insights and actionable outcomes.

## How to Run the Project
### Prerequisites
- Docker
- Flyctl (for deployment)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Build the Docker image**:
   ```bash
   docker build -t flask-app .
   ```

3. **Run the Docker container**:
   ```bash
   docker run -p 5000:5000 flask-app
   ```

4. **Deploy to Fly.io**:
   ```bash
   fly launch
   fly deploy
   ```

5. **Access the application**:
   Open your web browser and go to `https://waec-prediction.fly.dev/`

## Contributing
We welcome contributions! Please reach out via emails at drkalugoodness@gmail.com
