# Dense retrieval system for general court laws

## Abstract
In this report, we present and discuss the development of a Civil Law dense information retrieval system made to identify a user's specific case and retrieve similar contexts from the available civil laws and case articles. Despite the lack of annotated data, we discover that our system does a fine job of retrieving relevant information based on user questions. According to the results, our Civile-Law-IR model outperforms other models that have been trained on millions of data points but have not been fine-tuned to domain knowledge. The goal of this project is to provide easy access to legal information for the general public. It does not replace any professional help, the information provided is just informative, containing some aspects that can help the individual understand their position and rights.

## Installation
Requires the following packages:
- Python 3.10

The libraries required for the successful execution of this code are mentioned in `requirements.txt`. In order to install all the libraries:
`pip install -r requirements.txt`.

### Download Dataset
By clicking on this [link](https://drive.google.com/drive/folders/1BOxC6u1HAdYBWFeCVpOU7H5yO-O8UKN5?usp=share_link) you will be redirected to a drive where our dataset has been stocked. This data was created by using CASS dataset's Civile files and it represents the summaries made by the lawyers.


## Repository structure

- [dataset/](dataset) : link to drive containing an archive with all the civile files converted from xml files to story files, using the script "preprocess_original_CASS_xml_files"

- [presentations/](presentations) : folder containing all the intermediate presentations as PDF. Each file is labled using the template SoftwarePresentation_[month-and-date].pdf

- [results/](results) : 

- [scripts/](scripst) : all the scripts for preprocessing, training models and evaluating
