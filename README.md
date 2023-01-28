# Dense retrieval system for general court laws

## Abstract
In this report, we present and discuss the development of a Civil Law dense information retrieval system made to identify a user's specific case and retrieve similar contexts from the available civil laws and case articles. Despite the lack of annotated data, we discover that our system does a fine job of retrieving relevant information based on user questions. According to the results, our Civile-Law-IR model outperforms other models that have been trained on millions of data points but have not been fine-tuned to domain knowledge. The goal of this project is to provide easy access to legal information for the general public. It does not replace any professional help, the information provided is just informative, containing some aspects that can help the individual understand their position and rights.

## Installation
Required environment:
- Unix system
- Python 3.10

Clone this repository: git clone https://github.com/Rowan1697/software-project.git

To download and setup the necessary data and libraries: `./setup.sh`


## Repository structure

- [dataset/](dataset) : once you run `setup.sh`

- [presentations/](presentations) : folder containing all the intermediate presentations as PDF. Each file is labled using the template SoftwarePresentation_[month-and-date].pdf

- [results/](results) : 

- [scripts/](scripst) : all the scripts for preprocessing, training models and evaluating
