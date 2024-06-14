# RnD_L2B - Research and Development: LLM to Behaviour
## Overview
RnD_L2B is a research project that investigates the use of Large Language Models (LLM), like T5 and BERT, integrated with Domain-Specific Languages (DSL) to understand and generate dynamic behaviors from textual data. This project focuses on exploring natural language understanding by tackling ambiguities in command interpretation and behavior generation. The project specifically addresses the temporal aspects of task sequencing and execution based on natural language commands.

## Repository Structure
* `data/`: Contains CSV files for training and classification data.

* `figures/`: Stores visual representations of data and model performance.

* `logs/`: Contains training logs for various model configurations.

* `models/`: Stores fine-tuned models and tokenizers.

* `notebooks/`: Contains Jupyter notebooks for exploratory data analysis and experiments.

* `results/`: Stores training results, including loss evaluations and model checkpoints.

* `src/`: Source files for the main project code, including:

  * `inference/`: Scripts for model inference.
  * `training/`: Scripts for training the models.

requirements.txt: Lists required Python packages for the project.

## Project Details
The training for the different models can be found in the `src/training` directory. This includes scripts for training the two  T5 model (`train_t5_label_concat.py`/`train_y5_text_only.py`) and the BERT model (`train_bert.py`).

The project uses a BERT embedding to better understand temporal dependencies and provide a temporal tagging mechanism to augment the input prompt. This aims to helps in improving the accuracy and efficiency of the task sequencing and execution process. The temporal tagging information is integrated into the input prompt for the T5 model to enhance the model's ability to interpret and generate correctly sequenced tasks.

## Setting Up the Environment

Set up the environmet through Anaconda with `conda_requirements.txt`

Alternatively, to manually set up the environment for this project, follow these steps:

1. Create a new conda environment:
   ```bash
   conda create --name myenv python=3.8
   conda activate
   ```
2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
