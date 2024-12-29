# Microbial Sequencing Time Series Prediction Framework

## Introduction  

This project provides a time series prediction framework specifically designed for microbial OTU (Operational Taxonomic Unit) abundance tables. With this framework, users can analyze microbiota data from different subjects and predict changes in microbial abundance at future time points.

## Data Sources

- **Subject A**: From David, L.A., Materna, A.C., Friedman, J. et al. [Host lifestyle affects human microbiota on daily timescales](https://doi.org/10.1186/gb-2014-15-7-r89) (Genome Biol 15, R89, 2014).  
- **Subjects F4 and M3**: From Caporaso, J.G., Lauber, C.L., Costello, E.K. et al. [Moving pictures of the human microbiome](https://doi.org/10.1186/gb-2011-12-5-r50) (Genome Biol 12, R50, 2011).  

## Directory Structure  
<pre>
.  
├── data                    # Contains original OTU abundance tables and adjacency matrix binary files  
├── checkpoints             # Contains trained weight files and process files  
├── lib                     # Contains model architecture and preprocessing code  
├── results                 # Contains results presented in the paper  
└── examples                # Contains command scripts and cases for reproducing results in the paper  
</pre>

## Installation of Dependencies  

Before using this framework, ensure that all necessary Python libraries are installed:  

```bash
pip install -r requirements.txt  
```

## Download Pre-trained Weights  

Prior to running the program, please download the pre-trained model weights from [Google Drive](https://drive.google.com/drive/folders/19ZM7L4_L7fskJVzECu6tS0jEXR0oD-IS?usp=drive_link).  

Once downloaded, extract the archive and place the contents into the `checkpoints` directory within your project folder. This step is crucial as it ensures that you have the necessary weights for the models to function properly.   

## Usage
The program supports command-line arguments to configure runtime options. Below is a summary of the main parameters:  
<pre>
usage: program.py [-h] [--enable-cuda] [--lstm] [--num_timesteps_input NUM_TIMESTEPS_INPUT]  
                 [--num_timesteps_output NUM_TIMESTEPS_OUTPUT] [-e EPOCHS] [-b BATCH_SIZE]  
                 [-t THREADS] [-l {MSELoss,L1Loss}] -i INPUT -r [RATIO] -s SUBJECT  

optional arguments:
  -h, --help            Show this help message and exit  
  --enable-cuda         Enable CUDA acceleration (default off)  
  --lstm                Use LSTM model (default off)  
  --num_timesteps_input NUM_TIMESTEPS_INPUT  
                        Set the length of timesteps used for training, default is 12  
  --num_timesteps_output NUM_TIMESTEPS_OUTPUT  
                        Set the length of timesteps for prediction output, default is 1  
  -e, --epochs          Set the number of epochs, default is 200  
  -b, --batch_size      Set batch size, default is 8  
  -t, --threads         Set the number of threads to use, default is 120  
  -l, --loss_function   Set the type of loss function used for model training, default is L1Loss  
  -i, --input           Required: Set the input file path  
  -r, --ratio           Set ratio, default is 0.1  
  -s, --subject         Required: Set the subject name  
</pre>

## Contact Information
If you have any questions or need further assistance, please contact:  

Author: Gao Shichen   
Email: gaoshichend@163.com  
Date: 2024.12.28  
