Install dependencies using: pip install -r 'requirements.txt' in data folder

For dataset, refer to data.txt

Model weights are uploaded to the google drive and the link to the drive is pasted in model.txt

Instructions for Running the Program:


- Environment Setup:
This program is implemented in Python and requires several dependencies, including PyTorch and the Hugging Face Transformers library.
To set up the environment and install the necessary dependencies, you can use the following command
pip install -r requirements.txt


- Data Preparation:
The program requires two datasets:
Indiana University Chest X-rays for pretraining the multimodal model.
VQA-RAD (or another dataset for fine-tuning, depending on your setup).
The data loading and preprocessing are handled by the ChestXrayDataset class, which expects the following directory structure:
img_dir: The directory containing the chest X-ray images.
csv_path: A CSV file containing the captions corresponding to the images.
Download the VQA_RAD Image Folder.zip and VQA_RAD Dataset Public.json from VQA_RAD Dataset link. [3]
Extract the image folder into the directory specified by img_dir and the JSON file into the directory specified by json_dir.


- Running the PreTraining:
Once the environment is set up and data is ready, you can initiate the pre-training process by running the main.py script.
The script requires two arguments to specify the paths to the image directory and the CSV file containing captions:
python main.py “/path/to/images” “/path/to/captions.csv”
Replace /path/to/images with the actual path to the directory containing the chest X-ray images, and /path/to/captions.csv with the path to the CSV file that contains the captions for the images.
The program will:
Load the data.
Set up the multimodal model (using a combination of Vision Transformer and BERT).
Start the training loop, periodically saving model checkpoints for recovery or further fine-tuning.

- Model Checkpoints:
During training, model checkpoints will be saved periodically. These checkpoints include:
Model parameters for each individual task (MIM, MLM, ITM, ITCL).
The optimizer state.
The checkpoints will be saved in the directory specified by the model_save_path and encoder_save_path arguments.
After training, you can reload these checkpoints to continue training or fine-tune the model for downstream tasks.


- Running the Fine-tuning (VQA) Process:
After completing the pretraining step, you can fine-tune the model using the VQA-RAD dataset by running the Fine Tuning_VQA.ipynb file.
Load Pretrained Weights:
The pretrained weights from the pretraining task (saved in the encoder_checkpoints directory) will be loaded into the fine-tuning script.
Alternatively, you can download the pretrained weights from Google Drive.
Save the weights in a directory named encoder_checkpoints.
The fine-tuning script expects the following path for the weights:
"encoder_checkpoints/bert_model_checkpoint_epoch_1.pth" "encoder_checkpoints/image_encoder_checkpoint_epoch_1.pth"
