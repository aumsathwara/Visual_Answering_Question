# Visual Question Answering (VQA) System for Chest X-Ray Images

This repository implements a **Visual Question Answering (VQA)** system trained on chest X-ray images, using multimodal transformers such as **Vision Transformer (ViT)** and **BERT**. The system is designed to answer questions about chest X-ray images using both visual and textual data.

The project consists of two main stages:
1. **Pre-training**: A multimodal model is pre-trained using chest X-ray images and their associated captions.
2. **Fine-tuning**: The pre-trained model is fine-tuned on the VQA-RAD dataset for answering medical questions related to chest X-rays.


## Installation

Follow these steps to set up the environment and run the project.

### Step 1: Install Dependencies

1. Navigate to the `data/` directory and install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Data Preparation

You will need two datasets:
- **Indiana Chest X-ray Dataset**: Used for pre-training the multimodal model.
  - [Indiana Chest X-rays Dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/versions/1)
  
- **VQA-RAD Dataset**: Used for fine-tuning the model.
  - [VQA-RAD Dataset](https://osf.io/89kps/)

1. **Download and prepare the Chest X-ray dataset**:
   - Download the **Indiana Chest X-ray Images** and the associated CSV file with captions.
   - Organize the images in a directory (`img_dir`) and place the CSV file (`captions.csv`) in the appropriate path.

2. **Download and prepare the VQA-RAD dataset**:
   - Download the **VQA-RAD Image Folder.zip** and **VQA-RAD Dataset Public.json**.
   - Extract the image folder and place it in the specified directory (`img_dir`).
   - Place the JSON file in the specified directory (`json_dir`).

### Step 3: Pre-training the Multimodal Model

After setting up the environment and preparing the datasets, you can start pre-training the multimodal model (Vision Transformer + BERT).

1. Run the pre-training script:
    ```bash
    python main.py "/path/to/images" "/path/to/captions.csv"
    ```
    - Replace `/path/to/images` with the path to the chest X-ray images.
    - Replace `/path/to/captions.csv` with the path to the CSV file containing captions for the images.

2. The pretraining script will:
    - Load the datasets.
    - Set up the multimodal model combining **Vision Transformer (ViT)** for image encoding and **BERT** for text encoding.
    - Start the training loop, periodically saving model checkpoints.

### Step 4: Model Checkpoints

During training, the model checkpoints will be saved periodically. These checkpoints include:
- Model weights for different tasks (MIM, MLM, ITM, ITCL).
- Optimizer state.
- The saved checkpoints will be stored in the directory specified by `model_save_path` and `encoder_save_path`.

You can reload these checkpoints to continue training or use them for further fine-tuning.

### Step 5: Fine-tuning on the VQA-RAD Dataset

After pre-training the model, you can fine-tune it on the **VQA-RAD** dataset.

1. **Load Pretrained Weights**: Download the pretrained model weights from Google Drive:
    - [Pretrained BERT and ViT Weights](https://drive.google.com/drive/folders/1eRLodaMzFyocBVN0_7B0miak76asTEKp?usp=drive_link)
    - [Fine-tuned VQA Weights](https://drive.google.com/drive/folders/1Nla-kclA6hhOrC6tN8PNiG0MPC2mioVc?usp=drive_link)

   Save the weights in the directory named `encoder_checkpoints`.

2. **Run the Fine-Tuning**: 
   Fine-tune the model using the **Fine Tuning VQA.ipynb** notebook:
    - The script expects the following pretrained weights:
      - `"encoder_checkpoints/bert_model_checkpoint_epoch_1.pth"`
      - `"encoder_checkpoints/image_encoder_checkpoint_epoch_1.pth"`
    - Load the weights and start fine-tuning the model for VQA tasks.

   You can modify the training parameters such as learning rate, batch size, etc., depending on the fine-tuning needs.

## Dataset Details

### Indiana Chest X-rays Dataset

This dataset is used for pre-training the multimodal model, and it contains X-ray images of the chest with associated captions.

**Link**: [Indiana Chest X-rays Dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/versions/1)

### VQA-RAD Dataset

This dataset is used to fine-tune the model for Visual Question Answering tasks related to medical images. It includes questions and answers related to chest X-ray images.

**Link**: [VQA-RAD Dataset](https://osf.io/89kps/)

## Model Weights

The pretrained and fine-tuned weights used in the project can be downloaded from the following links:

1. **BERT and Vision Transformer Weights**:
   - [Pretrained BERT and ViT Weights](https://drive.google.com/drive/folders/1eRLodaMzFyocBVN0_7B0miak76asTEKp?usp=drive_link)

2. **Fine-tuned VQA Weights**:
   - [Fine-tuned VQA Weights](https://drive.google.com/drive/folders/1Nla-kclA6hhOrC6tN8PNiG0MPC2mioVc?usp=drive_link)

## Running the Model

Once the environment is set up and the data is ready, you can run the **fine-tuned VQA model** to answer questions on chest X-ray images.

1. Load the model from the checkpoint.
2. Pass an image and a corresponding question as input.
3. The model will provide an answer based on the image content.

## License

This project is licensed under the MIT License.

