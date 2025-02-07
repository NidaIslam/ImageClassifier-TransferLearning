# 🚀 Image Classifier in PyTorch - Transfer Learning

This repository contains an image classification project built using transfer learning with PyTorch. The project leverages pre-trained deep neural networks (such as VGG, DenseNet, and AlexNet) to extract features from images and its final fully connected (classifier) layers are replaced with a custom classifier on top for fine-tuning on a specific dataset. This approach allows you to achieve high accuracy with less training time and data. The provided scripts allow you to:

 ✅ Train a new classifier on your dataset. \
 ✅ Log training progress with metrics such as training loss, validation loss, and validation accuracy. \
 ✅ Save and load model checkpoints. \
 ✅ Predict the class of an input image, outputting the top-k predicted classes along with their probabilities.


## ✨ Features

- **Transfer Learning:** Utilizes state-of-the-art pre-trained models from `torchvision.models`.
- **Custom Classifier:** Replace the final layer(s) of the pre-trained model with a custom classifier.
- **Hyperparameter Flexibility:** Configure hyperparameters such as learning rate, number of epochs, and hidden units via command-line arguments.
- **GPU Support:** Easily switch between CPU and GPU training/inference.
- **Top-K Predictions:** Retrieve and display the top K predicted classes with probabilities.

## 🔧 Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/Image-Classifier-Transfer-Learning.git
   cd Image-Classifier-Transfer-Learning
  
2. **Install Dependencies:**
Ensure you have Python 3.6 or later installed. Then install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage
### 🎯 Training the Model
To train the model, run the `train.py` script. You can specify the dataset directory, model architecture, hyperparameters, and whether to use GPU.

Example:

  ```bash
  !python train.py --data_dir flowers \
               --save_dir checkpoints \
               --arch vgg19 \
               --learning_rate 0.001 \
               --epochs 5 \
               --hidden_units_1 1024 \
               --hidden_units_2 512 \
               --gpu
   ```
This command will:
- Load your dataset from the specified directory.
- Use the VGG19 architecture as the base model.
- Train for 5 epochs with the given learning rate and hidden unit configuration.
- Save the checkpoint to the default save directory (checkpoints/).

### 🔍 Predicting Classes
Once the model is trained and a checkpoint is saved, use `predict.py` to classify a new image.

Example:
  ```bash
  !python predict.py flowers/test/100/image_07938.jpg \
               checkpoints \
               --arch alexnet \
               --top_k 3 \
               --category_names cat_to_name.json \
               --gpu
  ```

This command will:
- Load the trained model checkpoint.
- Process and classify the input image.
- Display the top 5 predicted classes along with their probabilities.


## 💡 How This Project Can Help You
- Rapid Prototyping: Quickly build a powerful image classifier without starting from scratch.
- Educational Resource: Learn how transfer learning works and how to customize pre-trained networks for your own tasks.
- Adaptability: Easily modify the classifier architecture, hyperparameters, or use different pre-trained models to suit your project needs.
- Scalability: Leverage GPU support to speed up training and inference, making it practical for both small experiments and larger projects.

## 🤝 Contributing
Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, feel free to fork this repository and submit a pull request. 

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙌 Acknowledgments
PyTorch, Torchvision, and Udacity for project inspiration and guidelines.

Happy coding! 🚀
