# GAN Project

This project implements a Generative Adversarial Network (GAN) for synthesizing new data using the MNIST dataset. The GAN consists of a generator and a discriminator model, which are trained together in a competitive manner.

## Project Structure

```
gan-project
├── src
│   ├── __init__.py          # Marks the directory as a Python package
│   ├── main.py              # Entry point for the GAN project
│   ├── models.py            # Defines the generator and discriminator models
│   ├── train.py             # Contains the training logic for the GAN
│   ├── utils.py             # Utility functions for the project
│   └── data
│       └── dataset.py       # Handles loading and preprocessing the dataset
├── requirements.txt         # Lists the project dependencies
└── README.md                # Documentation for the project
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gan-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the GAN training process, execute the following command:
```
python src/main.py
```

## Models

- **Generator**: The generator model creates new data samples from random noise.
- **Discriminator**: The discriminator model evaluates the authenticity of the generated samples.

## Training Process

The training process involves alternating between training the discriminator and the generator. The discriminator learns to distinguish between real and fake samples, while the generator learns to produce samples that are indistinguishable from real data.

## License

This project is licensed under the MIT License.