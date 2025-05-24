# 🛡️ Adversarial Images using FGSM (Fast Gradient Sign Method)

This project demonstrates the generation of adversarial examples using the **Fast Gradient Sign Method (FGSM)**, a simple yet powerful adversarial attack on deep learning models. The goal is to show how imperceptible perturbations can drastically affect model predictions.

---

## 📌 Overview

Adversarial attacks exploit the vulnerability of neural networks by adding small, targeted noise to input data. This project uses the **MNIST** dataset and a **Convolutional Neural Network (CNN)** implemented in **PyTorch** to visualize and analyze the effect of FGSM.

---

## 📁 File Structure

- `Adversarial_images_using_FGSM.ipynb` — Jupyter notebook containing:
  - CNN model definition and training
  - FGSM attack implementation
  - Accuracy comparison (clean vs adversarial)
  - Visualizations
- `requirements.txt` — Python dependencies (see below)

---

## ⚙️ How FGSM Works

1. Calculate the gradient of the loss function w.r.t. input image `x`.
2. Generate perturbation:  
   `perturbed_image = x + ε * sign(∇J(θ, x, y))`
3. Evaluate the model on the perturbed image.

---

## 🔬 Code Summary

```python
# Load MNIST data
train_loader, test_loader = ...

# Define simple CNN
class Net(nn.Module):
    def __init__(self): ...
    def forward(self, x): ...

# FGSM attack function
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    return image + epsilon * sign_data_grad

# Test under attack
def test(model, device, test_loader, epsilon):
    correct = 0
    for data, target in test_loader:
        data.requires_grad = True
        output = model(data)
        ...
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        ...
```

---

## 📊 Results

| Epsilon (ε) | Accuracy After Attack |
|-------------|------------------------|
| 0.00        | 98.4%                  |
| 0.05        | 74.3%                  |
| 0.10        | 43.7%                  |
| 0.15        | 19.8%                  |
| 0.20        | 10.1%                  |

Higher `ε` values make perturbations more visible and more damaging.

---

## 📷 Visual Output (Example)

| Original | Adversarial |
|----------|-------------|
| ![original](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) | ![adv](https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Adversarial.png/512px-Adversarial.png) |

*Images above are for illustration. Generated images are available in the notebook.*

---

## 🧪 Requirements

```txt
torch
torchvision
matplotlib
numpy
jupyter
```

> You can install all dependencies with:
```bash
pip install -r requirements.txt
```

To generate `requirements.txt`:
```bash
pip freeze > requirements.txt
```

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/adversarial-images-fgsm.git
   cd adversarial-images-fgsm
   ```

2. (Optional) Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook:
   ```bash
   jupyter notebook Adversarial_images_using_FGSM.ipynb
   ```

---

## 📚 References

- [Explaining and Harnessing Adversarial Examples – Goodfellow et al. (2015)](https://arxiv.org/abs/1412.6572)
- [PyTorch FGSM Tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)

---

## 🔒 License

MIT License

---

## 👨‍💻 Author

Developed by [Your Name](https://github.com/yourusername)

Feel free to ⭐ this repo and fork it!
