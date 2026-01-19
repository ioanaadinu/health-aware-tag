# Health-Aware Tag (HAT)

### ESP32-Powered Wearable for Real-time Activity & Fatigue Analysis

The **Health-Aware Tag (HAT)** is an embedded Edge AI solution designed to monitor human physical states using motion sensors. By leveraging an ESP32, the system performs real-time inference to classify activities, identify fatigue levels, and detect movement anomalies locally on the device, ensuring privacy and low latency.

## Project Structure

The repository is organized to handle the full pipeline from raw data to hardware deployment:

* **`/src`**: Main C++ source files. Contains the logic for sensor polling (IMU), signal preprocessing, and the TensorFlow Lite Micro inference loop.
* **`/include`**: Contains the quantized `.h` files. These are the "brains" of the device, converted from Python into C-byte arrays for the ESP32 memory.
* **`/scripts`**: Python Notebooks used for data cleaning, model architecture design, training, and TFLite conversion.
* **`/sample_data`**: Time-series datasets consisting of Accelerometer and Gyroscope readings used to train the neural network.

---

## How it Works

This project utilizes two distinct types of machine learning to analyze motion data. Understanding the difference between them is key to how the device "thinks".

### 1. CNN (Convolutional Neural Network)
* **Type:** Supervised Learning (Classification).
* **Role:** Identifies **Human Activity Recognition (HAR)** and **Fatigue**.
* **The Logic:** The CNN treats a window of motion data (time-series) similarly to how a computer vision model treats an image. It applies "filters" that slide across the timeline to find specific local patterns, such as the rhythmic acceleration of a stride or the erratic shaking associated with physical exhaustion.
* **Outcome:** Outputs a discrete label (e.g., "Fresh Walking," "Tired Walking," or "Sitting").

### 2. Autoencoder
* **Type:** Unsupervised Learning (Anomaly Detection).
* **Role:** Detects **Unusual Activity** (e.g., falls, seizures).
* **The Logic:** An Autoencoder is trained *only* on "normal" movements. Its job is to compress the input data into a tiny bottleneck (encoding) and then try to reconstruct the original signal perfectly (decoding).
* **The Detection:** If the user performs a movement the model has never seen before, it won't know how to reconstruct it accurately. This results in a high **Reconstruction Error**. If the error passes a certain threshold, the device flags an "Anomaly."
* **Outcome:** Outputs a continuous score representing how "weird" the current movement is compared to the training data.

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Microcontroller** | ESP32 (Dual Core 240MHz) |
| **IMU Sensor** | MPU6050 (Accel + Gyro, 6 axis) |
| **Environment** | Zephyr RTOS |
| **Communication** | BLE |
| **Framework** | TensorFlow Lite Micro |
| **Sampling Rate** | 20 Hz |
| **Inference Window** | 5 seconds |

## Usage

### Monitoring
1. Open the Serial Plotter to see real-time classification results and the reconstruction error from the Autoencoder.
2. Connect to the device via nrfConnect and read received results.
