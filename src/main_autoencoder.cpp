#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/sys/printk.h>
#include <math.h>

/* TensorFlow Lite Micro Headers */
#include "auto_data.h" 
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define SAMPLING_RATE_MS 50     // 20Hz
#define WINDOW_SIZE      20     // 1 Second Window
#define STACK_SIZE       4096  // Increased for TFLM stack requirements
#define PRIORITY         7

/* --- Feature Vector Structure --- */
typedef struct {
    float accel_smv_mean;
    float accel_smv_std;
    float gyro_smv_mean;
    float gyro_smv_std;
    float temp_avg;
} feature_vector_t;

/* --- Buffers & Sync --- */
static float accel_smv_buf[WINDOW_SIZE];
static float gyro_smv_buf[WINDOW_SIZE];
static float temp_buf[WINDOW_SIZE];
static uint16_t idx = 0;

K_SEM_DEFINE(window_full_sem, 0, 1);
K_MUTEX_DEFINE(data_mutex);

/* --- TFLM Globals --- */
// Tensor Arena: Memory where TFLM allocates input/output tensors
// alignas(16) ensures the memory is correctly aligned for the ESP32 FPU
alignas(16) static uint8_t tensor_arena[20 * 1024]; 

static const struct device *mpu = DEVICE_DT_GET(DT_NODELABEL(mpu6050));

/* --- Helper Functions --- */
float get_magnitude(struct sensor_value *val) {
    float x = (float)sensor_value_to_double(&val[0]);
    float y = (float)sensor_value_to_double(&val[1]);
    float z = (float)sensor_value_to_double(&val[2]);
    return sqrtf(x*x + y*y + z*z);
}

/* --- Data Thread: Sensor Sampling --- */
void data_sampling_thread(void *p1, void *p2, void *p3) {
    struct sensor_value accel[3], gyro[3], temp;
    while (1) {
        if (sensor_sample_fetch(mpu) == 0) {
            sensor_channel_get(mpu, SENSOR_CHAN_ACCEL_XYZ, accel);
            sensor_channel_get(mpu, SENSOR_CHAN_GYRO_XYZ, gyro);
            sensor_channel_get(mpu, SENSOR_CHAN_DIE_TEMP, &temp);

            k_mutex_lock(&data_mutex, K_FOREVER);
            accel_smv_buf[idx] = get_magnitude(accel);
            gyro_smv_buf[idx]  = get_magnitude(gyro);
            temp_buf[idx]      = (float)sensor_value_to_double(&temp);
            idx++;

            if (idx >= WINDOW_SIZE) {
                idx = 0;
                k_sem_give(&window_full_sem);
            }
            k_mutex_unlock(&data_mutex);
        }
        k_msleep(SAMPLING_RATE_MS);
    }
}

/* --- Inference Thread: Feature Extraction & TFLM --- */
void inference_thread(void *p1, void *p2, void *p3) {
    /* 1. Setup TFLM Interpreter */
    const tflite::Model* model = tflite::GetModel(g_model_tflite);
    static tflite::MicroMutableOpResolver<5> resolver;
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));

    // Most Autoencoders only use these three operations
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddMean(); 

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        printk("TFLM: Failed to allocate tensors!\n");
        return;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    while (1) {
        k_sem_take(&window_full_sem, K_FOREVER);
        
        feature_vector_t f = {0};
        float a_sq_sum = 0, g_sq_sum = 0, t_sum = 0;

        /* 2. Feature Extraction (Mean/StdDev) */
        k_mutex_lock(&data_mutex, K_FOREVER);
        for (int i = 0; i < WINDOW_SIZE; i++) {
            f.accel_smv_mean += accel_smv_buf[i];
            f.gyro_smv_mean  += gyro_smv_buf[i];
            t_sum            += temp_buf[i];
        }
        f.accel_smv_mean /= WINDOW_SIZE;
        f.gyro_smv_mean  /= WINDOW_SIZE;
        f.temp_avg        = t_sum / WINDOW_SIZE;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            a_sq_sum += powf(accel_smv_buf[i] - f.accel_smv_mean, 2);
            g_sq_sum += powf(gyro_smv_buf[i] - f.gyro_smv_mean, 2);
        }
        k_mutex_unlock(&data_mutex);

        f.accel_smv_std = sqrtf(a_sq_sum / WINDOW_SIZE);
        f.gyro_smv_std  = sqrtf(g_sq_sum / WINDOW_SIZE);

        /* Log data in CSV format for training your Autoencoder */
        printk("%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                (double)f.accel_smv_mean, (double)f.accel_smv_std,
                (double)f.gyro_smv_mean, (double)f.gyro_smv_std,
                (double)f.temp_avg);

        /* 3. Prepare Input Tensor */
        input->data.f[0] = f.accel_smv_mean;
        input->data.f[1] = f.accel_smv_std;
        input->data.f[2] = f.gyro_smv_mean;
        input->data.f[3] = f.gyro_smv_std;
        input->data.f[4] = f.temp_avg;

        /* 4. Run Inference */
        if (interpreter.Invoke() != kTfLiteOk) {
            printk("TFLM: Inference failed!\n");
            continue;
        }

        /* 5. Calculate Reconstruction Error (MSE) */
        float mse = 0;
        for (int i = 0; i < 5; i++) {
            printk("F[%d]: In: %.2f | Out: %.2f\n", i, (double)input->data.f[i], (double)output->data.f[i]);
            float diff = input->data.f[i] - output->data.f[i];
            mse += (diff * diff);
        }
        mse /= 5.0f;

        /* 6. Output Results */
        printk("Inference Done | MSE: %.6f | Temp: %.1f C\n", (double)mse, (double)f.temp_avg);
        
        // Example logic: Flag an anomaly if MSE exceeds your threshold
        if (mse > 0.05) { // Use the threshold from your Colab training script
            printk("ALERT: Unusual Activity or Fatigue Detected!\n");
        }
    }
}

/* --- Thread Definitions --- */
K_THREAD_DEFINE(data_tid, STACK_SIZE, data_sampling_thread, NULL, NULL, NULL, PRIORITY, 0, 0);
K_THREAD_DEFINE(inf_tid, STACK_SIZE, inference_thread, NULL, NULL, NULL, PRIORITY, 0, 0);

int main(void) { 
    printk("Wearable Fatigue Monitor Started...\n");
    return 0; 
}