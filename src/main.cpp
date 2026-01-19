#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/sys/printk.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Bluetooth Headers */
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>

/* TensorFlow Lite Micro Headers */
#include "cnn_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#define SAMPLING_RATE_MS 50   // 20Hz 
#define WINDOW_SIZE      100  // 5 Second Window

#define STACK_SIZE       8192  
#define PRIORITY         7

#define CONFIDENCE_THRESHOLD 0.50f  // Model must be 75% sure to switch
#define PERSISTENCE_COUNT    2      // Activity must stay the same for 3 windows

static int sitting_count = 0;
static const char* last_stable_activity = "Sitting";

/* --- BLE Service & Characteristic Definitions --- */
static struct bt_uuid_128 fatigue_svc_uuid = BT_UUID_INIT_128(
	BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef0));

static struct bt_uuid_128 fatigue_char_uuid = BT_UUID_INIT_128(
	BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef1));

static char ble_string_buffer[32];

BT_GATT_SERVICE_DEFINE(fatigue_svc,
	BT_GATT_PRIMARY_SERVICE(&fatigue_svc_uuid),
	BT_GATT_CHARACTERISTIC(&fatigue_char_uuid.uuid,
			       BT_GATT_CHRC_NOTIFY,
			       BT_GATT_PERM_READ,
			       NULL, NULL, ble_string_buffer),
	BT_GATT_CCC(NULL, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
);

/* Define Advertising Data */
static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

/* Define Scan Response Data (Optional: includes the Service UUID) */
static const struct bt_data sd[] = {
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, 
        BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef0)),
};

/* --- Global Buffers & Scalers --- */
static float scaler_mean[] = {10.16861631, 2.18992693, 1.0458362, 0.50270241, 30.34509955};
static float scaler_std[]  = {0.92505035, 1.66408874, 0.95093265, 0.42027198, 2.13042094};
static float accel_smv_buf[WINDOW_SIZE];
static float gyro_smv_buf[WINDOW_SIZE];
static float temp_buf[WINDOW_SIZE];
static uint16_t idx = 0;

K_SEM_DEFINE(window_full_sem, 0, 1);
K_MUTEX_DEFINE(data_mutex);
alignas(16) static uint8_t tensor_arena[60 * 1024]; 
static const struct device *mpu = DEVICE_DT_GET(DT_NODELABEL(mpu6050));

/* --- Sensor Helper --- */
float get_magnitude(struct sensor_value *val) {
    float x = (float)sensor_value_to_double(&val[0]);
    float y = (float)sensor_value_to_double(&val[1]);
    float z = (float)sensor_value_to_double(&val[2]);
    return sqrtf(x*x + y*y + z*z);
}

/* --- Statistics Helper --- */
void calculate_stats(float *buf, int size, float *mean, float *std) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += buf[i];
    }
    *mean = sum / size;
    
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = buf[i] - *mean;
        var += diff * diff;
    }
    *std = sqrtf(var / size);
}

float calculate_mean(float *buf, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += buf[i];
    }
    return sum / size;
}

/* --- Sampling Thread --- */
void data_sampling_thread(void *p1, void *p2, void *p3) {
    struct sensor_value accel[3], gyro[3], temp;
    static int print_counter = 0;
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
            
            /* Print every 5 seconds (5000ms / 50ms = 100 samples) */
            if (idx >= WINDOW_SIZE) { 
                idx = 0; 
                float accel_mean, accel_std, gyro_mean, gyro_std, temp_mean;
                calculate_stats(accel_smv_buf, WINDOW_SIZE, &accel_mean, &accel_std);
                calculate_stats(gyro_smv_buf, WINDOW_SIZE, &gyro_mean, &gyro_std);
                temp_mean = calculate_mean(temp_buf, WINDOW_SIZE);
                
                printk("Accel Mean: %.3f Std: %.3f | Gyro Mean: %.3f Std: %.3f | Temp: %.2f\n",
                       accel_mean, accel_std, gyro_mean, gyro_std, temp_mean);
                
                k_sem_give(&window_full_sem); 
            }
            k_mutex_unlock(&data_mutex);
        }

        k_msleep(SAMPLING_RATE_MS);
    }
}

/* --- Inference & BLE Notify Thread --- */
void inference_thread(void *p1, void *p2, void *p3) {
    const tflite::Model* model = tflite::GetModel(g_model_data);
    static tflite::MicroMutableOpResolver<10> resolver;

    resolver.AddExpandDims(); 
    resolver.AddReshape();
    resolver.AddConv2D();      
    resolver.AddMaxPool2D();   
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, sizeof(tensor_arena));
    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Define hysteresis variables here
    static int stability_counter = 0;
    static const char* candidate_activity = "Sitting";
    static const char* last_stable_activity = "Sitting";

    /* Start Bluetooth Advertising */
    bt_enable(NULL);
    int err = bt_le_adv_start(BT_LE_ADV_PARAM(BT_LE_ADV_OPT_CONN, 
                                               BT_GAP_ADV_FAST_INT_MIN_2, 
                                               BT_GAP_ADV_FAST_INT_MAX_2, NULL), 
                                               ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    if (err) {
        printk("Advertising failed to start (err %d)\n", err);
        return;
    }

    while (1) {
        k_sem_take(&window_full_sem, K_FOREVER);

        /* 1. Feature Pre-processing & Physical Check */
        float a_sum = 0.0f;
        float a_sq_sum = 0.0f;

        k_mutex_lock(&data_mutex, K_FOREVER);
        
        // First pass: Calculate Mean (Average)
        for (int i = 0; i < WINDOW_SIZE; i++) {
            a_sum += accel_smv_buf[i];
        }
        float a_mean = a_sum / WINDOW_SIZE;

        // Second pass: Calculate Variance and Load Tensor
        for (int i = 0; i < WINDOW_SIZE; i++) {
            // Calculate square of the difference from mean
            a_sq_sum += powf(accel_smv_buf[i] - a_mean, 2);

            // Existing Tensor Mapping logic
            int base = i * 5;
            input->data.f[base + 0] = (accel_smv_buf[i] - scaler_mean[0]) / scaler_std[0];
            input->data.f[base + 1] = 0.0f; 
            input->data.f[base + 2] = (gyro_smv_buf[i] - scaler_mean[2]) / scaler_std[2];
            input->data.f[base + 3] = 0.0f;
            input->data.f[base + 4] = (temp_buf[i] - scaler_mean[4]) / scaler_std[4];
        }
        k_mutex_unlock(&data_mutex);

        // Calculate final standard deviation
        float accel_std_val = sqrtf(a_sq_sum / WINDOW_SIZE);

        /* 2. Run Inference and apply the Physical Override */
        if (interpreter.Invoke() == kTfLiteOk) {
            float prob_sit   = output->data.f[0];
            float prob_fresh = output->data.f[1];
            float prob_tired = output->data.f[2];

            const char* current_prediction;

            /* Logic: Priority to Physical Reality */
            if (accel_std_val < 0.2f) {
                // Physical sitting override
                current_prediction = "Sitting";
            } else {
                // Model-based prediction for walking
                if (prob_sit > prob_fresh && prob_sit > prob_tired) {
                    current_prediction = "Sitting";
                } else if (prob_fresh > prob_tired) {
                    current_prediction = "Fresh";
                } else {
                    current_prediction = "Tired";
                }
            }

            printk("Inference Results - Sitting: %.2f, Fresh: %.2f, Tired: %.2f --> %s\n", 
                   prob_sit, prob_fresh, prob_tired, current_prediction);

            /* --- HYSTERESIS FILTER --- */
            if ((prob_fresh > 0.95f || prob_tired > 0.95f) && accel_std_val > 1.0f) {
                last_stable_activity = current_prediction;
                stability_counter = 0; 
            } else {
                /* 3. Standard Hysteresis for lower confidence/transitions */
                if (strcmp(current_prediction, candidate_activity) == 0) {
                    stability_counter++;
                } else {
                    candidate_activity = current_prediction;
                    stability_counter = 0;
                }

                if (stability_counter >= PERSISTENCE_COUNT) {
                    last_stable_activity = candidate_activity;
                }
            }

            /* 4. BLE Update */
            snprintf(ble_string_buffer, sizeof(ble_string_buffer), "Act: %s", last_stable_activity);
            bt_gatt_notify(NULL, &fatigue_svc.attrs[1], ble_string_buffer, strlen(ble_string_buffer));
            printk("BLE Sent: %s\n", ble_string_buffer);
        }
        else {
            printk("TFLM: Inference failed!\n");
    }
}

K_THREAD_DEFINE(data_tid, STACK_SIZE, data_sampling_thread, NULL, NULL, NULL, PRIORITY, 0, 0);
K_THREAD_DEFINE(inf_tid, STACK_SIZE, inference_thread, NULL, NULL, NULL, PRIORITY, 0, 0);

int main(void) { 
    return 0; 
}
