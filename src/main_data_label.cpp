#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/sys/printk.h>
#include <zephyr/shell/shell.h>
#include <math.h>

#define SAMPLING_RATE_MS 50    // 20Hz
#define WINDOW_SIZE      20    // 1 Second Window
#define STACK_SIZE       2048
#define PRIORITY         7

/* --- Global Label State --- */
// 0: Sitting, 1: Fresh Walking, 2: Tired Walking
static int current_label = 0; 

/* --- Buffers & Sync --- */
static float accel_smv_buf[WINDOW_SIZE];
static float gyro_smv_buf[WINDOW_SIZE];
static float temp_buf[WINDOW_SIZE];
static uint16_t idx = 0;

K_SEM_DEFINE(window_full_sem, 0, 1);
K_MUTEX_DEFINE(data_mutex);

static const struct device *mpu = DEVICE_DT_GET(DT_NODELABEL(mpu6050));

/* --- Helper: Magnitude --- */
float get_magnitude(struct sensor_value *val) {
    float x = (float)sensor_value_to_double(&val[0]);
    float y = (float)sensor_value_to_double(&val[1]);
    float z = (float)sensor_value_to_double(&val[2]);
    return sqrtf(x*x + y*y + z*z);
}

// /* --- Shell Command to Set Label --- */
// static int cmd_set_label(const struct shell *shell, size_t argc, char **argv) {
//     if (argc != 2) {
//         shell_error(shell, "Usage: label <0|1|2>");
//         return -EINVAL;
//     }
//     current_label = atoi(argv[1]);
//     shell_print(shell, "Data Label set to: %d", current_label);
//     return 0;
// }
// SHELL_CMD_REGISTER(label, NULL, "Set data label (0=Sit, 1=Fresh, 2=Tired)", cmd_set_label);

/* --- Data Thread: Sampling --- */
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

/* --- Logging Thread (Instead of Inference) --- */
void logging_thread(void *p1, void *p2, void *p3) {
    while (1) {
        k_sem_take(&window_full_sem, K_FOREVER);
        float a_sum = 0, g_sum = 0, t_sum = 0;
        float a_sq_sum = 0, g_sq_sum = 0;

        k_mutex_lock(&data_mutex, K_FOREVER);
        for (int i = 0; i < WINDOW_SIZE; i++) {
            a_sum += accel_smv_buf[i];
            g_sum += gyro_smv_buf[i];
            t_sum += temp_buf[i];
        }
        float a_mean = a_sum / WINDOW_SIZE;
        float g_mean = g_sum / WINDOW_SIZE;
        float t_avg  = t_sum / WINDOW_SIZE;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            a_sq_sum += powf(accel_smv_buf[i] - a_mean, 2);
            g_sq_sum += powf(gyro_smv_buf[i] - g_mean, 2);
        }
        k_mutex_unlock(&data_mutex);

        float a_std = sqrtf(a_sq_sum / WINDOW_SIZE);
        float g_std = sqrtf(g_sq_sum / WINDOW_SIZE);

        /* Log in CSV format: features + label */
        printk("%.2f,%.2f,%.2f,%.2f,%.2f,%d\n", 
                (double)a_mean, (double)a_std, (double)g_mean, (double)g_std, (double)t_avg, current_label);
    }
}

K_THREAD_DEFINE(data_tid, STACK_SIZE, data_sampling_thread, NULL, NULL, NULL, PRIORITY, 0, 0);
K_THREAD_DEFINE(log_tid, STACK_SIZE, logging_thread, NULL, NULL, NULL, PRIORITY, 0, 0);

int main(void) { 
    return 0; 
}
