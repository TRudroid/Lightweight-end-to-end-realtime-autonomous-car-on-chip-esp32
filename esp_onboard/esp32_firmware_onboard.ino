/*
 * TinyNav onboard — ESP32-S3 + OV2640
 *
 * Fixes vs previous:
 *  - Motor PWM floor high enough to overcome stall (~110+)
 *  - Kick-start whenever starting from stop
 *  - Inference on its own task; loop keeps applying last command so wheels don't die
 *  - Large buffers in PSRAM; deep stack for TFLite
 *
 * Note: paper uses 24x24x20 depth → ~30 ms INT8. We match that spatial size
 * by downsampling OV2640 gray ROI to 24x24 (not 128x128 — that was ~9s).
 */

SET_LOOP_TASK_STACK_SIZE(16 * 1024);

#define CAMERA_MODEL_ESP32S3_EYE

#if defined(CAMERA_MODEL_AI_THINKER)
#define PIN_MOTOR_A_PWM 12
#define PIN_MOTOR_B_PWM 13
#elif defined(CAMERA_MODEL_ESP32S3_EYE)
#define PIN_MOTOR_A_PWM 42
#define PIN_MOTOR_B_PWM 41
#endif

#include "camera_pins.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <EloquentTinyML.h>
#include "model_data.h"
#include <math.h>
#include <string.h>

#define NUM_FRAMES 20
#define FRAME_H 24
#define FRAME_W 24
#define FRAME_BYTES (FRAME_H * FRAME_W)
#define MODEL_INPUT_SIZE (FRAME_H * FRAME_W * NUM_FRAMES)
#define MODEL_OUTPUT_SIZE 2
#define ARENA_SIZE (192 * 1024)

Eloquent::TinyML::TfLite<MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, ARENA_SIZE> ml;
TfLiteTensor *input_tensor = nullptr;
TfLiteTensor *output_tensor = nullptr;
float input_scale = 1.0f;
int32_t input_zero_point = 0;
float output_scale = 1.0f;
int32_t output_zero_point = 0;

const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";
const uint16_t udpPortLaptop = 3000;
const uint16_t udpPortCam = 3001;
WiFiUDP Udp;
IPAddress pcIP;
bool pcConnected = false;

// Drive tuning — match collect_data (BASE~80); slow = easier to stay in grey lane
uint8_t BASE_SPEED = 85;
const uint8_t MIN_DRIVE_PWM = 70;
const uint8_t MAX_DRIVE_PWM = 120;
const uint8_t KICK_PWM = 130;
const float STEER_DEADZONE = 0.05f;
const float STEER_GAIN = 1.35f;   // amplify AI steering for sharper lane corrections
#define INVERT_STEERING false
#define FLIP_CAMERA_H false
#define FLIP_CAMERA_V false

SemaphoreHandle_t stateMutex = NULL;
volatile bool auto_pilot = true;
volatile float current_steering = 0.0f;
volatile float current_throttle = 0.0f;
volatile uint32_t current_inference_time = 0;
volatile bool has_inference = false;
volatile bool new_telemetry_ready = false;
volatile bool inference_busy = false;
volatile bool on_grey_lane = false;
volatile int grey_pixel_count = 0;
volatile uint32_t last_infer_done_ms = 0;

// With 24x24 (paper size), inference should be tens of ms — keep driving until stale.
const uint32_t AI_STALE_MS = 250;

// Grey-lane detector thresholds (grayscale 0..255). Tune if lighting changes.
const uint8_t GREY_MIN = 55;     // darker than this ≈ green off-track
const uint8_t GREY_MAX = 195;    // brighter ≈ white line / glare (still OK as lane cue)
const int GREY_MIN_PIXELS = 35; // ~bottom ROI grey pixels on 24x24
const int OFFTRACK_HOLD_MS = 250; // require loss for this long before hard stop
uint32_t offtrack_since_ms = 0;

uint8_t *frame_ring = nullptr;
uint8_t *gray24 = nullptr;
uint8_t *telemetry_frame = nullptr;
uint8_t *telemetry_packet = nullptr;
volatile int buffer_index = 0;
volatile int frame_count = 0;

#define pwmFrequency 800
#define pwmResolution 8
#define STOP 0
#define STRAIGHT 1
#define LEFT 3
#define RIGHT 4

class DCMotorControl {
  uint8_t pinA, pinB;
public:
  DCMotorControl(uint8_t a, uint8_t b) : pinA(a), pinB(b) {
    ledcAttachChannel(pinA, pwmFrequency, pwmResolution, 2);
    ledcAttachChannel(pinB, pwmFrequency, pwmResolution, 3);
  }
  void SettingMotor(uint8_t sa, uint8_t sb) {
    ledcWrite(pinA, sa);
    ledcWrite(pinB, sb);
  }
  void CarMovementControl(uint8_t direction, uint8_t speed, int8_t alpha) {
    int a = speed;
    int b = speed;
    switch (direction) {
      case STRAIGHT: a = speed; b = speed; break;
      case LEFT:     a = speed - alpha; b = speed; break;
      case RIGHT:    a = speed; b = speed - alpha; break;
      default:       a = 0; b = 0; break;
    }
    if (a < 0) a = 0;
    if (b < 0) b = 0;
    if (a > 255) a = 255;
    if (b > 255) b = 255;
    SettingMotor((uint8_t)a, (uint8_t)b);
  }
};

DCMotorControl *motorControl = nullptr;
uint8_t last_sent_speed = 0;

static camera_config_t camera_config = {
  .pin_pwdn = PWDN_GPIO_NUM,
  .pin_reset = RESET_GPIO_NUM,
  .pin_xclk = XCLK_GPIO_NUM,
  .pin_sscb_sda = SIOD_GPIO_NUM,
  .pin_sscb_scl = SIOC_GPIO_NUM,
  .pin_d7 = Y9_GPIO_NUM,
  .pin_d6 = Y8_GPIO_NUM,
  .pin_d5 = Y7_GPIO_NUM,
  .pin_d4 = Y6_GPIO_NUM,
  .pin_d3 = Y5_GPIO_NUM,
  .pin_d2 = Y4_GPIO_NUM,
  .pin_d1 = Y3_GPIO_NUM,
  .pin_d0 = Y2_GPIO_NUM,
  .pin_vsync = VSYNC_GPIO_NUM,
  .pin_href = HREF_GPIO_NUM,
  .pin_pclk = PCLK_GPIO_NUM,
  .xclk_freq_hz = 20000000,
  .ledc_timer = LEDC_TIMER_0,
  .ledc_channel = LEDC_CHANNEL_0,
  .pixel_format = PIXFORMAT_GRAYSCALE,
  .frame_size = FRAMESIZE_QVGA,
  .jpeg_quality = 12,
  .fb_count = 2,
  .fb_location = CAMERA_FB_IN_PSRAM,
  .grab_mode = CAMERA_GRAB_LATEST,
};

static void *psram_or_die(size_t n, const char *tag) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) {
    Serial.printf("PSRAM alloc failed: %s (%u)\n", tag, (unsigned)n);
    while (true) delay(1000);
  }
  memset(p, 0, n);
  return p;
}

void applySteerThrottle(float steering, float throttle) {
  if (!motorControl) return;

  // Never move without grey lane + a fresh AI burst window
  if (!on_grey_lane || !has_inference) {
    motorControl->CarMovementControl(STOP, 0, 0);
    last_sent_speed = 0;
    return;
  }

  uint32_t age = millis() - last_infer_done_ms;
  if (age > AI_STALE_MS) {
    motorControl->CarMovementControl(STOP, 0, 0);
    last_sent_speed = 0;
    return;
  }

  if (INVERT_STEERING) steering = -steering;
  steering = fmaxf(-1.0f, fminf(1.0f, steering * STEER_GAIN));
  throttle = fmaxf(0.0f, fminf(1.0f, throttle));

  float steer_val = fabsf(steering);
  throttle *= (1.0f - 0.40f * steer_val);
  if (throttle < 0.20f) throttle = 0.20f;
  if (throttle > 0.50f) throttle = 0.50f;

  float speed_f = MIN_DRIVE_PWM + throttle * (float)(BASE_SPEED - MIN_DRIVE_PWM);
  if (speed_f > MAX_DRIVE_PWM) speed_f = MAX_DRIVE_PWM;
  uint8_t target_speed = (uint8_t)speed_f;

  uint8_t target_dir = STRAIGHT;
  int8_t target_alpha = 0;
  if (steer_val > STEER_DEADZONE) {
    target_dir = (steering < 0.0f) ? LEFT : RIGHT;
    target_alpha = (int8_t)(steer_val * target_speed);
  }

  if (last_sent_speed == 0) {
    motorControl->CarMovementControl(STRAIGHT, KICK_PWM, 0);
    delay(70);
  }

  motorControl->CarMovementControl(target_dir, target_speed, target_alpha);
  last_sent_speed = target_speed;
}

void preprocessToGray24(const uint8_t *src, uint8_t *dst) {
  // QVGA 320x240 → drop top 40% → resize to 24x24 (paper spatial size)
  for (int r = 0; r < FRAME_H; r++) {
    int src_r = 96 + (r * 144) / FRAME_H;
    const uint8_t *row = src + src_r * 320;
    for (int c = 0; c < FRAME_W; c++) {
      dst[r * FRAME_W + c] = row[(c * 320) / FRAME_W];
    }
  }
}

// Returns true if enough grey-lane pixels are visible in the lower part of the ROI.
bool detectGreyLane(const uint8_t *img24) {
  int grey = 0;
  int white_edge = 0;
  // Focus on bottom ~60%
  for (int r = FRAME_H * 2 / 5; r < FRAME_H; r++) {
    const uint8_t *row = img24 + r * FRAME_W;
    for (int c = 1; c < FRAME_W - 1; c++) {
      uint8_t p = row[c];
      if (p >= GREY_MIN && p <= GREY_MAX) grey++;
      if (p > 200) white_edge++;
    }
  }
  grey_pixel_count = grey;

  bool ok = (grey >= GREY_MIN_PIXELS) || (grey >= (GREY_MIN_PIXELS * 2 / 3) && white_edge >= 4);
  return ok;
}

void addFrameToBuffer(const uint8_t *src) {
  if (!frame_ring || !stateMutex) return;
  if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(5)) != pdTRUE) return;
  memcpy(&frame_ring[buffer_index * FRAME_BYTES], src, FRAME_BYTES);
  memcpy(telemetry_frame, src, FRAME_BYTES);
  new_telemetry_ready = true;
  buffer_index = (buffer_index + 1) % NUM_FRAMES;
  if (frame_count < 100000) frame_count++;
  xSemaphoreGive(stateMutex);
}

bool fillInputTensor() {
  if (!frame_ring || !input_tensor || frame_count < NUM_FRAMES) return false;
  if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(50)) != pdTRUE) return false;

  int8_t *in = input_tensor->data.int8;
  const int32_t zp = input_zero_point;
  int write_idx = buffer_index;

  for (int f = 0; f < NUM_FRAMES; f++) {
    int src_idx = (write_idx + f) % NUM_FRAMES;
    const uint8_t *src = &frame_ring[src_idx * FRAME_BYTES];
    for (int hw = 0; hw < FRAME_BYTES; hw++) {
      int32_t qi = (int32_t)src[hw] + zp;
      if (qi < -128) qi = -128;
      if (qi > 127) qi = 127;
      in[hw * NUM_FRAMES + f] = (int8_t)qi;
    }
    // Yield so camera/motor on the other core can run during long copy
    if ((f & 3) == 3) {
      vTaskDelay(1);
    }
  }
  xSemaphoreGive(stateMutex);
  return true;
}

void InferenceTask(void *pv) {
  vTaskDelay(pdMS_TO_TICKS(2000));
  Serial.println("InferenceTask started");

  while (true) {
    if (!auto_pilot || frame_count < NUM_FRAMES || !input_tensor) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    inference_busy = true;
    if (!fillInputTensor()) {
      inference_busy = false;
      vTaskDelay(pdMS_TO_TICKS(20));
      continue;
    }

    uint32_t t0 = millis();
    TfLiteStatus st = ml.getInterpreter()->Invoke();
    uint32_t inf_ms = millis() - t0;

    if (st == kTfLiteOk && output_tensor) {
      int8_t *out = output_tensor->data.int8;
      float steer_logit = (out[0] - output_zero_point) * output_scale;
      float throt_logit = (out[1] - output_zero_point) * output_scale;
      float steering = tanhf(steer_logit);
      float throttle = 1.0f / (1.0f + expf(-throt_logit));
      if (throttle > 0.55f) throttle = 0.55f;

      current_steering = steering;
      current_throttle = throttle;
      current_inference_time = inf_ms;
      has_inference = true;
      last_infer_done_ms = millis();
      Serial.printf("inf=%ums steer=%.2f throt=%.2f\n", inf_ms, steering, throttle);
    } else {
      Serial.println("Invoke failed");
    }
    inference_busy = false;
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

void TelemetryTask(void *pv) {
  while (true) {
    int packetSize = Udp.parsePacket();
    if (packetSize > 0) {
      char buf[64];
      int len = Udp.read(buf, sizeof(buf) - 1);
      if (len > 0) {
        buf[len] = 0;
        if (strcmp(buf, "HB") == 0 || strcmp(buf, "HB\n") == 0) {
          pcIP = Udp.remoteIP();
          pcConnected = true;
        } else if (buf[0] == 'A') {
          auto_pilot = true;
        } else if (buf[0] == 'M') {
          auto_pilot = false;
          if (motorControl) motorControl->CarMovementControl(STOP, 0, 0);
          last_sent_speed = 0;
        } else if (buf[0] == '+') {
          if (BASE_SPEED < 200) BASE_SPEED += 10;
          Serial.printf("BASE_SPEED=%u\n", BASE_SPEED);
        } else if (buf[0] == '-') {
          if (BASE_SPEED > MIN_DRIVE_PWM) BASE_SPEED -= 10;
          Serial.printf("BASE_SPEED=%u\n", BASE_SPEED);
        }
      }
    }

    if (pcConnected && new_telemetry_ready && telemetry_packet) {
      float st = current_steering;
      float th = current_throttle;
      uint32_t inf = current_inference_time;
      uint8_t mode = auto_pilot ? 1 : 0;
      if (xSemaphoreTake(stateMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        memcpy(telemetry_packet + 21, telemetry_frame, FRAME_BYTES);
        new_telemetry_ready = false;
        xSemaphoreGive(stateMutex);
        telemetry_packet[0] = 'T';
        telemetry_packet[1] = 'M';
        memcpy(telemetry_packet + 2, &st, 4);
        memcpy(telemetry_packet + 6, &th, 4);
        float dummy = 0.0f;
        memcpy(telemetry_packet + 10, &dummy, 4);
        memcpy(telemetry_packet + 14, &inf, 4);
        telemetry_packet[18] = mode;
        uint16_t img_size = FRAME_BYTES;
        memcpy(telemetry_packet + 19, &img_size, 2);
        Udp.beginPacket(pcIP, udpPortLaptop);
        Udp.write(telemetry_packet, 21 + img_size);
        Udp.endPacket();
      }
    }
    vTaskDelay(pdMS_TO_TICKS(40));
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  // Do NOT disableLoopWDT()/disableCore*WDT() — Arduino still calls
  // esp_task_wdt_reset() → spam "task not found".
  // Lengthen TWDT instead so ~10s TFLite Invoke does not reboot.
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
  {
    esp_task_wdt_config_t cfg = {
      .timeout_ms = 60000,
      .idle_core_mask = 0,
      .trigger_panic = true,
    };
    esp_task_wdt_reconfigure(&cfg);
  }
#else
  esp_task_wdt_deinit();
  esp_task_wdt_init(60, true);  // 60 seconds
  enableLoopWDT();
  enableCore0WDT();
  enableCore1WDT();
#endif

  Serial.printf("PSRAM: %u bytes free=%u\n",
                (unsigned)ESP.getPsramSize(), (unsigned)ESP.getFreePsram());

  gray24 = (uint8_t *)psram_or_die(FRAME_BYTES, "gray24");
  frame_ring = (uint8_t *)psram_or_die(NUM_FRAMES * FRAME_BYTES, "frame_ring");
  telemetry_frame = (uint8_t *)psram_or_die(FRAME_BYTES, "telemetry_frame");
  telemetry_packet = (uint8_t *)psram_or_die(21 + FRAME_BYTES, "telemetry_packet");

  if (esp_camera_init(&camera_config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (true) delay(1000);
  }
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, FLIP_CAMERA_V ? 1 : 0);
  s->set_hmirror(s, FLIP_CAMERA_H ? 1 : 0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.println(WiFi.localIP());
  Udp.begin(udpPortCam);

  stateMutex = xSemaphoreCreateMutex();

  Serial.println("Init TFLite Micro...");
  if (!ml.begin(model_data)) {
    Serial.println(ml.errorMessage());
    while (true) delay(1000);
  }
  input_tensor = ml.getInputTensor();
  output_tensor = ml.getOutputTensor();
  input_scale = input_tensor->params.scale;
  input_zero_point = input_tensor->params.zero_point;
  output_scale = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;
  if (input_scale == 0.0f) input_scale = 1.0f;
  if (output_scale == 0.0f) output_scale = 1.0f;
  Serial.printf("in scale=%.6f zp=%d | out scale=%.6f zp=%d\n",
                input_scale, input_zero_point, output_scale, output_zero_point);

  motorControl = new DCMotorControl(PIN_MOTOR_A_PWM, PIN_MOTOR_B_PWM);
  // No motor self-test / no creep — stay stopped until first AI result

  xTaskCreatePinnedToCore(TelemetryTask, "telemetry", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(InferenceTask, "infer", 32768, NULL, 1, NULL, 0);

  Serial.println("TinyNav ready: 24x24x20 paper-size input; STOP until AI");
}

void loop() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) {
    preprocessToGray24(fb->buf, gray24);
    esp_camera_fb_return(fb);
    addFrameToBuffer(gray24);

    bool lane_now = detectGreyLane(gray24);
    uint32_t now = millis();
    if (lane_now) {
      on_grey_lane = true;
      offtrack_since_ms = 0;
    } else {
      if (offtrack_since_ms == 0) offtrack_since_ms = now;
      if (now - offtrack_since_ms >= (uint32_t)OFFTRACK_HOLD_MS) {
        on_grey_lane = false;
      }
    }
  }

  // applySteerThrottle gates: grey + non-stale AI
  if (auto_pilot) {
    applySteerThrottle(current_steering, current_throttle);
  }

  static uint32_t last_lane_log = 0;
  if (millis() - last_lane_log > 800) {
    last_lane_log = millis();
    const char *mode;
    if (!on_grey_lane) mode = "STOP(no-grey)";
    else if (!has_inference) mode = "STOP(wait-AI)";
    else if ((millis() - last_infer_done_ms) > AI_STALE_MS) mode = "STOP(stale-AI)";
    else mode = "AI-DRIVE";
    Serial.printf("%s grey_px=%d steer=%.2f throt=%.2f pwm=%u\n",
                  mode, grey_pixel_count, current_steering, current_throttle,
                  (unsigned)last_sent_speed);
  }

  delay(30); // ~30 Hz motor refresh
}
