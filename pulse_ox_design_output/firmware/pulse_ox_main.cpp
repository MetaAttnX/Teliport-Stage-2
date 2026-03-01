// ===========================================
// Pulse Oximeter Firmware - Main Controller
// File: pulse_ox_main.cpp
// ===========================================

#include <Arduino.h>
#include "MAX30102.h"
#include "SSD1306.h"
#include "BLEHandler.h"

// Pin Definitions
#define PIN_OLED_SDA    21
#define PIN_OLED_SCL    22
#define PIN_SENSOR_INT  19
#define PIN_BUTTON      23

// Global objects
MAX30102 sensor;
SSD1306 display(128, 64, &Wire);
BLEHandler ble;
volatile bool sensorDataReady = false;

// Constants
const float SPO2_MIN = 70.0;
const float SPO2_MAX = 100.0;
const int HR_MIN = 30;
const int HR_MAX = 240;
const int SAMPLE_RATE = 100; // Hz

// Interrupt handler
void IRAM_ATTR sensorISR() {
    sensorDataReady = true;
}

class PulseOximeter {
private:
    float spo2;
    int heartRate;
    float irBuffer[100];
    float redBuffer[100];
    int bufferIndex;
    
    // Moving average filter
    float movingAverage(float* buffer, int size) {
        float sum = 0;
        for(int i = 0; i < size; i++) {
            sum += buffer[i];
        }
        return sum / size;
    }
    
    // Ratio-of-Ratios algorithm for SpO2
    float calculateSpO2(float redAC, float irAC, float redDC, float irDC) {
        float ratio = (redAC / redDC) / (irAC / irDC);
        // Calibration curve: SpO2 = 110 - 25 * ratio
        float calculated = 110.0 - (25.0 * ratio);
        return constrain(calculated, SPO2_MIN, SPO2_MAX);
    }
    
    // Peak detection for heart rate
    int detectHeartRate(float* signal, int length) {
        int peakCount = 0;
        float threshold = 0.5;
        
        for(int i = 1; i < length - 1; i++) {
            if(signal[i] > signal[i-1] && 
               signal[i] > signal[i+1] && 
               signal[i] > threshold) {
                peakCount++;
            }
        }
        
        // Convert peaks to BPM
        float duration = length / SAMPLE_RATE; // seconds
        return (peakCount * 60) / duration;
    }
    
public:
    PulseOximeter() : spo2(0), heartRate(0), bufferIndex(0) {}
    
    void init() {
        Wire.begin();
        sensor.begin();
        sensor.setMode(MAX30102_MODE_SPO2);
        sensor.setSampleRate(SAMPLE_RATE);
        
        display.init();
        display.setFont(ArialMT_Plain_10);
        display.setTextAlignment(TEXT_ALIGN_LEFT);
        
        attachInterrupt(PIN_SENSOR_INT, sensorISR, FALLING);
    }
    
    void processData() {
        if (!sensorDataReady) return;
        
        uint32_t irValue = sensor.getIR();
        uint32_t redValue = sensor.getRed();
        
        // Store in circular buffer
        irBuffer[bufferIndex] = irValue;
        redBuffer[bufferIndex] = redValue;
        bufferIndex = (bufferIndex + 1) % 100;
        
        if (bufferIndex == 0) {
            // Calculate SpO2 and HR every 100 samples
            float irAC = movingAverage(irBuffer, 100);
            float redAC = movingAverage(redBuffer, 100);
            float irDC = movingAverage(irBuffer, 100);
            float redDC = movingAverage(redBuffer, 100);
            
            spo2 = calculateSpO2(redAC, irAC, redDC, irDC);
            heartRate = detectHeartRate(irBuffer, 100);
            
            updateDisplay();
            ble.sendData(spo2, heartRate);
        }
        
        sensorDataReady = false;
    }
    
    void updateDisplay() {
        display.clear();
        display.drawString(0, 0, "SpO2: " + String(spo2) + "%");
        display.drawString(0, 20, "HR: " + String(heartRate) + " BPM");
        display.display();
    }
    
    float getSpO2() { return spo2; }
    int getHeartRate() { return heartRate; }
};

PulseOximeter pox;

void setup() {
    Serial.begin(115200);
    pinMode(PIN_BUTTON, INPUT_PULLUP);
    pox.init();
    Serial.println("Pulse Oximeter initialized");
}

void loop() {
    pox.processData();
    
    if (digitalRead(PIN_BUTTON) == LOW) {
        // Button pressed - send data via BLE
        pox.updateDisplay();
        delay(200); // Debounce
    }
    
    delay(10); // 100Hz sampling
}
