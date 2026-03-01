#!/usr/bin/env python3
# ===========================================
# Pulse Oximeter Test Bench
# File: test_pulse_ox.py
# ===========================================

import numpy as np
import matplotlib.pyplot as plt
import time
import serial
import sys

class PulseOximeterSimulator:
    """Simulator for testing pulse oximeter algorithms"""
    
    def __init__(self, sampling_rate=100):
        self.fs = sampling_rate
        self.time = 0
        self.heart_rate = 72  # BPM
        self.spo2 = 98  # percentage
        
    def generate_ppg_signal(self, duration_seconds):
        """Generate realistic PPG signal"""
        t = np.linspace(self.time, self.time + duration_seconds, 
                        int(self.fs * duration_seconds))
        
        # Heart rate component
        hr_freq = self.heart_rate / 60.0  # Hz
        cardiac = 0.5 * np.sin(2 * np.pi * hr_freq * t)
        
        # Add harmonics
        cardiac += 0.3 * np.sin(4 * np.pi * hr_freq * t)
        cardiac += 0.2 * np.sin(6 * np.pi * hr_freq * t)
        
        # Respiratory component (modulates amplitude)
        resp_freq = 0.25  # 15 breaths per minute
        respiration = 0.2 * np.sin(2 * np.pi * resp_freq * t)
        
        # Baseline wander and noise
        baseline = 0.1 * np.sin(2 * np.pi * 0.1 * t)
        noise = np.random.normal(0, 0.02, len(t))
        
        # Combine components
        signal = 1.0 + respiration + baseline + cardiac + noise
        
        # IR and Red channels (different absorption for SpO2)
        ir_signal = signal * (1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t))
        red_signal = signal * (1.0 - (self.spo2 / 100.0) * 0.3)
        
        self.time += duration_seconds
        return t, ir_signal, red_signal
    
    def calculate_spo2(self, ir_ac, red_ac, ir_dc, red_dc):
        """Implement ratio-of-ratios algorithm"""
        ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
        spo2 = 110 - 25 * ratio  # Calibration curve
        return max(70, min(100, spo2))
    
    def calculate_hr(self, signal):
        """Simple heart rate detection via peak counting"""
        # Find peaks
        peaks = []
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > threshold:
                    peaks.append(i)
        
        if len(peaks) < 2:
            return 0
            
        # Calculate average interval
        intervals = np.diff(peaks) / self.fs  # in seconds
        avg_interval = np.mean(intervals)
        hr = 60.0 / avg_interval
        return hr

def test_algorithm():
    """Test the pulse oximeter algorithms"""
    sim = PulseOximeterSimulator()
    
    # Generate 10 seconds of data
    t, ir, red = sim.generate_ppg_signal(10)
    
    # Calculate AC and DC components
    ir_dc = np.mean(ir)
    red_dc = np.mean(red)
    ir_ac = np.std(ir)  # Simple approximation
    red_ac = np.std(red)
    
    # Calculate SpO2
    calculated_spo2 = sim.calculate_spo2(ir_ac, red_ac, ir_dc, red_dc)
    print(f"Actual SpO2: {sim.spo2}%")
    print(f"Calculated SpO2: {calculated_spo2:.1f}%")
    print(f"Error: {abs(sim.spo2 - calculated_spo2):.1f}%")
    
    # Calculate heart rate
    calculated_hr = sim.calculate_hr(ir)
    print(f"Actual HR: {sim.heart_rate} BPM")
    print(f"Calculated HR: {calculated_hr:.1f} BPM")
    print(f"Error: {abs(sim.heart_rate - calculated_hr):.1f} BPM")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(t, ir, label='IR Signal', color='red')
    ax1.plot(t, red, label='Red Signal', color='blue', alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('PPG Signals')
    ax1.legend()
    ax1.grid(True)
    
    # Plot spectrum
    freqs = np.fft.fftfreq(len(ir), 1/sim.fs)
    spectrum = np.abs(np.fft.fft(ir))
    
    ax2.plot(freqs[:len(freqs)//2], spectrum[:len(freqs)//2])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Spectrum')
    ax2.set_xlim(0, 5)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()
    
    return calculated_spo2, calculated_hr

if __name__ == "__main__":
    print("=== Pulse Oximeter Test Bench ===\n")
    spo2, hr = test_algorithm()
    
    # Hardware test (if connected)
    try:
        ser = serial.Serial('COM3', 115200, timeout=1)
        print("\nHardware test:")
        for i in range(10):
            ser.write(b'READ\n')
            response = ser.readline().decode().strip()
            if response:
                print(f"  Sample {i+1}: {response}")
            time.sleep(0.1)
        ser.close()
    except:
        print("\nNo hardware connected - simulation only")
