#include <Wire.h>

const int num_pins = 5;
const int analogPins[num_pins] = {A0, A1, A2, A3, A4};
const float V = 3.3;
const float Rref = 22000;
float values[num_pins];

void setup() {
  Wire.begin(8);                // I2C slave address = 8
  Wire.onRequest(requestEvent); // called when master requests data
  analogReadResolution(12);
}

void loop() {
  // Nothing needed here
}

// Called automatically when master requests data
void requestEvent() {
  for (int i = 0; i < num_pins; i++) {
    float sensorValue = analogRead(analogPins[i]);
    float voltage = (sensorValue / 4095.0) * V;
    values[i] = (float)((voltage * Rref) / (V - voltage));
  }
  Wire.write((byte*)values, sizeof(values));
}
