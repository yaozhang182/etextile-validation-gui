


const int num_pins = 4;
const int analogPins[num_pins] = {A0, A1, A2, A3};
const float V = 3.3;
const float Rref = 22000;
float resistances_master[num_pins];

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
}

void loop() {
  // Read master's own analog pins
  for (int i = 0; i < num_pins; i++) {
    float sensorValue = analogRead(analogPins[i]);
    float voltage = (sensorValue / 4095.0) * V;
    resistances_master[i] = ((voltage * Rref) / (V - voltage));
  }

  // Print master readings
  for (int i = 0; i < num_pins; i++) {
    Serial.print(resistances_master[i], 2);
    Serial.print(",");
  }

  Serial.println();

  // Delay between readings
  delay(100);
}
