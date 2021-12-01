#include <Wire.h>
#include <Servo.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
Servo y;
int width = 1280, height = 720;  // total resolution of the video
int ypos = 90;

void setup() {
  Serial.begin(9600);
//  Serial.begin(11520);

  y.attach(10);
  y.write(ypos);

  Serial.println("Adafruit MLX90614 test");  
  mlx.begin();  
}
const int angle = 2;   // degree of increment or decrement

void loop() {
  if (Serial.available() > 0) {
      int y_mid;

      if (Serial.read() == 'Y')
        y_mid = Serial.parseInt();

      if (y_mid < height / 2 + 100)
        ypos += angle;
        ypos += 20;
      if (y_mid > height / 2 - 100)
        ypos -= angle;
        ypos -= 20;

      if (ypos >= 180)
        ypos = 180;
      else if (ypos <= 0)
        ypos = 0;

      y.write(ypos);

      Serial.print(mlx.readObjectTempC());
  }
}
