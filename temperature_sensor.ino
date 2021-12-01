#include <Wire.h>
#include <Servo.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
//Servo x, y;
Servo y;
int width = 1280, height = 720;  // total resolution of the video
//int xpos = 90, ypos = 0;  // initial positions of both Servos
int ypos = 90;

void setup() {
  Serial.begin(9600);
//  Serial.begin(11520);
//  x.attach(9); 
  y.attach(10);
//    y.attach(9);
//  x.write(xpos);
  y.write(ypos);
  Serial.println("Adafruit MLX90614 test");  
  mlx.begin();  
}
const int angle = 2;   // degree of increment or decrement

void loop() {
//  Serial.write(45);
  if (Serial.available() > 0) {
//    if (mlx.readObjectTempC() > 30) {
//      int x_mid, y_mid;
      int y_mid;
//      if (Serial.read() == 'X')
//      {
//        x_mid = Serial.parseInt();  // read center x-coordinate
//        if (Serial.read() == 'Y')
//          y_mid = Serial.parseInt(); // read center y-coordinate
//      }
      if (Serial.read() == 'Y')
        y_mid = Serial.parseInt();
      /* adjust the servo within the squared region if the coordinates
          is outside it
      */
//      if (x_mid > width / 2 + 30)
//        xpos += angle;
//      if (x_mid < width / 2 - 30)
//        xpos -= angle;
      if (y_mid < height / 2 + 100)
        ypos += angle;
        ypos += 20;
      if (y_mid > height / 2 - 100)
        ypos -= angle;
        ypos -= 20;

      // if the servo degree is outside its range
//      if (xpos >= 180)
//        xpos = 180;
//      else if (xpos <= 0)
//        xpos = 0;
      if (ypos >= 180)
        ypos = 180;
      else if (ypos <= 0)
        ypos = 0;

//      x.write(xpos);
      y.write(ypos);

      Serial.print(mlx.readObjectTempC());
//      Serial.println();
//      Serial.print(y_mid);
  }
}
