#include <Servo.h>
#include <stdlib.h>

Servo servo1; 
int angle = 91;

void setup() {

Serial.begin(9600);
servo1.attach(11); 

}


void loop() {
  int x = Serial.parseInt();
  if (x != 0) {
    angle = x;
  }
  Serial.println(angle-1);
  servo1.write(angle-1); 
  delay(15); 


}
