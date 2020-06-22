#include <Servo.h>  // Servo kutuphanesini projemize dahil ettik
#include <stdlib.h>

Servo servo1; 
int angle = 0;

void setup() {

Serial.begin(9600);
servo1.attach(11); 

}


void loop() {
  int x = Serial.parseInt();
  if (x != 0) {
    angle = x;
  }
  Serial.println(angle);
  servo1.write(angle); 
  delay(15); 


}
