#include "Wire.h"
#include <MPU6050_light.h>
#include "BluetoothSerial.h"

MPU6050 mpu(Wire);
BluetoothSerial SerialBT;

unsigned long timer = 0;

// Cette variable sert de "mémoire"
bool estEnAlerte = false; 

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_Gyro"); 
  Serial.println("Bluetooth démarré !");

  Wire.begin(21, 22);
  
  byte status = mpu.begin();
  while(status != 0){ 
    Serial.println("Echec MPU6050");
    delay(1000); 
  }

  Serial.println("Calibration... Ne bougez pas !");
  delay(1000);
  mpu.calcOffsets();
  Serial.println("Pret !");
}

void loop() {
  mpu.update();

  if((millis() - timer) > 50) { // Vérification rapide (50ms)
    
    float angleX = mpu.getAngleX();
    
    // --- LOGIQUE DE DÉCLENCHEMENT ---
    
    // CAS 1 : L'objet est penché (Hors de la zone -5 à 5)
    if (abs(angleX) > 5) {
      
      // On envoie le message SEULEMENT si on n'est pas déjà en alerte
      if (estEnAlerte == false) {
        SerialBT.println("TRUE");      // Envoi Bluetooth unique
        Serial.println(">>> ALERTE ENVOYÉE"); // Debug USB
        
        estEnAlerte = true;            // ON VERROUILLE : On note qu'on est en alerte
      }
      // Si estEnAlerte est déjà true, on ne fait RIEN (on ignore les autres mesures)
    } 
    
    // CAS 2 : L'objet est revenu à plat (Dans la zone -5 à 5)
    else {
      // Si on était en alerte, on "réarme" le système
      if (estEnAlerte == true) {
        estEnAlerte = false;           // ON DÉVERROUILLE
        Serial.println("<<< Système réarmé (Retour position initiale)");
        delay(2000);
      }
    }
    
    timer = millis(); 
  }
}
