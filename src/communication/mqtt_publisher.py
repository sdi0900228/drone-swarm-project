import paho.mqtt.client as mqtt

client = mqtt.Client()

connected = False

try:
    client.connect("broker.hivemq.com", 1883, 60)
    connected = True
except Exception as e:
    print("MQTT connection failed:", e)

def send_alert(message):
    if connected:
        client.publish("drone/alerts", message)
        print(f"Sent: {message}")
    else:
        print(f"(Offline) {message}")