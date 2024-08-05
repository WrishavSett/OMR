from confluent_kafka import Consumer, KafkaError

# Kafka configuration
conf = {
    'bootstrap.servers': '185.199.53.224:9092',   # Kafka broker(s)
    'group.id': 'my-group',                  # Consumer group id
    'auto.offset.reset': 'earliest'          # Start reading at the earliest available message
}

# Create Consumer instance
consumer = Consumer(conf)

# Define the Kafka topic
TOPIC = 'testtopic'

# Subscribe to the topic
consumer.subscribe([TOPIC])

print(f"Listening to topic: {TOPIC}")

try:
    while True:
        # Poll for a message
        msg = consumer.poll(timeout=1.0)  # Timeout in seconds

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event
                print(f"End of partition reached {msg.partition()}")
            elif msg.error():
                # Error
                print(f"Error: {msg.error()}")
                break
        else:
            # Proper message
            print(f"Received message: {msg.value().decode('utf-8')}")

except KeyboardInterrupt:
    print("Consumer closed.")

finally:
    # Close down consumer to commit final offsets
    consumer.close()
