```
1. Open Terminal and Docker Desktop
2. git clone https://github.com/WrishavSett/OMR
3. cd OMR/
4. cd ./kafkaomr/kafka-stack-docker-compose/
5. docker-compose -f ./full_stack.yml up -d
6. cd ..
7. cd ..
8. docker-compose build
9. docker-compose up -d
10. Update the `localhost` variable in the following files:
    ```
    root/.env
    root/kafkaomr/kafka-stack-docker-compose/.env
