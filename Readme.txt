First, put npy data into the data folder
 

How to run Docker file
1. docker build -t rene/renetest .
2. docker run rene/renetest 
	Container
	2'. docker run --publish 5000:5000 rene/renetest

Copy from Docker image to host
0. Check Container ID
	docker ps -a
1. docker cp 650932dbb69f:/usr/src/app-name/test_model_AlexNet_epochs_10_batchsize_500.h5 C:\Users\Jun\Documents\StudyingDocker\AIgamingGDA
or
1. docker cp C:\Users\Jun\Documents\StudyingDocker\AIgamingGDA\test_model_AlexNet_epochs_10_batchsize_500.h5 650932dbb69f:/usr/src/app-name/

How to remove container and image.
1. docker stop $(docker ps -a -q)
2. docker rm $(docker ps -a -q)
3. docker rmi $(docker images -q)


