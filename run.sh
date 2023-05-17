docker build -t project:0.1 .
docker run -v "$(pwd)/data":/tmp/data project:0.1

