docker rm -f preem
docker run --rm -it --name preem -p 8888:8888 -e PYTHONPATH=/preem douglasorr/preeminence jupyter notebook --allow-root --port 8888 --ip 0.0.0.0
