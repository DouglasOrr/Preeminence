SET PREEM_NOTEBOOK=%USERNAME%-preem

docker pull douglasorr/preeminence
docker rm -f %PREEM_NOTEBOOK%
docker run -d --name %PREEM_NOTEBOOK% -p 8888:8888 -e PYTHONPATH=/preem -v %cd%:/preem -w /preem douglasorr/preeminence jupyter notebook --allow-root --port 8888 --ip 0.0.0.0

ping 127.0.0.1 -n 3 > nul
docker logs %PREEM_NOTEBOOK%
ECHO "Your notebook is running at http://localhost:8888/notebooks/Tutorial.ipynb ; see login token listed above"
ECHO "When you want to stop your server, run: docker rm -f %PREEM_NOTEBOOK%"
